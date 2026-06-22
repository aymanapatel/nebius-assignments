import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


"""
Resource: [Megatron-LM paper: section 3. Model Parallel](https://arxiv.org/pdf/1909.08053)
"""
class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        y = x.clone()
        dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class TransformerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ColumnParallelLinear(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            tp_mesh: dist.DeviceMesh,
        ) -> None:
        super().__init__()
        self.tp_group = tp_mesh.get_group()
        self.tp_world_size = dist.get_world_size(self.tp_group)
        assert output_size % self.tp_world_size == 0
        self.weight = nn.Parameter(torch.empty(output_size // self.tp_world_size, input_size))
        assert self.weight.data.shape[1] == input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTensorParallelRegion.apply(x, self.tp_group)
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            tp_mesh: dist.DeviceMesh,
        ) -> None:
        super().__init__()
        self.tp_group = tp_mesh.get_group()
        self.tp_world_size = dist.get_world_size(self.tp_group)
        assert input_size % self.tp_world_size == 0
        self.weight = nn.Parameter(torch.empty(output_size, input_size // self.tp_world_size))
        assert isinstance(self.weight, torch.Tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.weight)
        return _ReduceFromTensorParallelRegion.apply(x, self.tp_group)


class ParallelTransformerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_mesh: dist.DeviceMesh,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, tp_mesh)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(intermediate_size, hidden_size, tp_mesh)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
