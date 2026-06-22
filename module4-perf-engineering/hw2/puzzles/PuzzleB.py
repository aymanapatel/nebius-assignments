import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            tp_mesh: dist.DeviceMesh,
        ) -> None:
        super().__init__()
        # Your code is here
        assert self.weight.data.shape[1] == input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code is here
        pass

class RowParallelLinear(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            tp_mesh: dist.DeviceMesh,
        ) -> None:
        super().__init__()
        # Your code is here
        raise NotImplementedError
        assert isinstance(self.weight, torch.Tensor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code is here
        raise NotImplementedError


class ParallelTransformerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_mesh: dist.DeviceMesh,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = None # Your code is here
        self.act = nn.GELU()
        self.fc2 = None # Your code is here
        self.dropout = nn.Dropout(dropout)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code is here
        raise NotImplementedError