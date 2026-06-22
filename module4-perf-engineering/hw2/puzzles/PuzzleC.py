import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange


class DiffAllToAll(torch.autograd.Function):
    """
    Differentiable all-to-all (across dim 0).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        assert x.shape[0] == dist.get_world_size(group)
        # Your code is here
        pass

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Your code is here
        pass


class UlyssesSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mesh: dist.DeviceMesh,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group = mesh.get_group("sp")
        self.sp_size = dist.get_world_size(self.group)
        assert num_heads % self.sp_size == 0

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: [batch, seq, n_heads / sp_size, head_size]
        Returns:
            out: : [batch, seq, n_heads / sp_size, head_size]
        """
        # Your code is here
        # Hint: Use F.scaled_dot_product_attention
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq / sp_size, hidden_size]
        Returns:
            out: [batch, seq / sp_size, hidden_size]
        """
        batch, seq_local, hidden_size = x.shape
        # Your code is here:
        # 1. QKV projections
        # 2. QKV all2alls: SP into HP:
        # 3. Exact attention on full sequence, but only for the local rank's heads
        # 4. Reverse all2all: HP into SP:
        # 5. Reshape & the final projection
        pass

    def seq_par_to_head_par(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq / sp_size, n_heads, head_size]
        Returns:
            out: [batch, seq, n_heads / sp_size, head_size]
        """
        # Your code is here
        pass

    def head_par_to_seq_par(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, n_heads / sp_size, head_size]
        Returns:
            out: [batch, seq / sp_size, n_heads, head_size]
        """
        # Your code is here
        pass