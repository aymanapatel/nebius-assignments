import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

""" Ulysses
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiffAllToAll(torch.autograd.Function):
    """
    Differentiable all-to-all (across dim 0).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        assert x.shape[0] == dist.get_world_size(group)
        ctx.group = group
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        dist.all_to_all_single(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_x = torch.empty(grad_out.shape, dtype=grad_out.dtype, device=grad_out.device)
        dist.all_to_all_single(grad_x, grad_out.contiguous(), group=ctx.group)
        return grad_x, None


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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2)

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

        qkv = self.qkv(x).view(
            batch,
            seq_local,
            3,
            self.num_heads,
            self.head_dim,
        )
        heads_per_rank = self.num_heads // self.sp_size
        qkv = rearrange(
            qkv,
            "b s three (p h) d -> p b s three h d",
            p=self.sp_size,
            h=heads_per_rank,
        )
        qkv = DiffAllToAll.apply(qkv.contiguous(), self.group)
        qkv = rearrange(qkv, "p b s three h d -> b (p s) three h d")
        q, k, v = qkv.unbind(2)

        out = self.local_attention(q, k, v)
        out = self.head_par_to_seq_par(out)
        out = out.reshape(batch, seq_local, hidden_size)
        return self.out_proj(out)

    def seq_par_to_head_par(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq / sp_size, n_heads, head_size]
        Returns:
            out: [batch, seq, n_heads / sp_size, head_size]
        """
        heads_per_rank = self.num_heads // self.sp_size
        x = rearrange(x, "b s (p h) d -> p b s h d", p=self.sp_size, h=heads_per_rank)
        x = DiffAllToAll.apply(x.contiguous(), self.group)
        return rearrange(x, "p b s h d -> b (p s) h d")

    def head_par_to_seq_par(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, n_heads / sp_size, head_size]
        Returns:
            out: [batch, seq / sp_size, n_heads, head_size]
        """
        seq_local = x.shape[1] // self.sp_size
        x = rearrange(x, "b (p s) h d -> p b s h d", p=self.sp_size, s=seq_local)
        x = DiffAllToAll.apply(x.contiguous(), self.group)
        return rearrange(x, "p b s h d -> b s (p h) d")
