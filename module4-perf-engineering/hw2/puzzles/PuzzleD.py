import torch
import torch.distributed as dist

# start_kv_rotate
def start_kv_rotate(
    k: torch.Tensor,
    v: torch.Tensor,
    k_recv: torch.Tensor,
    v_recv: torch.Tensor,
    src_rank: int,
    dst_rank: int,
    pg: dist.ProcessGroup,
) -> list[dist.Work]:
    """
    Start one async KV ring-rotation step.
    Each rank sends its current (k, v) to dst_rank and receives next (k, v) from
    src_rank into (k_recv, v_recv).
    """
    # Your code is here: use `dist.batch_isend_irecv`, `dist.P2POp`,
    # `dist.isend`, and `dist.irecv`
    raise NotImplementedError


def wait_all(reqs: list[dist.Work]) -> None:
    """Wait for all async dist requests to complete"""
    # Your code is here
    raise NotImplementedError


# Ring attention

import math
import torch
import torch.distributed as dist

def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pg: dist.ProcessGroup,
):
    """
    distributed exact unmasked self-attention over a context
    sharded across pg group.
    ---
    inputs:
        q, k, v: [b, h, s_local, d]
    outputs:
        a: [b, h, s_local, d]
    ----
    * b: batch size
    * h: number of heads
    * s_local: local sequence length (s_local == s_global / world_size)
    * d: head dimension
    """
    rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    src_rank = (rank - 1 + world_size) % world_size
    dst_rank = (rank + 1) % world_size

    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    # Your code is here: initialize the buffers

    for step in range(world_size):
        # Your code is here: schedule the comms (if necessary)

        # Your code is here: update the running statistics

        # Your code is here: wait for the comms and update data (if necessary)

        raise NotImplementedError

    # Your code is here: return the result