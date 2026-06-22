import torch
import torch.distributed as dist

"""
Modern LLMs may support extremely long contexts! However, fitting that context into the devices introduces a bunch of engineering challenges. In this puzzle, you will implement one of the very elegant algorithms for long context training: [Ring Attention](https://arxiv.org/abs/2310.01889).

Learning resources breaking down the ring attention algorithm besides the original paper:
* [GPU MODE Ring Attention Lecture](https://youtu.be/ws7angQYIxI);
* [Ultra-Long Sequence Parallelism](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention);
* [Ring Attention Explained](https://coconut-mode.com/posts/ring-attention/).

Some simplifications:
* Compute bidirectional attention (no attention mask);
* Assume the full sequence length is divisible by the world size;
* We only implement Ring Attention on its own, without QKV linear projections, output projection, or tensor-parallel comms.
* We only implement on Ring Attention forward.

Some hints:
* Remember about the scale $\sqrt{d}$ and the batched computation. That is, the actual attention formula:
$$A=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V,\quad Q,K,V,A\in\mathbb{R}^{b\times h\times s\times d}$$
* Use `dist.batch_isend_irecv` in `start_kv_rotate` to schedule all p2p comms as a single collective operation.
"""

def _global_rank(group_rank: int, pg: dist.ProcessGroup) -> int:
    return dist.get_process_group_ranks(pg)[group_rank]

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
    src = _global_rank(src_rank, pg)
    dst = _global_rank(dst_rank, pg)
    ops = [
        dist.P2POp(dist.isend, k.contiguous(), dst, pg, 0),
        dist.P2POp(dist.isend, v.contiguous(), dst, pg, 1),
        dist.P2POp(dist.irecv, k_recv, src, pg, 0),
        dist.P2POp(dist.irecv, v_recv, src, pg, 1),
    ]
    return dist.batch_isend_irecv(ops)


def wait_all(reqs: list[dist.Work]) -> None:
    """Wait for all async dist requests to complete"""
    for req in reqs:
        req.wait()


"""
Now, implement the Ring Attention algorithm using the [Online Softmax idea](https://arxiv.org/abs/1805.02867).

Hints:
* You follow the approach from the lectures (slide 58) -- storing the unnormalized numerator and denominator separately -- it would make the code more concise.
* Alternatively, you may implement inner loop as in Flash Attention ([paper](https://arxiv.org/pdf/2205.14135) -- see `Algorithm 1`) -- this may require slightly more code, but would update the result online instead of keeping unnormalized numerator.
* Reminder on attention simplifications: forward-only, bidirectional (unmasked).
"""        
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

    q_work = q.float()
    k_cur = k
    v_cur = v
    k_next = k_recv
    v_next = v_recv

    row_max = torch.full(q.shape[:-1] + (1,), -torch.inf, dtype=torch.float32, device=q.device)
    denom = torch.zeros_like(row_max)
    numer = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(q.shape[-1])

    for step in range(world_size):
        reqs = []
        if step + 1 < world_size:
            reqs = start_kv_rotate(k_cur, v_cur, k_next, v_next, src_rank, dst_rank, pg)

        scores = torch.matmul(q_work, k_cur.float().transpose(-2, -1)) * scale
        block_max = scores.max(dim=-1, keepdim=True).values
        new_row_max = torch.maximum(row_max, block_max)

        old_scale = torch.exp(row_max - new_row_max)
        probs = torch.exp(scores - new_row_max)
        numer = numer * old_scale + torch.matmul(probs, v_cur.float())
        denom = denom * old_scale + probs.sum(dim=-1, keepdim=True)
        row_max = new_row_max

        if step + 1 < world_size:
            wait_all(reqs)
            k_cur, k_next = k_next, k_cur
            v_cur, v_next = v_next, v_cur

    return (numer / denom).to(q.dtype)