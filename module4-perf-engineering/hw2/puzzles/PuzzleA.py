import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def ring_allreduce(x: torch.Tensor, pg: dist.ProcessGroup):
    group_rank = dist.get_rank(group=pg)
    group_size = dist.get_world_size(group=pg)

    if group_size == 1:
        return

    flat = x.view(-1)
    chunk_size = flat.numel() // group_size
    chunks = list(flat.split(chunk_size))

    ranks = dist.get_process_group_ranks(pg)
    prev_rank = ranks[(group_rank - 1) % group_size]
    next_rank = ranks[(group_rank + 1) % group_size]

    for step in range(group_size - 1):
        send_idx = (group_rank - step) % group_size
        recv_idx = (group_rank - step - 1) % group_size
        recv_buf = torch.empty_like(chunks[recv_idx])

        work = dist.isend(chunks[send_idx], dst=next_rank)
        dist.recv(recv_buf, src=prev_rank)
        work.wait()

        chunks[recv_idx].add_(recv_buf)

    for step in range(group_size - 1):
        send_idx = (group_rank - step + 1) % group_size
        recv_idx = (group_rank - step) % group_size

        work = dist.isend(chunks[send_idx], dst=next_rank)
        dist.recv(chunks[recv_idx], src=prev_rank)
        work.wait()


def main():
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    rank = dist.get_rank()
    x = (torch.arange(4) + rank * 4).float()
    print(f"[rank {rank}] allreduce input: {x}", flush=True)

    pg = mesh.get_group("dp")
    ring_allreduce(x, pg)

    print(f"[rank {rank}] allreduce output: {x}", flush=True)

if __name__ == "__main__":
    main()