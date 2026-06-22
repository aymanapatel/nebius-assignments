"""
Pipeline parallelism partitions a model along its depth dimension, typically by assigning the consecutive groups of layers to different devices. Compared other training parallelisms, PP requirements to the network bandwidth are typically minimal, as we only send a portion of activations from one device to another! However, pipelining may introduce some idle time when devices are not utilized, called bubbles. The main goal of pipeline-parallel algorithms is to minimise pipeline bubbles: periods when some stages are idle.

In this puzzle, you will implement a [ZB-H2-style Zero Bubble Pipeline](https://arxiv.org/abs/2401.10241) schedule from the paper. Zero Bubble pipelining splits the backward computation for each microbatch into two parts:
* Gradient with respect to the input;
* Gradient with respect to the weights -- and this stage is not on the critical path, so it may be delayed.

We'll rely on some idealized assumptions:
1. `forward time (F) == backward weight time (W) == backward input time (B)`, communication time is ignored.
2. $m \geqslant 2n$, where $n$ is the number of processes, and $m$ is the number of microbatches.

It turns out that in this case you have the optimal schedule in closed form! *(there are actually a lot of possible solutions correct under those assumptions)*
"""
def zb_h2_schedule(n: int, m: int) -> list[list[str]]:
    """
    Return Zero Bubble ZB-H2 schedule as rows of tokens.
    ----
    inputs:
        n: number of devices
        m: number of microbatches

    returns:
        out: list n timeline rows. out[i][j] is the operation run by stage i
        at time j. each operation is a string like: "F1", "B2", "W123",
        or "" for idle compute.
        Operations meanings:
          "F{j}": forward computation for microbatch j
          "B{j}": backward-input computation for microbatch j
          "W{j}": backward-weight computation for microbatch j
          "": idle compute

    """
    if m < 2 * n:
        raise ValueError("This closed-form ZB-H2 schedule assumes m >= 2n")

    schedule = [[""] * stage for stage in range(n)]
    done: set[tuple[int, str, int]] = set()
    next_f = [1] * n
    next_b = [1] * n

    for time in range(3 * m + n - 1):
        for stage in range(n):
            if not stage <= time < stage + 3 * m:
                continue

            op = None
            mb = None

            i = next_b[stage]
            if (
                i <= m
                and (stage, "F", i) in done
                and (stage == n - 1 or (stage + 1, "B", i) in done)
            ):
                op, mb = "B", i
                next_b[stage] += 1
            else:
                i = next_f[stage]
                if i <= m and (stage == 0 or (stage - 1, "F", i) in done):
                    op, mb = "F", i
                    next_f[stage] += 1
                else:
                    for i in range(m, 0, -1):
                        if (stage, "B", i) in done and (stage, "W", i) not in done:
                            op, mb = "W", i
                            break

            assert op is not None and mb is not None
            schedule[stage].append(f"{op}{mb}")
            done.add((stage, op, mb))

    return schedule
