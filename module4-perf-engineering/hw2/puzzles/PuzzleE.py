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

    schedule = []

    for stage in range(n):
        row = [""] * stage

        # Your code is here

        schedule.append(row)

    return schedule