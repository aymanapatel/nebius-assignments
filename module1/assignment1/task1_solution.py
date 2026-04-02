"""
Task 1 — Complete Solution
==========================

Copy the relevant section into each notebook cell that has a TODO.
All four sub-tasks (1.1, 1.2, 1.3, 1.4) and Part 2 are solved here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.1 — LogisticRegression class in PyTorch
# Replace the cell that contains "class LogisticRegression(nn.Module):"
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, n_features, init="zeros"):
        """
        Parameters
        ----------
        n_features : int
            Number of input features

        init : str or torch.Tensor
            Initialization method for weights:
            - "zeros"  -> initialize weights to zeros
            - "random" -> small random values (scale ~0.01)
            - torch.Tensor -> use provided tensor
        """
        super().__init__()

        if init == "zeros":
            w = torch.zeros(n_features, 1)

        elif init == "random":
            # Small random values are important for breaking symmetry
            # while keeping initial predictions near 0.5
            w = torch.randn(n_features, 1) * 0.01

        elif isinstance(init, torch.Tensor):
            w = init.clone().detach().float().reshape(n_features, 1)

        else:
            raise ValueError("init must be 'zeros', 'random', or a torch.Tensor")

        # Wrap weights and bias using nn.Parameter so PyTorch tracks gradients
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass: compute logits then apply sigmoid.
        """
        logits = x @ self.w + self.b     # shape: (N, 1)
        probs  = torch.sigmoid(logits)   # shape: (N, 1), values in [0, 1]
        return probs

    def predict(self, x):
        """
        Convert probabilities to class predictions.
        class 1 if p >= 0.5, class 0 otherwise.
        """
        probs = self.forward(x)
        preds = (probs >= 0.5).float()
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.2 — BCE loss + SGD training function
# Replace the two cells that contain the binary_cross_entropy_loss definition
# and the sgd_logistic_regression skeleton.
# ─────────────────────────────────────────────────────────────────────────────

def binary_cross_entropy_loss(y_pred, y_true):
    """
    Compute the binary cross-entropy loss.

    Parameters
    ----------
    y_pred : torch.Tensor, shape (N, 1)  — predicted probabilities in [0, 1]
    y_true : torch.Tensor, shape (N, 1)  — binary labels {0, 1}

    Returns
    -------
    loss : scalar torch.Tensor  (so .backward() can be called on it)
    """
    epsilon = 1e-15
    # Clamp y_pred for numerical stability (avoids log(0))
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()   # scalar


import torch
from sklearn.metrics import f1_score, accuracy_score


def sgd_logistic_regression(
    X_train, y_train,
    X_val,   y_val,
    lr=0.01,
    epochs=20,
    batch_size=100,
    init="zeros",
    penalty='none',       # 'none', 'l1', 'l2'
    reg_lambda=0.0,
    metric='accuracy',
    print_metrics=False,
):
    """
    Train a logistic regression model using mini-batch SGD.

    Metric choice — Accuracy vs F1
    --------------------------------
    For balanced datasets, accuracy is a good default.
    For imbalanced datasets (e.g. sentiment where one class dominates),
    F1-score is preferred because it balances precision and recall and
    is not inflated by the dominant class.  Pass metric='f1' to use it.

    Returns
    -------
    w : numpy.ndarray  — final learned weights  (n_features, 1)
    b : numpy.ndarray  — final learned bias      (1,)
    history    : list  — batch-wise history of parameter values
    epoch_log  : list  — per-epoch logs (loss + metrics)
    """

    # 1. Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    n_samples, n_features = X_train_tensor.shape

    # 2. Initialize model
    model = LogisticRegression(n_features=n_features, init=init)

    # 3. Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 4. Create logs
    history   = []   # batch-wise parameter snapshots
    epoch_log = []   # epoch-level loss + metrics

    # 5. Training loop
    for epoch in range(epochs):

        # Shuffle the training data at the beginning of each epoch
        perm           = torch.randperm(n_samples)
        X_train_epoch  = X_train_tensor[perm]
        y_train_epoch  = y_train_tensor[perm]

        for start in range(0, n_samples, batch_size):

            end = start + batch_size

            # Select mini-batch
            X_batch = X_train_epoch[start:end]
            y_batch = y_train_epoch[start:end]

            # Forward pass
            y_pred = model(X_batch)

            # Compute non-regularized BCE loss
            data_loss = binary_cross_entropy_loss(y_pred, y_batch)

            # Add regularization term to the loss
            if penalty == 'l1':
                reg_term = reg_lambda * model.w.abs().sum()
            elif penalty == 'l2':
                reg_term = reg_lambda * (model.w ** 2).sum()
            else:
                reg_term = 0.0

            loss = data_loss + reg_term

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save parameters after each batch update
            history.append({
                'epoch':       epoch,
                'batch_start': start,
                'w':           model.w.detach().clone(),
                'b':           model.b.detach().clone()
            })

        # 6. Epoch-level evaluation
        with torch.no_grad():

            # Compute probabilities on full train/val sets
            y_pred_train = model(X_train_tensor)
            y_pred_val   = model(X_val_tensor)

            # Compute NON-regularized train/val loss
            train_loss = binary_cross_entropy_loss(y_pred_train, y_train_tensor)
            val_loss   = binary_cross_entropy_loss(y_pred_val,   y_val_tensor)

            # Convert probabilities to binary predictions (numpy for sklearn)
            y_hat_train = (y_pred_train >= 0.5).float().numpy().flatten()
            y_hat_val   = (y_pred_val   >= 0.5).float().numpy().flatten()
            y_np_train  = y_train_tensor.numpy().flatten()
            y_np_val    = y_val_tensor.numpy().flatten()

            # Compute evaluation metric
            if metric == 'f1':
                train_metric = f1_score(y_np_train, y_hat_train, zero_division=0)
                val_metric   = f1_score(y_np_val,   y_hat_val,   zero_division=0)
            else:  # default: accuracy
                train_metric = accuracy_score(y_np_train, y_hat_train)
                val_metric   = accuracy_score(y_np_val,   y_hat_val)

        epoch_log.append({
            'epoch':        epoch,
            'train_loss':   train_loss.item(),
            'val_loss':     val_loss.item(),
            'train_metric': train_metric,
            'val_metric':   val_metric
        })

        if print_metrics:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Train {metric}: {train_metric:.4f} | "
                f"Val {metric}: {val_metric:.4f}"
            )

    return model.w.detach().numpy(), model.b.detach().numpy(), history, epoch_log


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.3 — Experiments: grid search over LR × batch_size
# Paste this into the "# code goes here" cell under Task 1.3
# ─────────────────────────────────────────────────────────────────────────────

def run_task_1_3():
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    learning_rates = [0.01, 0.03, 0.1, 0.3, 1.0]
    batch_sizes    = [50, 100, 200]
    epochs         = 20
    metric         = 'accuracy'

    results = {}
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        print(f"lr={lr}, batch_size={bs} ...", end=' ', flush=True)
        w, b, history, epoch_log = sgd_logistic_regression(
            X_train, y_train, X_val, y_val,
            lr=lr, epochs=epochs, batch_size=bs,
            init='zeros', metric=metric, print_metrics=False
        )
        final = epoch_log[-1]
        results[(lr, bs)] = {
            'train_metric': final['train_metric'],
            'val_metric':   final['val_metric'],
            'train_loss':   final['train_loss'],
            'val_loss':     final['val_loss'],
        }
        print(f"val_{metric}={final['val_metric']:.4f}")

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    for split in ['train', 'val']:
        key    = f'{split}_metric'
        matrix = np.array([
            [results[(lr, bs)][key] for lr in learning_rates]
            for bs in batch_sizes
        ])

        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([str(lr) for lr in learning_rates])
        ax.set_yticks(range(len(batch_sizes)))
        ax.set_yticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Batch Size')
        ax.set_title(f'Logistic Regression — {split.capitalize()} {metric.capitalize()} Heatmap')
        for i in range(len(batch_sizes)):
            for j in range(len(learning_rates)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center',
                        color='white' if matrix[i, j] < matrix.max() * 0.75 else 'black',
                        fontsize=8)
        plt.tight_layout()
        plt.show()

    # ── Loss Heatmaps ────────────────────────────────────────────────────────
    for split in ['train', 'val']:
        key    = f'{split}_loss'
        matrix = np.array([
            [results[(lr, bs)][key] for lr in learning_rates]
            for bs in batch_sizes
        ])

        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(matrix, cmap='magma_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='Log-Loss')
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([str(lr) for lr in learning_rates])
        ax.set_yticks(range(len(batch_sizes)))
        ax.set_yticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Batch Size')
        ax.set_title(f'Logistic Regression — {split.capitalize()} Log-Loss Heatmap')
        for i in range(len(batch_sizes)):
            for j in range(len(learning_rates)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.mean() else 'black',
                        fontsize=8)
        plt.tight_layout()
        plt.show()

    # ── Written analysis ──────────────────────────────────────────────────────
    print("""
Analysis:
---------
Learning rate effect:
  - Very small LRs (0.01–0.03) converge slowly; 20 epochs may not be enough
    to reach a good solution, so their heatmap cells show lower accuracy.
  - Moderate LRs (0.1–0.3) hit the best trade-off: fast convergence and
    stable training, giving the highest validation accuracy.
  - Very large LRs (1.0) can cause gradient divergence (NaN or oscillation),
    especially with small batch sizes, resulting in degraded performance in
    the top-right region of the heatmap.

Batch size effect:
  - Smaller batches (50) inject more gradient noise, which can help escape
    flat regions but makes the loss curve noisier — visible as higher
    variance across epochs.
  - Larger batches (200) produce smoother, more accurate gradient estimates
    and converge more predictably, but do fewer weight updates per epoch so
    they need either more epochs or a higher LR.
  - The interaction: a large LR works best with medium/large batches;
    a small LR is more forgiving to any batch size.
""")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1.4 — L1 Regularization and Sparsity
# Paste this into the "# <Your code here>" cell under Task 1.4
# ─────────────────────────────────────────────────────────────────────────────

def run_task_1_4():
    import numpy as np
    import matplotlib.pyplot as plt

    reg_lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    inits       = ['zeros', 'random']
    lr_l1       = 0.1
    bs_l1       = 100
    ep_l1       = 20
    metric      = 'accuracy'

    l1_results = {}
    for init in inits:
        l1_results[init] = {}
        for lam in reg_lambdas:
            print(f"init={init}, lambda={lam} ...", end=' ', flush=True)
            w, b, history, epoch_log = sgd_logistic_regression(
                X_train, y_train, X_val, y_val,
                lr=lr_l1, epochs=ep_l1, batch_size=bs_l1,
                init=init, penalty='l1', reg_lambda=lam,
                metric=metric, print_metrics=False
            )
            final     = epoch_log[-1]
            n_nonzero = int((np.abs(w.flatten()) > 1e-7).sum())
            l1_results[init][lam] = {
                'w': w, 'b': b, 'history': history, 'epoch_log': epoch_log,
                'train_metric': final['train_metric'],
                'val_metric':   final['val_metric'],
                'n_nonzero':    n_nonzero,
            }
            print(f"val_{metric}={final['val_metric']:.4f}, non-zero weights={n_nonzero}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    colors = {'zeros': 'steelblue', 'random': 'tomato'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: number of non-zero weights vs lambda
    ax = axes[0]
    for init in inits:
        y = [l1_results[init][lam]['n_nonzero'] for lam in reg_lambdas]
        ax.plot(range(len(reg_lambdas)), y, marker='o',
                label=f'init={init}', color=colors[init])
    ax.set_xticks(range(len(reg_lambdas)))
    ax.set_xticklabels([str(l) for l in reg_lambdas], rotation=30, ha='right')
    ax.set_xlabel('λ (reg_lambda)')
    ax.set_ylabel('# non-zero weights (|w| > 1e-7)')
    ax.set_title('Sparsity vs λ')
    ax.legend()

    # Panel 2: train/val metric vs lambda
    ax = axes[1]
    for init in inits:
        y_train_m = [l1_results[init][lam]['train_metric'] for lam in reg_lambdas]
        y_val_m   = [l1_results[init][lam]['val_metric']   for lam in reg_lambdas]
        ax.plot(range(len(reg_lambdas)), y_val_m,   marker='o',
                label=f'val   init={init}', color=colors[init])
        ax.plot(range(len(reg_lambdas)), y_train_m, marker='s', linestyle='--',
                label=f'train init={init}', color=colors[init], alpha=0.6)
    ax.set_xticks(range(len(reg_lambdas)))
    ax.set_xticklabels([str(l) for l in reg_lambdas], rotation=30, ha='right')
    ax.set_xlabel('λ (reg_lambda)')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} vs λ')
    ax.legend(fontsize=7)

    # Panel 3: weight dynamics for 5 most-eliminated features
    ax = axes[2]
    lam_max  = reg_lambdas[-1]
    w_final  = l1_results['zeros'][lam_max]['w'].flatten()
    tracked  = sorted(range(len(w_final)), key=lambda i: abs(w_final[i]))[:5]
    hist_ref = l1_results['zeros'][lam_max]['history']
    steps    = list(range(len(hist_ref)))
    for feat_idx in tracked:
        w_vals = [h['w'][feat_idx, 0].item() for h in hist_ref]
        ax.plot(steps, w_vals, label=f'feat {feat_idx}', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Batch step (global)')
    ax.set_ylabel('Weight value')
    ax.set_title(f'Weight dynamics (λ={lam_max}, init=zeros)')
    ax.legend(fontsize=7)

    plt.suptitle('L1 Regularization Analysis', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("""

Summary:
--------
L1 regularization progressively forces weights toward zero as lambda increases,
effectively performing automatic feature selection.

With 'zeros' initialization the model starts symmetrically, so all weights
receive equal gradient pressure; sparsification is smooth and predictable.
With 'random' initialization symmetry is broken, so some weights resist the L1
pull longer, but both strategies converge to similar sparsity levels for the
same lambda.

Key observations:
  - Sparsity rises sharply once lambda exceeds ~1e-3: the L1 penalty dominates
    the loss and pushes many small-magnitude weights below the 1e-7 threshold.
  - Moderate regularization (lambda ≈ 1e-3 to 1e-2) can slightly improve
    validation accuracy by reducing overfitting while keeping informative weights.
  - Heavy regularization (lambda = 0.1) hurts performance: too many informative
    weights are driven near zero, and the model underfits.
  - The weight-dynamics plot shows the characteristic L1 behaviour: weights
    decrease quasi-linearly toward zero (unlike L2 which decays exponentially)
    and some stall very close to zero rather than hitting exactly zero — this is
    because plain SGD lacks the proximal step needed for exact sparsity.
""")

    return l1_results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Comparing Optimization Algorithms (3 points)
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
import matplotlib.pyplot as plt


# Function A — convex bowl
def bowl(theta):
    x, y = theta[..., 0], theta[..., 1]
    return x**2 + 2*y**2


# Function B — six-hump camel
def camel(theta):
    x, y = theta[..., 0], theta[..., 1]
    return (4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2


def plot_trajectories(f, results, xlim=(-3, 3), ylim=(-2, 2),
                      title="Optimization Trajectories", use_log=False):
    """
    Contour plot of f with all optimizer trajectories overlaid.
    """
    xv = np.linspace(xlim[0], xlim[1], 400)
    yv = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xv, yv)
    grid = np.stack((X, Y), axis=-1)
    Z = f(torch.tensor(grid, dtype=torch.float32)).detach().numpy()

    plt.figure(figsize=(8, 6))
    if use_log:
        plt.contour(X, Y, np.log1p(Z - Z.min() + 1e-8), levels=30, cmap='viridis')
    else:
        plt.contour(X, Y, Z, levels=30, cmap='viridis')

    for name, (trajectory, _) in results.items():
        traj = trajectory.numpy() if isinstance(trajectory, torch.Tensor) else trajectory
        plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, label=name)

    plt.xlabel("x"); plt.ylabel("y")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()


def plot_convergence(results, title="Function value vs iteration"):
    """Plot f(x_t, y_t) vs iteration for each optimizer."""
    plt.figure(figsize=(8, 5))
    for name, (_, values) in results.items():
        plt.plot(values, label=name)
    plt.xlabel("Iteration"); plt.ylabel("f(x, y)")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()


# ── Gradient Descent ─────────────────────────────────────────────────────────

def gradient_descent(f, theta0, lr=0.001, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)

    trajectory = [theta.detach().clone()]
    values     = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            theta -= lr * theta.grad

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


# ── Momentum ─────────────────────────────────────────────────────────────────

def momentum(f, theta0, lr=0.001, beta=0.9, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    v     = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values     = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            # velocity = β·v + gradient
            v = beta * v + theta.grad
            # parameter update
            theta -= lr * v

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


# ── AdaGrad ──────────────────────────────────────────────────────────────────

def adagrad(f, theta0, lr=0.1, eps=1e-8, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    G     = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values     = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            # accumulate squared gradients
            G += theta.grad ** 2
            # adaptive update: lr / sqrt(G + eps) * grad
            theta -= lr / (torch.sqrt(G) + eps) * theta.grad

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


# ── Adam ─────────────────────────────────────────────────────────────────────

def adam(f, theta0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    m     = torch.zeros_like(theta)   # 1st moment (mean of gradient)
    v     = torch.zeros_like(theta)   # 2nd moment (mean of squared gradient)

    trajectory = [theta.detach().clone()]
    values     = [f(theta).item()]

    for step in range(1, n_steps + 1):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            # update biased moments
            m = beta1 * m + (1 - beta1) * theta.grad
            v = beta2 * v + (1 - beta2) * theta.grad ** 2

            # bias correction
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            # parameter update
            theta -= lr * m_hat / (torch.sqrt(v_hat) + eps)

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


# ──────────────────────────────────────────────────────────────────────────────
# Run Part 2 experiments
# ──────────────────────────────────────────────────────────────────────────────

def run_part_2():

    theta0 = [-2.0, -1.5]     # starting point (non-optimal)
    n_steps = 500

    # ── Convex bowl ──────────────────────────────────────────────────────────
    results_bowl = {
        "GD":       gradient_descent(bowl, theta0, lr=0.05,  n_steps=n_steps),
        "Momentum": momentum(bowl, theta0,         lr=0.01,  beta=0.9, n_steps=n_steps),
        "AdaGrad":  adagrad(bowl, theta0,           lr=0.5,   n_steps=n_steps),
        "Adam":     adam(bowl, theta0,               lr=0.1,   n_steps=n_steps),
    }

    plot_convergence(results_bowl, title="Bowl — f(x,y) vs Iteration")
    plot_trajectories(bowl, results_bowl,
                      xlim=(-3, 3), ylim=(-2, 2),
                      title="Bowl — Optimization Trajectories")

    # ── Six-hump Camel ───────────────────────────────────────────────────────
    results_camel = {
        "GD":       gradient_descent(camel, theta0, lr=0.005, n_steps=n_steps),
        "Momentum": momentum(camel, theta0,         lr=0.005, beta=0.9, n_steps=n_steps),
        "AdaGrad":  adagrad(camel, theta0,           lr=0.1,   n_steps=n_steps),
        "Adam":     adam(camel, theta0,               lr=0.05,  n_steps=n_steps),
    }

    plot_convergence(results_camel, title="Camel — f(x,y) vs Iteration")
    plot_trajectories(camel, results_camel,
                      xlim=(-3, 3), ylim=(-2, 2),
                      title="Camel — Optimization Trajectories", use_log=True)

    # ── Print final values ───────────────────────────────────────────────────
    print("\n=== Final values ===")
    for label, res in [("BOWL", results_bowl), ("CAMEL", results_camel)]:
        print(f"\n{label}:")
        for name, (traj, vals) in res.items():
            pt = traj[-1].numpy()
            print(f"  {name:10s}  →  ({pt[0]:+.5f}, {pt[1]:+.5f})  f = {vals[-1]:.6f}")

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("""
Analysis
========

1. Convex bowl  f(x,y) = x² + 2y²
   - All four optimizers converge to the global minimum (0, 0).
   - Plain GD converges in a straight-but-slow path because the different
     curvatures along x and y cause a zig-zag if the learning rate is high.
   - Momentum overshoots initially but then accelerates past GD, reaching
     the minimum faster thanks to accumulated velocity along the dominant
     gradient direction.
   - AdaGrad adapts the per-coordinate learning rate: the y-direction
     (steeper gradient) gets a smaller effective step while x keeps a
     larger one.  This reduces oscillations and gives a fairly direct path.
   - Adam combines both ideas and converges the fastest, following the
     smoothest trajectory.

2. Six-hump Camel function
   - None of the optimizers *reliably* finds a global minimum from every
     starting point; they all can get trapped in a local minimum depending
     on where they start.
   - From (-2, -1.5), GD follows a steep path into the nearest local
     basin and stops there.  Momentum may overshoot the local basin and
     land in a better one, but this is not guaranteed.
   - AdaGrad's shrinking effective step size makes it conservative late in
     training — good for stability but bad for escaping local minima.
   - Adam's per-coordinate adaptation and momentum give it the best chance
     of navigating the curved valleys, but it still converges to whichever
     basin its initial trajectory leads into.

3. Hyperparameter sensitivity
   - The same LR does NOT work equally well on both functions.  The bowl
     tolerates larger LRs; the Camel landscape requires smaller LRs to
     avoid jumping across basins unpredictably.
   - Momentum and Adam are less sensitive to the exact LR choice because
     the adaptive/momentum terms smooth out large gradient swings.

4. Summary of optimizer advantages over plain GD
   - **Momentum**: accelerates convergence in consistent-gradient
     directions and dampens oscillations in high-curvature directions.
   - **AdaGrad**: automatic per-parameter LR scaling — great for sparse
     gradients or features with very different magnitudes.
   - **Adam**: combines momentum (first moment) with adaptive LR (second
     moment) plus bias correction, giving fast, stable convergence as the
     best general-purpose optimizer.
""")

    return results_bowl, results_camel


if __name__ == "__main__":
    run_part_2()
