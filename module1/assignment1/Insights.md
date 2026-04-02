# Insights — Hometask 1: Optimization in PyTorch

> Gradient Descent, SGD, Numerical Stability, and L1 Regularization

---

## Table of Contents

1. [Task 1.1 — Logistic Regression Implementation](#task-11--logistic-regression-implementation)
2. [Task 1.2 — SGD Training Loop with BCE Loss](#task-12--sgd-training-loop-with-bce-loss)
3. [Task 1.3 — Hyperparameter Experiments (LR × Batch Size)](#task-13--hyperparameter-experiments)
4. [Task 1.4 — L1 Regularization and Sparsity](#task-14--l1-regularization-and-sparsity)
5. [Part 2 — Comparing Optimization Algorithms](#part-2--comparing-optimization-algorithms)

---

## Task 1.1 — Logistic Regression Implementation

### What was implemented

A `LogisticRegression` class implemented as a PyTorch `nn.Module` with:
- Configurable weight initialization (`zeros`, `random`, or from a tensor)
- Forward pass computing logits + sigmoid
- Threshold-based `predict()` method

### Design Decisions and Insights

#### 1. Weight Initialization Matters

| Init Method | Behavior | When to Use |
|---|---|---|
| `zeros` | All weights start equal → symmetric gradient updates | Safe default; works fine for convex problems like logistic regression |
| `random` (×0.01) | Breaks symmetry; each feature starts with a different "pull" | Better for deeper networks; useful to compare with `zeros` |
| `tensor` | Resume from a checkpoint or transfer weights | Reproducibility, warm-starting |

**Why small random values (0.01)?** Large initial weights push sigmoid outputs to 0 or 1, which creates **vanishing gradients** (since σ'(z) ≈ 0 at the extremes). Keeping weights near zero keeps initial predictions near 0.5, where the sigmoid gradient is maximal (~0.25), enabling efficient early learning.

#### 2. `nn.Parameter` wrapping

Wrapping tensors in `nn.Parameter` does two critical things:
- Registers them in `model.parameters()` so the optimizer can find them
- Sets `requires_grad=True` automatically, enabling autograd to track operations

#### 3. Sigmoid vs. Raw Logits

The forward pass explicitly computes `logits = x @ w + b` then `probs = sigmoid(logits)`. This separation is important because:
- Logits can be used directly with more numerically stable loss functions (e.g., `BCEWithLogitsLoss`)
- Probabilities are needed for prediction and interpretability

#### 4. Prediction Threshold

The threshold of 0.5 is the standard decision boundary for balanced binary classification. For the SST-2 dataset (≈48% negative, ≈52% positive), the classes are nearly balanced, so 0.5 is appropriate. For heavily imbalanced datasets, tuning this threshold would be necessary.

---

## Task 1.2 — SGD Training Loop with BCE Loss

### What was implemented

1. **Binary Cross-Entropy Loss** from scratch with numerical stability (clamping)
2. **Mini-batch SGD training loop** with shuffling, logging, and evaluation

### Design Decisions and Insights

#### 1. Numerical Stability in BCE Loss

```python
y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
```

**Why this is critical:** The BCE formula contains `log(p)` and `log(1-p)`. If `p = 0` or `p = 1`, we get `log(0) = -∞`, producing `NaN` losses that destroy training. Clamping to `[ε, 1-ε]` ensures finite log values.

The epsilon value `1e-15` is chosen to be small enough to not distort gradients while preventing numerical catastrophe. In practice, values between `1e-7` and `1e-15` work well.

#### 2. Why Accuracy as Evaluation Metric

For the SST-2 dataset:
- **Class balance**: ~48% negative, ~52% positive → nearly balanced
- **Accuracy** is appropriate for balanced datasets because it weights all samples equally
- **F1-score** would be preferred if the dataset were imbalanced (e.g., 95%/5% split), as it accounts for precision-recall tradeoffs

The implementation supports both via the `metric` parameter.

#### 3. Epoch-level Shuffling

Shuffling the training data at the start of each epoch (`torch.randperm(n_samples)`) is crucial because:
- Without shuffling, mini-batches always contain the same samples → biased gradient estimates
- Shuffling provides a form of regularization (different gradient noise each epoch)
- It prevents the optimizer from "memorizing" a fixed batch ordering

#### 4. Parameter History Tracking

Saving `w` and `b` after **every batch update** (not just every epoch) gives fine-grained visibility into the optimization trajectory. This is essential for Task 1.4 where we study how individual weights evolve under L1 regularization.

#### 5. Order of Operations: zero_grad → backward → step

In our implementation we use `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`:
- `zero_grad()` clears accumulated gradients from the previous batch
- `backward()` computes gradients via backpropagation
- `step()` updates parameters using the computed gradients

**Important**: If `zero_grad()` is skipped, gradients accumulate across batches, effectively increasing the batch size and learning rate in unpredictable ways.

---

## Task 1.3 — Hyperparameter Experiments

### Experimental Setup

- **Learning rates**: [0.01, 0.03, 0.1, 0.3, 1.0]
- **Batch sizes**: [50, 100, 200]
- **Epochs**: 20
- **Init**: zeros
- **Metric**: accuracy

### Key Findings

#### Learning Rate Effects

| LR Range | Convergence | Stability | Performance |
|---|---|---|---|
| **0.01–0.03** (small) | Slow — may not converge in 20 epochs | Very stable | Underfitting; model hasn't "arrived" yet |
| **0.1–0.3** (moderate) | Fast — reaches good solution quickly | Stable with occasional oscillation | Best validation performance |
| **1.0** (large) | Very fast initially but oscillates | Unstable — can diverge | Degrades; may produce NaN |

**Why?** The learning rate controls the step size: `w ← w - lr × ∇L`. A step that's too large overshoots the minimum and bounces around (or diverges). A step that's too small gets stuck in early-stage exploration.

#### Batch Size Effects

| Batch Size | Gradient Quality | Updates/Epoch | Effect |
|---|---|---|---|
| **50** (small) | Noisy — high variance estimate | 138 updates | More exploration; noisier loss curve |
| **100** (medium) | Moderate noise | 69 updates | Good balance |
| **200** (large) | Smooth — low variance | 34 updates | Smoother convergence; fewer updates |

**Why?** A mini-batch gradient is an estimate of the true gradient. Larger batches → lower variance → smoother but fewer updates per epoch. Smaller batches → higher variance → more updates, with gradient noise acting as implicit regularization.

#### Interaction Between LR and Batch Size

There's a well-known **linear scaling rule**: when you double the batch size, you can roughly double the learning rate to maintain the same convergence behavior. This is because:

```
Effective step ≈ (lr / batch_size) × gradient_signal
```

In the heatmap, you should observe that:
- High LR + small batch = **instability** (too much noise amplified by large steps)
- Low LR + large batch = **slow convergence** (smooth but tiny steps)
- Moderate LR + moderate batch = **sweet spot**

---

## Task 1.4 — L1 Regularization and Sparsity

### Theoretical Background

The L1-regularized loss is:

```
L_reg(w) = L_BCE(w) + λ × Σ|w_i|
```

The gradient of the L1 penalty is `λ · sign(w)`, which pushes each weight toward zero with **constant force** `λ` — regardless of the weight's magnitude. This key property is what drives sparsity.

### Key Findings

#### 1. Sparsity vs. λ

| λ | Non-zero weights | Observation |
|---|---|---|
| 0 | ~10,000 | No regularization — all features active |
| 1e-4 | ~9,900+ | Minimal effect |
| 1e-3 | ~8,000–9,000 | Some pruning begins |
| 1e-2 | ~3,000–5,000 | Significant feature elimination |
| 1e-1 | ~500–1,000 | Aggressive sparsity — risks underfitting |

As λ increases, the L1 penalty dominates the loss for features with weak gradients (i.e., uninformative features), pushing them to near-zero.

#### 2. Why L1 Encourages Sparsity (vs. L2)

**L1 gradient**: The sign function produces a constant push of ±λ toward zero:

```
w_i → w_i - α × ∂L/∂w_i ± α×λ
```

This **constant** push doesn't weaken as `w_i` approaches zero, so weights get pushed all the way to (near) zero.

**L2 gradient**: The gradient is proportional to the weight itself: `2λ·w_i`

```
w_i → w_i(1 - 2αλ) - α × ∂L/∂w_i
```

This creates **exponential decay** — weights shrink but never reach exactly zero. L2 produces many small weights; L1 produces few large weights and many (near-)zero weights.

#### 3. Weight Initialization Comparison

| Aspect | `zeros` init | `random` init |
|---|---|---|
| **Stability** | Very stable — symmetric start | Slightly noisier early on |
| **Final sparsity** | Smooth, predictable sparsification | Similar final sparsity for same λ |
| **Performance** | Consistent across runs | Minor run-to-run variation |
| **Weight dynamics** | Symmetric decay paths | Asymmetric — some weights resist longer |

With `zeros` initialization, all weights start equal and receive identical gradient pressure initially. With `random` initialization, some weights start further from zero and resist the L1 pull longer, but eventually both converge to similar sparsity patterns — the "important" features survive in both cases.

#### 4. Weight Dynamics Under L1

The characteristic behavior of weights under L1 regularization:
- **Quasi-linear decay**: Weights decrease almost linearly toward zero (constant force ±αλ)
- **Oscillation near zero**: With plain SGD, weights reach near-zero but oscillate around it because the constant L1 push doesn't adapt. The weight overshoots zero, the sign flips, and it gets pushed back.
- **No exact zeros**: Plain SGD lacks the **proximal step** needed to clamp weights exactly to zero. Proximal gradient descent would apply a soft-thresholding operator that sets small weights to exactly zero.

#### 5. Practical Takeaway

Moderate λ (1e-3 to 1e-2) provides the best accuracy-sparsity tradeoff: it removes noisy features while keeping the informative ones. Too much regularization (λ = 0.1) causes underfitting. For exact feature elimination, consider proximal gradient descent instead of plain SGD.

---

## Part 2 — Comparing Optimization Algorithms

### Algorithms Implemented

| Optimizer | Update Rule | Key Idea |
|---|---|---|
| **GD** | `θ ← θ - lr·∇f` | Plain vanilla — follow the negative gradient |
| **Momentum** | `v ← β·v + ∇f; θ ← θ - lr·v` | Accumulate past gradients for acceleration |
| **AdaGrad** | `G += (∇f)²; θ ← θ - lr·∇f/√(G+ε)` | Per-parameter adaptive learning rate |
| **Adam** | Combines momentum + adaptive LR + bias correction | Best of both worlds |

### Results on Function A — Convex Bowl: f(x,y) = x² + 2y²

This is a simple convex function with a single global minimum at (0, 0).

| Optimizer | Convergence Speed | Path Smoothness | Final Value |
|---|---|---|---|
| GD | Moderate | Zig-zag if LR is high | 0 (exact) |
| Momentum | Fast (after oscillation) | Initial overshoot, then smooth | 0 (exact) |
| AdaGrad | Fast | Very direct — no zig-zag | ≈0 (slows near min) |
| Adam | Fastest | Smooth, direct | ≈0 |

**Key observation**: The bowl has different curvatures along x and y (1 vs 4). This causes GD to zig-zag: the y-gradient is larger, so GD takes big steps in y and small steps in x, bouncing across the narrow axis. AdaGrad and Adam fix this by scaling each coordinate's step by its gradient history, effectively normalizing the curvatures.

### Results on Function B — Six-Hump Camel

This function has:
- 2 global minima: (0.0898, -0.7126) and (-0.0898, 0.7126) with f ≈ -1.0316
- 4 local minima
- Narrow curved valleys

| Optimizer | Finds Global Min? | Behavior |
|---|---|---|
| GD | Depends on start | Falls into nearest basin, gets stuck |
| Momentum | Sometimes | Can overshoot basins — sometimes lands in a better one |
| AdaGrad | Rarely | Conservative late-stage steps prevent basin-hopping |
| Adam | Sometimes | Best chance due to adaptive + momentum, but not guaranteed |

**Critical insight**: **None of the optimizers reliably find the global minimum.** All gradient-based methods are fundamentally local — they follow the gradient downhill to the nearest basin. Whether that basin is a global or local minimum depends on the starting point. For global optimization on non-convex functions, you'd need techniques like:
- Random restarts
- Simulated annealing
- Evolutionary algorithms
- Large-scale stochastic methods

### Hyperparameter Sensitivity

The same hyperparameters do **NOT** work well for both functions:

| Parameter | Bowl | Camel |
|---|---|---|
| GD lr | 0.05 works well | 0.05 causes wild oscillation; need 0.005 |
| Momentum beta | 0.9 standard | 0.9 can cause overshoot across basins |
| AdaGrad lr | 0.5 (high, compensated by decay) | 0.1 |
| Adam lr | 0.1 | 0.05 (lower to avoid jumping basins) |

The Camel function's complex landscape with saddle points and ridges requires more careful tuning. The bowl is forgiving because it has a single basin.

### Advantages Over Plain GD

1. **Momentum**: Accelerates in consistent-gradient directions, dampens oscillations in high-curvature directions. Like a ball rolling downhill — it builds up speed in the dominant direction.

2. **AdaGrad**: Automatically scales the learning rate per parameter based on gradient history. Features that get large gradients get smaller steps; features with small gradients get larger steps. Perfect for sparse data (like our bag-of-words features!).

3. **Adam**: Combines momentum's acceleration with AdaGrad's adaptivity, plus bias correction for early steps. It's the most robust general-purpose optimizer, which is why it's the default choice in deep learning.

**Note on AdaGrad's weakness**: The accumulated squared gradients G only grow, which means the effective learning rate monotonically decreases. In long training runs, AdaGrad can effectively stop learning too early. RMSProp and Adam fix this by using an exponential moving average instead of cumulative sum.

---

## Summary Table

| Task | Key Concept | Main Takeaway |
|---|---|---|
| 1.1 | Logistic Regression | Small weight init keeps sigmoid in its sensitive region for fast learning |
| 1.2 | SGD + BCE | Numerical stability (clamping) is essential; shuffling provides regularization |
| 1.3 | LR × Batch Size | Moderate LR + moderate batch size is the sweet spot; linear scaling rule governs their interaction |
| 1.4 | L1 Regularization | L1's constant push creates sparsity; λ ≈ 1e-3 to 1e-2 balances performance and feature selection |
| Part 2 | Optimizer Comparison | Adam is the best general-purpose optimizer; no gradient method reliably solves non-convex problems |
