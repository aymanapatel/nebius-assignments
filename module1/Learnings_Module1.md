# Module 1 Learnings: LLM Architectures

---

## Assignment 1: Optimization in PyTorch

### Logistic Regression from Scratch

Implemented a custom `LogisticRegression` class using PyTorch's `nn.Module` with configurable weight initialization (`zeros`, `random`, or tensor warm-start). Key design decisions:

- **`nn.Parameter` wrapping**: Wrapping tensors registers them in `model.parameters()` so the optimizer can discover them automatically, and sets `requires_grad=True` for autograd.
- **Small random init**: Weights initialized with `randn * 0.01` keep initial sigmoid outputs near 0.5, where the gradient is maximal (~0.25). Large initial weights cause vanishing gradients at the sigmoid extremes.
- **Sigmoid vs. raw logits**: Computing `logits = x @ w + b` then `probs = sigmoid(logits)` preserves the option to use numerically stable `BCEWithLogitsLoss` while exposing probabilities for prediction.
- **Thresholding**: 0.5 is appropriate for balanced binary tasks like SST-2 (~48%/52%); imbalanced datasets require threshold tuning.

### SGD Training Loop & Numerical Stability

- **Binary Cross-Entropy from scratch**: `loss = -(y*log(p) + (1-y)*log(1-p))`.
- **Clamping for numerical stability**: `torch.clamp(y_pred, 1e-15, 1 - 1e-15)` prevents `log(0) → -∞` and `NaN` losses.
- **Epoch-level shuffling**: `torch.randperm(n_samples)` at each epoch start debiases mini-batch gradient estimates and acts as implicit regularization.
- **Parameter history tracking**: Saving `w` and `b` after every batch update enables fine-grained analysis of weight trajectories under regularization.
- **Training loop order**: `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`. Skipping `zero_grad()` causes gradient accumulation.

### Hyperparameter Experiments: Learning Rate × Batch Size

Grid search over LR ∈ {0.01, 0.03, 0.1, 0.3, 1.0} × BS ∈ {50, 100, 200}:

| LR Range | Behavior |
|----------|----------|
| 0.01–0.03 | Slow convergence; may underfit |
| 0.1–0.3 | Best validation accuracy; stable |
| 1.0 | Oscillation/divergence; especially with small batches |

| Batch Size | Behavior |
|------------|----------|
| 50 | Noisy gradients; more updates; implicit regularization |
| 200 | Smooth gradients; fewer updates; needs more epochs or higher LR |

**Linear scaling rule**: Doubling batch size allows roughly doubling LR while preserving convergence behavior: `Effective step ≈ (lr / batch_size) × gradient_signal`.

### L1 vs. L2 Regularization and Sparsity

- **L1 loss**: `L_reg(w) = L_BCE(w) + λ·Σ\|w_i\|`
- **Why L1 creates sparsity**: The gradient `λ·sign(w)` applies a constant push toward zero regardless of weight magnitude. This drives weak-feature weights to (near) zero. In contrast, L2's gradient `2λ·w_i` causes exponential decay but weights rarely reach exactly zero.
- **Sparsity vs. λ**:
  - λ=0: ~10,000 non-zero weights
  - λ=1e-3: ~8,000–9,000 non-zero
  - λ=1e-2: ~3,000–5,000 non-zero
  - λ=1e-1: ~500–1,000 non-zero (risks underfitting)
- **Weight dynamics**: Under plain SGD, weights exhibit quasi-linear decay toward zero and oscillate around it because vanilla SGD lacks the proximal soft-thresholding step needed for exact zeros.

### Comparing Optimization Algorithms

Implemented GD, Momentum, AdaGrad, and Adam from scratch on two loss landscapes:

**Convex bowl: f(x,y) = x² + 2y²**
- All four converge to (0,0)
- GD zig-zags due to different curvatures along x and y axes
- Momentum overshoots initially then accelerates past GD
- AdaGrad normalizes per-coordinate step sizes, reducing oscillation
- Adam converges fastest and smoothest

**Six-Hump Camel (non-convex)**
- **None reliably find the global minimum** — gradient methods are fundamentally local
- Adam has the best chance due to adaptive + momentum, but convergence depends on starting point
- For global optimization on non-convex functions, random restarts or evolutionary methods are needed

**Hyperparameter sensitivity**: The same LR does NOT work across landscapes. The bowl tolerates larger LRs; the Camel requires smaller LRs to avoid jumping basins.

---

## Assignment 2: RNN Language Model & CIFAR-10 Classification

### Binary CIFAR-10 Classification

- **Airplane vs. Ship**: Achieved **94.35%** accuracy (target >94%). This pair is visually separable; moderate-capacity MLP with BatchNorm/Dropout converges quickly. Threshold calibration (0.5 → 0.665) improved accuracy from 94.00% to 94.35%.
- **Cat vs. Dog**: Achieved **65.35%** accuracy (target >64%). A harder pair due to visual similarity; increasing depth + regularization did not help, likely due to optimization instability.
- **Activation comparisons** (fixed architecture):
  - Airplane/Ship: sigmoid (93.90%), tanh (90.40%), relu (93.85%)
  - Cat/Dog: sigmoid (61.80%), tanh (60.95%), relu (60.55%)

### Multi-Class CIFAR-10 Classification

- **10-class MLP**: Achieved **56.22%** accuracy (target >53%).
- **Key principle**: Correct loss/output pairing is essential — `CrossEntropyLoss` with raw logits (no final Softmax) for multi-class; BCE + sigmoid for binary.
- **Input normalization** via custom `CIFAR10Dataset` significantly stabilizes convergence.
- **Activation comparisons**: ReLU (51.58%), Sigmoid (49.48%), Tanh (48.05%). ReLU generally converges faster and to better final values.

### Custom Dataset Implementation

- **`CIFAR10Dataset`**: Implements `__init__` (loading data, building label maps), `__len__`, and `__getitem__` with optional normalization and binary label mapping.

### Char-RNN Language Model

- **`DinosDataset`**: Custom PyTorch `Dataset` that concatenates dinosaur names with `<name>` delimiters, builds a character vocabulary, and creates fixed-length sequence pairs `(x, y)` where `y` is `x` shifted by one character.
- **One-hot encoding**: `np.eye(vocab_size)[char_ids]` converts character indices to sparse vectors (28-char vocabulary).
- **Model architecture**: `CharRNN` with `nn.LSTM(vocab_size, n_hidden, n_layers, dropout=drop_prob, batch_first=True)`, followed by `nn.Dropout` and `nn.Linear(n_hidden, vocab_size)`.
- **Training**: Adam optimizer, `CrossEntropyLoss`, gradient clipping (`clip_grad_norm_` with `max_norm=5`), hidden state detachment between batches.
- **Generation**: Autoregressive sampling — feed one character at a time, apply softmax with temperature, sample via `torch.multinomial`, stop at `>` delimiter.

---

## Assignment 3: Build Your Own Tiny Transformer

### Multi-Head Causal Self-Attention

- **Q/K/V projection**: Single `nn.Linear(n_embd, 3*n_embd, bias=False)` produces Q, K, V together (more efficient than three separate linears). Split via `.chunk(3, dim=-1)`.
- **Head splitting**: Reshape `(B, T, C)` → `(B, n_head, T, head_size)` via `.view(B, T, n_head, head_size).transpose(1, 2)`.
- **Scaled dot-product attention**: `scores = (q @ k^T) / sqrt(head_size)`, shape `(B, n_head, T, T)`.
- **Causal mask BEFORE softmax**: `scores.masked_fill(~mask[:T,:T], float("-inf"))` — this is critical. Applying the mask after softmax would leak future information.
- **Output projection**: Merge heads via `.transpose(1,2).contiguous().view(B, T, C)`, then `nn.Linear(n_embd, n_embd)` + dropout.

### Position-Wise Feed-Forward Network

- **Architecture**: `Linear(n_embd, 4*n_embd)` → `GELU()` → `Linear(4*n_embd, n_embd)` → `Dropout`.
- **Position-wise independence**: Each position is processed independently; verified by checking that per-position outputs match whole-sequence outputs.

### Pre-Norm Transformer Block

- **Pattern**: `x = x + self.attn(self.ln1(x))` then `x = x + self.ffwd(self.ln2(x))`.
- **Why pre-norm**: More stable training than post-norm (original Transformer). LayerNorm before each sublayer stabilizes gradients.
- **Residual connections**: Enable training of deep stacks by allowing gradients to flow directly through the addition.

### Full Language Model (`TinyTransformerLM`)

- **Architecture**: Token embeddings + positional embeddings → stack of `n_layer` Blocks → final LayerNorm → `lm_head` Linear.
- **Positional embeddings**: `nn.Embedding(block_size, n_embd)` — learned absolute positions, added to token embeddings.
- **Cross-entropy loss**: `F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))` — logits shape `(B, T, vocab_size)`, targets shape `(B, T)`.

### Autoregressive `generate()` Method

- **Context cropping**: `idx_cond = idx[:, -self.block_size:]` — essential when the running context exceeds `block_size`.
- **Sampling**: `logits[:, -1, :]` → `softmax` → `torch.multinomial(probs, num_samples=1)` → concatenate.
- **`@torch.no_grad()`** decorator for inference.

### Sanity Checks

- **Output shape**: `logits.shape == (batch_size, block_size, vocab_size)`.
- **Causality verification**: Perturbing the last input position must not change outputs at earlier positions (max diff < 1e-6).
- **Untrained model loss**: Cross-entropy should start near `log(vocab_size)` (~4.17 for 65-char vocabulary). Deviation > 0.5 indicates a bug.
- **Variable-T handling**: The causal mask must be sliced to current `T` during generation.

### Perplexity and Bits-Per-Character

- **Perplexity (PPL)**: `exp(NLL)` — interpretable as the effective branching factor.
- **Bits-per-character (BPC)**: `NLL / ln(2)`.

---

## Assignment 4: LoRA from Scratch & MoE Transformer

### LoRA (Low-Rank Adaptation)

- **Core formula**: `y = W₀x + b + (α/r)·B·A·dropout(x)`, where `A ∈ R^(r×d_in)`, `B ∈ R^(d_out×r)`, `r ≪ min(d_in, d_out)`.
- **Initialization**: A → Kaiming uniform (`a=√5`), B → **zeros**. This ensures `ΔW = 0` at init, so the wrapped model behaves identically to the original on step 0.
- **`LoRALinear` wrapper**: Freezes base weight and bias (`requires_grad=False`), adds `lora_A` and `lora_B` as `nn.Parameter`, optional `nn.Dropout` on input before LoRA branch.
- **`merged_weight()`**: Returns `W₀ + (α/r)·B·A` for inference-time weight merging (same FLOPs as original).

### Injecting LoRA into GPT-2

- **Target modules**: `c_attn` (fused QKV) and `c_proj` (attention output + MLP down-projection).
- **Conv1D→Linear conversion**: HuggingFace GPT-2 historically uses `Conv1D` (weight transposed). The `conv1d_to_linear()` helper transposes weights so `LoRALinear` can wrap standard `nn.Linear`.
- **36 wrappings**: 12 blocks × (1 `c_attn` + 1 `attn.c_proj` + 1 `mlp.c_proj`) = 36 LoRA-adapted layers.
- **Trainable parameter count**: ~811K out of 124M ≈ **0.65%**. Per-layer breakdown:
  - `c_attn` (768→2304): 8×(768+2304)=24,576 params
  - `attn.c_proj` (768→768): 8×(768+768)=12,288 params
  - `mlp.c_proj` (3072→768): 8×(3072+768)=30,720 params

### Fine-Tuning and Style Transfer

- **Dataset**: Tiny Shakespeare tokenized into fixed-length chunks.
- **Optimizer**: AdamW over **only** trainable parameters to save optimizer-state memory.
- **Style transfer verification**:
  - Baseline GPT-2 generates generic modern prose / legal text.
  - Post-LoRA output shifts to pseudo-Elizabethan diction (`thou`, `thee`, `lord`), play-like speaker labels, and dramatic line breaks.

### Adapter Persistence and Merging

- **Save/load adapter**: Only LoRA parameters saved (~3 MB vs ~500 MB for full model). Round-trip verification: load adapter into fresh GPT-2+LoRA, confirm `torch.allclose(logits_a, logits_b, atol=1e-5)`.
- **Merged weight equivalence**: Computing `F.linear(x, W_merged, bias)` yields identical output to the separate forward pass, enabling inference-time FLOP parity.

### Comparison with `peft` Library

- **Configuration**: `LoraConfig(task_type=CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["c_attn", "c_proj"])`.
- **Parameter count parity**: Both hand-rolled and `peft` yield ~811,008 trainable parameters when targeting the same modules.
- **Behavioral parity**: With matched hyperparameters, both approaches reach similar Shakespeare PPLs; small differences arise from random initializations, dropout masks, and dataloader shuffle order.

### Rotary Position Embeddings (RoPE)

- **Frequency caching**: `inv_freq = base^(-2i/d)` for `i ∈ [0, d/2)`, with `base=10000`. Precompute `cos` and `sin` grids up to `max_seq_len` and register as non-trainable buffers.
- **`rotate_half(x)`**: Splits the last dimension in half, swaps and negates: `[-x₂, x₁]`. This implements the rotation matrix in real-number form.
- **Forward pass**: `q_rot = q * cos + rotate_half(q) * sin` (same for k). Slice cached cos/sin to current sequence length `T`.
- **Norm preservation**: Rotation matrices are orthogonal, so `\|q_rot\| = \|q\|`. Verified by unit test: `torch.allclose(original_norm, rotated_norm, atol=1e-5)`.
- **Replaces learned positional embeddings**: The `TinyMoeLM` uses only token embeddings (no `pos_emb`); RoPE encodes position directly into Q and K.

### Mixture of Experts (MoE)

- **DeepSeek-style fine-grained experts**: Halve the hidden dimension (`exp_hid_dim = hid_dim // 2`) but double the number of experts (8 instead of 4) to maintain the same compute budget.
- **Router**: `nn.Linear(n_embd, num_experts, bias=False)` produces logits → softmax → top-K selection.
- **Top-K routing with normalization**: Select top-K=2 experts per token; normalize selected weights to sum to 1.0 per token.
- **Expert capacity formula**: `capacity = int((total_tokens × top_k / num_experts) × capacity_factor)`. With `capacity_factor=1.25`, this provides 25% buffer above the even split.
- **Token dropping**: If an expert receives more tokens than its capacity, excess tokens are dropped (ignored). Track `total_dropped` across all experts.
- **Drop rate metric**: `drop_rate = total_dropped / (total_tokens × top_k)`. Should stay below ~20% after initial training iterations.
- **Output accumulation**: `out_flat.index_add_(0, token_idx, expert_out * weights)` — weighted sum of expert outputs for each token.

### MoE Transformer Architecture

- **`BlockMoe`**: Pre-norm block with `MultiHeadSelfAttentionRope` + `MoE` (replacing `FeedForward`). Returns `(x, drop_rate)` tuple.
- **`TinyMoeLM`**: Token embeddings only (no positional embeddings) → stack of `BlockMoe` → final LayerNorm → `lm_head`. Returns `(logits, loss, avg_drop_rate)`.

---

## Cross-Cutting Themes

1. **Numerical stability is non-negotiable**: Clamping in BCE (A1), causal masking before softmax (A3), and zero-init of LoRA B (A4) all prevent training catastrophes.
2. **Shape and causality invariants**: Every assignment reinforces that tensor shapes must be explicitly checked and autoregressive models must strictly prevent future-information leakage.
3. **Efficiency vs. capacity trade-offs**: L1 sparsity (A1), LoRA low-rank updates (A4), and MoE sparse routing (A4) are all ways to control parameter count or compute without sacrificing model expressiveness.
4. **Validation beyond loss**: Perplexity/BPC (A3), control-corpus PPL (A4), and drop-rate metrics (A4) show that single-number loss is insufficient — domain shift, forgetting, and load balancing need dedicated diagnostics.
5. **Weight initialization matters**: From small random values in logistic regression (A1) to zeros for LoRA B (A4), initialization determines whether training starts in a viable regime.
6. **Pre-norm > post-norm for deep stacks**: Modern GPT-style transformers use `x = x + sublayer(LN(x))` because it trains more stably than the original post-norm layout.
7. **Attention is permutation-invariant without position**: Positional embeddings (learned in A3, RoPE in A4) are required because self-attention alone has no notion of token order.
8. **Gradient flow requires care**: Residual connections enable deep stacks, gradient clipping prevents explosions, and detaching hidden states in RNNs limits backpropagation scope.
