# Combined Learnings: All Modules

---

# Module 1 Learnings: LLM Architectures

---

## Assignment 1: Optimization in PyTorch

### Logistic Regression from Scratch

Implemented a custom `LogisticRegression` class using PyTorch's `nn.Module` with configurable weight initialization (`zeros`, `random`, or tensor waarm-start). Key design decisions:

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

- **L1 loss**: `L_reg(w) = L_BCE(w) + λ·Σ|w_i|`
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
- **Norm preservation**: Rotation matrices are orthogonal, so `‖q_rot‖ = ‖q‖`. Verified by unit test: `torch.allclose(original_norm, rotated_norm, atol=1e-5)`.
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

## Cross-Cutting Themes (Module 1)

1. **Numerical stability is non-negotiable**: Clamping in BCE (A1), causal masking before softmax (A3), and zero-init of LoRA B (A4) all prevent training catastrophes.
2. **Shape and causality invariants**: Every assignment reinforces that tensor shapes must be explicitly checked and autoregressive models must strictly prevent future-information leakage.
3. **Efficiency vs. capacity trade-offs**: L1 sparsity (A1), LoRA low-rank updates (A4), and MoE sparse routing (A4) are all ways to control parameter count or compute without sacrificing model expressiveness.
4. **Validation beyond loss**: Perplexity/BPC (A3), control-corpus PPL (A4), and drop-rate metrics (A4) show that single-number loss is insufficient — domain shift, forgetting, and load balancing need dedicated diagnostics.
5. **Weight initialization matters**: From small random values in logistic regression (A1) to zeros for LoRA B (A4), initialization determines whether training starts in a viable regime.
6. **Pre-norm > post-norm for deep stacks**: Modern GPT-style transformers use `x = x + sublayer(LN(x))` because it trains more stably than the original post-norm layout.
7. **Attention is permutation-invariant without position**: Positional embeddings (learned in A3, RoPE in A4) are required because self-attention alone has no notion of token order.
8. **Gradient flow requires care**: Residual connections enable deep stacks, gradient clipping prevents explosions, and detaching hidden states in RNNs limits backpropagation scope.

---

# Module 2 Learnings: AI Agents & Sovereign Systems

---

## Assignment 1: Sovereign Agent — Week 1

### Exercise 1: Context Engineering

- **"Lost in the Middle" effect**: LLM answer retrieval drops when relevant information sits in the middle of long context. Beginning and end positions perform better.
- **Prompt formatting matters**: Instruction order, delimiters, and output constraints materially change model accuracy.
- **XML-style structure**: Using clear separators and structured prompts improves reliability across vendors (OpenAI, Anthropic).
- **Context placement strategies**: Sandwiching critical info at the beginning or end of prompts yields more reliable extractions than burying it mid-context.
- **Evaluation mindset**: One-off prompt checks should evolve into repeatable eval workflows with dataset-driven regression tracking.

### Exercise 2: LangGraph Research Agent (The Headless Automator)

- **ReAct pattern**: Interleaving reasoning (`Thought:`) with actions (`Action:`) allows the agent to decide its own tool-calling sequence.
- **LangGraph StateGraph**: Graph-based execution with conditional edges and explicit state handling.
- **Tool implementation**: `venue_search`, `get_edinburgh_weather`, `calculate_catering_cost`, and `generate_event_flyer` — each returning structured `ToolResult` objects.
- **Autonomous planning**: The agent receives a high-level task ("find a pub for 160 people") and decides which tools to call, in what order, without explicit scripting.
- **Failure handling**: When the first-choice venue is unavailable, the agent must find an alternative without human guidance.
- **Graph visualization**: The agent's execution path can be rendered as a Mermaid diagram for debugging.

### Exercise 3: Rasa Pro CALM (The Digital Employee)

- **Deterministic flow contract**: `flows.yml` guarantees slot collection order (`guest_count → vegan_count → deposit_amount_gbp → action_validate_booking`). The LLM cannot reorder or skip steps.
- **Python enforces business rules; LLM handles language**: The LLM extracts slot values; Python custom actions (`ActionValidateBooking`) apply hard constraints.
- **Time-based cutoff guards**: Business rules like "no confirmations after 16:45" are enforced in Python, not inferred by the LLM.
- **Deposit and party-size validation**: `MAX_DEPOSIT_GBP` and max party size caps trigger escalation with clear reasons.
- **Out-of-scope deflection**: `handle_out_of_scope` displays `utter_out_of_scope`, then offers to resume the paused flow with slot state preserved.
- **Two-terminal architecture**: `rasa run actions` (port 5055) handles custom logic; `rasa run --enable-api` (port 5005) handles conversation.
- **Why CALM for confirmations**: Auditable, deterministic, every decision traceable — essential when "every word could cost money."

### Exercise 4: MCP Shared Tool Layer

- **MCP contract**: The server (`mcp_venue_server.py`) is the single source of truth. Changing `"status": "available"` → `"full"` immediately changes all clients' results without code edits.
- **Schema is not optional**: Without `args_schema`, `StructuredTool` silently degrades to text generation. The LLM writes out what it would call, but tools never execute.
- **Async bridge pattern**: Mixing sync LangGraph with async MCP requires a `ThreadPoolExecutor` boundary — each `asyncio.run()` call runs in a fresh thread.
- **System prompts for function calling**: Without a system prompt instructing one-tool-at-a-time calling, Llama-3.3-70B batches all intended calls into a single JSON text block.
- **Trace extraction**: LangChain's `AIMessage.tool_calls` attribute (not just Anthropic-style content blocks) must be checked to capture tool invocations.
- **Tool discovery over code changes**: New tools registered in the MCP server are automatically discoverable by all clients.

---

## Assignment 2: Pub Booking — Ex5 through Ex9

### Ex5: Edinburgh Research Scenario (The Loop Half)

- **Four tools**: `venue_search`, `get_weather`, `calculate_cost`, `generate_flyer` — each logs arguments and outputs to `_TOOL_CALL_LOG`.
- **Parallel-safe annotations**: Read-only tools marked `parallel_safe=True`; `generate_flyer` (file write) marked `False`.
- **Dataflow integrity check (`verify_dataflow`)**: Every fact in the final flyer (venue name, price, weather condition) must trace back to a tool call. Hallucinated facts fail the check.
- **Fabrication test**: Deliberately editing the flyer (e.g., changing £540 to £9999) causes `verify_dataflow` to report the unverified fact.
- **Deterministic vs. real mode**: `FakeLLMClient` with scripted trajectory for fast testing; `OpenAICompatibleClient` with `--real` for actual LLM behavior.
- **Tool-spiral detection**: Qwen-3-32B may make 5+ `venue_search` calls with increasingly desperate params. The diagnostic histogram reveals uncalled tools.
- **Session directory structure**: Every run creates `sess_<id>/` with `SESSION.md`, `session.json`, `workspace/`, `logs/trace.jsonl`, `extras/tickets/`, and `ipc/`.

### Ex6: Rasa Structured Half

- **`StructuredHalf` subclass**: Routes booking intent dicts into Rasa via HTTP POST and maps responses back to `HalfResult`.
- **Rasa flows**: `confirm_booking` (happy path), `resume_from_loop` (mid-scenario handoff), `request_research` (structured rejection → back to loop).
- **`ActionValidateBooking`**: Checks deposit <= £300 and party size <= 8; returns rejection reason to the flow if either fails.
- **Validator (`validator.py`)**: Normalizes loose booking data into Rasa's REST message shape — parses £ into int, canonicalizes dates, handles timezone and venue_id.
- **Three-terminal setup**: Terminal 1 (`make rasa-actions`), Terminal 2 (`make rasa-serve`), Terminal 3 (`make ex6-real`).
- **Mock mode**: `make ex6` uses stdlib mock server for development without a Rasa license.

### Ex7: Handoff Bridge (Bidirectional Round-Trip)

- **Bridge orchestration**: `HandoffBridge.run()` manages loop → structured → loop → structured → completion, max 3 rounds.
- **Atomic file IPC**: Handoff messages written to `ipc/handoff_to_*.json`. At most one handoff file visible at any time (fail-closed rule).
- **Rejection flow**: Loop finds venue → structured rejects (party > 8) → bridge builds reverse task → loop re-researches → second structured attempt succeeds.
- **Session state machine**: Clear `session.state_changed` events for each transition (`loop → structured`, `structured → loop`, `structured → complete`).
- **Grader's planted failure**: The structured half may always reject; the bridge must catch and report this rather than looping forever.

### Ex8: Voice Pipeline

- **Manager persona**: Llama-3.3-70B-Instruct with a gruff Edinburgh pub manager system prompt.
- **STT → Agent → TTS round-trip**:
  - Speechmatics real-time STT over websocket
  - Agent processes text and generates response
  - Rime Arcana/ElevenLabs TTS → MP3 → pydub decode → sounddevice playback
- **Text mode (primary gradeable)**: `--text` reads from stdin, prints responses. No API keys needed.
- **Voice mode (bonus)**: `--voice` requires `SPEECHMATICS_KEY` and `RIME_API_KEY`, plus microphone access.
- **Graceful degradation**: Missing `SPEECHMATICS_KEY` falls back to text mode with a visible warning instead of crashing.
- **Trace events**: Every utterance logged as `voice.utterance_in` and `voice.utterance_out` with correct event types.

### Ex9: Reflection

- **Grounded answers**: Every answer cites specific `sess_xxxx` IDs, ticket IDs, and trace lines from actual runs.
- **Planner handoff analysis**: Understanding what signal caused the planner to assign a subgoal to the structured half.
- **Dataflow integrity in practice**: Describing specific scenarios where `verify_dataflow` catches failures a human reviewer wouldn't.
- **Production failure prediction**: Naming exactly one sovereign-agent primitive (ticket state machine, manifest discipline, IPC atomic rename, SessionQueue retry) and one failure mode it would surface.

---

## Cross-Cutting Themes (Module 2)

1. **Two-agent architecture**: The same problem (pub booking) requires two genuinely different architectures — a headless automator for open-ended research and a digital employee for deterministic confirmation.
2. **Deterministic vs. generative**: CALM flows guarantee behavior; LangGraph agents explore. Neither is universally better — the skill is knowing which to reach for.
3. **Dataflow integrity**: Every fact in an agent's output must trace back to a verifiable source. Without this, LLM hallucinations go undetected.
4. **Schema-first tool design**: Tools without schemas silently degrade to text generation. Always define `args_schema` when wrapping external capabilities.
5. **Session-as-directory**: Every run produces a complete artifact directory (`sess_<id>/`) with human-readable summaries and machine-parseable traces.
6. **Graceful degradation**: Production agents must handle missing APIs, rejected bookings, and unavailable services without crashing.
7. **Atomic IPC**: Handoff between agent halves uses atomic file writes to prevent race conditions and ensure fail-closed behavior.
8. **MCP as shared infrastructure**: A single tool server serves multiple clients (LangGraph agent, Rasa action, voice pipeline), eliminating code duplication and ensuring consistency.

---

# Module 3 Learnings: MLOps & Distributed Training

---

## Assignment 1: Distributed Training on Kubernetes

### Distributed Data Parallel (DDP) Training

- **`torchrun` launcher**: Orchestrates multi-node, multi-GPU training with `--nproc_per_node`, `--nnodes`, `--node_rank`, `--master_addr`, and `--master_port`.
- **NCCL backend**: `dist.init_process_group(backend="nccl")` is the standard for GPU-to-GPU communication in PyTorch distributed training.
- **Process group setup**: Each process gets a `local_rank` (GPU device ID), a global `rank`, and the `world_size` (total number of processes).
- **`torch.cuda.set_device(local_rank)`**: Each process binds to exactly one GPU to avoid device contention.

### Kubernetes & SkyPilot Orchestration

- **Nebius MK8S (Managed Kubernetes)**: Provides GPU-enabled node groups with exact resource presets (e.g., `1gpu-16vcpu-200gb` on `gpu-h100-sxm`).
- **SkyPilot job specification (`train_job.yaml`)**: Declares infrastructure requirements, container image, environment variables, and the run command.
- **`num_nodes: 2`**: The job runs across two Kubernetes pods, each with one H100 GPU.
- **Node group configuration**: JSON spec defines fixed node count (2), GPU drivers (CUDA 13.0), boot disk (100 GB NETWORK_SSD), and OS (Ubuntu 24.04).
- **`SKYPILOT_NODE_IPS`**: SkyPilot injects node IPs at runtime; the first IP becomes `MASTER_ADDR` for torchrun coordination.

### Docker Containerization

- **NVIDIA PyTorch base image**: `nvcr.io/nvidia/pytorch:25.12-py3` comes pre-installed with CUDA, PyTorch, and optimized libraries.
- **Layered installs**: `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `wandb`, `scipy` added on top of the base image.
- **Registry push**: Built image pushed to Nebius Container Registry (`cr.eu-north1.nebius.cloud/...`) and referenced in `train_job.yaml`.
- **Reproducible environments**: Container images ensure identical training environments across local development and cloud execution.

### Training Script Architecture

- **Model loading**: `AutoModelForCausalLM.from_pretrained(MODEL_ID)` with optional `HF_TOKEN` for gated models.
- **Tokenizer setup**: `AutoTokenizer.from_pretrained(MODEL_ID)` with fallback `pad_token = eos_token` when no explicit pad token exists.
- **Dataset**: WikiText-2 (`wikitext-2-v1`) with train and validation splits loaded via `datasets.load_dataset`.
- **Tokenization pipeline**:
  1. `tokenize()`: Convert raw text to token IDs, filter empty lines.
  2. `group_texts()`: Concatenate and chunk into fixed-length blocks (`block_size=512`).
- **Data collator**: `DataCollatorForLanguageModeling(tokenizer, mlm=False)` for causal language modeling (next-token prediction).

### Training Configuration & Hyperparameters

- **Max steps**: 500 steps (homework requirement; lower for smoke tests).
- **Learning rate**: `2e-6` with cosine scheduler and 5% warmup.
- **Batch sizing**:
  - `per_device_train_batch_size=4` (per GPU)
  - `gradient_accumulation_steps=1`
  - Effective batch size = `batch_size × num_gpus × num_nodes × grad_accum = 4 × 1 × 2 × 1 = 8`
- **Mixed precision**: `bf16=True` — optimal for H100/L40S; falls back to `fp16` or full precision on older GPUs.
- **Weight decay**: `0.01` for regularization.
- **Gradient clipping**: `max_grad_norm=1.0` to prevent exploding gradients.
- **Checkpointing**: Every 250 steps (`save_steps=250`).
- **Evaluation**: Every 50 steps (`eval_steps=50`) on the validation split.

### Environment-Driven Configuration

All training hyperparameters are externally configurable via environment variables (set in `train_job.yaml`):

| Variable | Purpose |
|----------|---------|
| `MODEL_ID` | HuggingFace model identifier (e.g., `facebook/opt-1.3b`) |
| `BLOCK_SIZE` | Sequence length per training example |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | Batch size per GPU |
| `PER_DEVICE_EVAL_BATCH_SIZE` | Eval batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | Accumulate gradients over N steps |
| `DATALOADER_NUM_WORKERS` | Parallel data loading workers |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer parallelism to avoid deadlocks |

### NCCL Network Initialization

- **NCCL version**: 2.28.9+cuda13.0
- **Network plugin**: Attempts IB (InfiniBand) RDMA plugin first, falls back to Socket transport when no IB device is found.
- **Socket transport**: Uses `eth0` interface for inter-node GPU communication when RDMA is unavailable.
- **GPU Direct RDMA**: Disabled for socket-based transport.
- **Init timing**: ~0.26s per rank for communication setup.

### Hardware & Infrastructure

- **GPU**: NVIDIA H100 80GB HBM3 SXM
- **CUDA Version**: 13.1
- **Driver**: 580.126.09
- **Node config**: 1 GPU, 16 vCPU, 200 GB RAM per node
- **Boot disk**: 100 GB NETWORK_SSD
- **Multi-node topology**: 2 nodes, 1 GPU each, connected via Socket (no IB)

### Training Dynamics (Observed)

- **Initial loss**: ~3.419 at step 1
- **Learning rate warmup**: Rises from `7.2e-07` to `2e-06` over first ~30 steps
- **Loss trajectory**: Increasing from ~3.4 → ~7.6 → ~9.1 → ~10.5 in early steps
- **Interpretation**: Loss divergence in early training suggests the learning rate (`2e-6`) may be too high for OPT-1.3B on WikiText-2, or the model requires more warmup steps. This highlights the importance of monitoring training curves and tuning hyperparameters for specific model-dataset pairs.

---

## Cross-Cutting Themes (Module 3)

1. **Infrastructure as code**: The entire training pipeline — from Docker image to Kubernetes job spec — is version-controlled and reproducible.
2. **Environment-driven configuration**: All tunable parameters live in `train_job.yaml`, not hardcoded in Python, enabling rapid experimentation without code changes.
3. **Observability by default**: NCCL debug logging (`NCCL_DEBUG=INFO`), training metrics every 10 steps, and explicit node IP reporting make distributed debugging tractable.
4. **Container portability**: The same Docker image runs locally (for debugging) and on Nebius MK8S (for scale), eliminating "works on my machine" issues.
5. **SkyPilot abstraction**: Hides Kubernetes complexity — you declare what you need (2 nodes, H100:1, 60GB+ RAM) and SkyPilot handles pod scheduling, networking, and `torchrun` coordination.
6. **bf16 as default**: Modern GPUs (H100, L40S) support bfloat16 natively, offering fp32 dynamic range with fp16 memory footprint — the sweet spot for LLM training.

---

# Module 4 Learnings: Performance Engineering & GPU Inference

---

## Assignment 1: GPU Roofline Model — Memory-Bound vs Compute-Bound Kernels

### Arithmetic Intensity (AI)

- **Definition**: `AI = FLOPs / Bytes` — the ratio of compute to memory traffic for a kernel.
- **Low AI**: Kernel is **memory-bound** (limited by HBM bandwidth).
- **High AI**: Kernel is **compute-bound** (limited by peak FLOP/s throughput).
- **Ridge point**: The crossover on the roofline where `peak_compute = bandwidth × AI`. On H100 SXM (FP32), this is ~20 FLOP/Byte.

### Roofline Model Formula

```
achievable FLOP/s = min(peak_compute, bandwidth × AI)
```

- Plotting kernels on a log-log roofline diagram immediately reveals whether a kernel is hitting the memory ceiling or the compute ceiling.
- **Kernels left of the ridge point** are memory-bound; optimizing them requires reducing data movement or improving locality.
- **Kernels right of the ridge point** are compute-bound; optimizing them requires more FLOPs per unit time (e.g., better parallelism, tensor cores).

### Measuring GPU Performance with CUDA Events

- `torch.cuda.Event(enable_timing=True)` provides precise GPU-side timing, avoiding CPU launch overhead and synchronization delays.
- Benchmark workflow: warmup (to trigger `torch.compile` and warm caches) → CUDA event start → kernel execution → CUDA event end → synchronize → compute median elapsed time.
- Median over 100 repetitions is more robust than mean for GPU benchmarks due to occasional scheduling jitter.

### Impact of `torch.compile` and Kernel Fusion

- **Eager mode**: Each Python loop iteration launches separate GPU kernels (`mul`, `add`), materializing intermediates in global memory. Traffic grows with loop iterations.
- **Compiled mode (`torch.compile`)**: The compiler fuses the loop body into fewer kernels (often one), keeping intermediates in registers. Traffic stays constant (one read, one write), while FLOPs grow with `num_ops`.
- **Measured result**: For 128 ops, compiled AI reached **32 FLOP/Byte** (approaching compute-bound), while eager AI stayed flat at **0.083 FLOP/Byte** (deeply memory-bound).

### Byte-Traffic Models

| Variant | Assumed Traffic per Element | AI scaling |
|---------|----------------------------|------------|
| Eager | `num_ops × 6 × bytes_per_element` (separate kernels, intermediates) | Flat / very low |
| Compiled | `2 × bytes_per_element` (one read + one write at kernel boundary) | Linear with `num_ops` |

### Observed Roofline Data (H100)

| Operation | AI (FLOP/Byte) | Achieved TFLOP/s |
|-----------|---------------|------------------|
| `clone()` (lowest AI) | 0.01 | ~29 (bandwidth-bound) |
| 64 ops compiled | 16.0 | ~39,800 |
| 128 ops compiled | 32.0 | ~53,460 |
| Matmul 1024×1024 | 170.7 | ~32,435 |
| Matmul 4096×4096 | 682.7 | ~51,901 |

### Key Insight

A small matmul (1024×1024) can underperform a simple compiled element-wise kernel on a large GPU because the GEMM may not fully occupy all SMs, while the element-wise kernel exposes massive parallelism across vector elements.

---

## Assignment 2: LLM Inference Optimization

### Baseline Problems

The naive autoregressive loop had three major inefficiencies:
1. **Full sequence forward**: Passed the entire growing sequence every decode step.
2. **CPU-GPU sync**: `.item()` every step forced a device synchronization.
3. **Repeated allocation**: `torch.cat([generated_ids, next_token_id], dim=1)` reallocated and copied memory each step.

### Optimization Ladder (Measured on H100)

| Step | Change | Time (128 tokens) | Speedup vs Baseline |
|------|--------|------------------|---------------------|
| v0 | Naive baseline | 0.943 s | 1.00× |
| v1 | Add `torch.inference_mode()` | 0.929 s | 1.01× |
| v2 | Remove per-token `.item()` sync | 0.884 s | 1.07× |
| v3 | Preallocate full sequence buffer | 0.883 s | 1.07× |
| v4 | **KV cache + one-token decode** | **0.141 s** | **6.69×** |
| v5 | Same with bf16 weights | 0.167 s | 5.66× |

**Final submission** (fp32, KV cache): **0.17 s** → **5.49× speedup**.

### The Biggest Win: KV Cache

- **`past_key_values`**: After the initial prompt prefill, each decode step passes only the **latest token** (`input_ids=next_token_id`) while reusing cached keys and values from all previous positions.
- This eliminates repeated attention computation over the full prompt and all previously generated tokens.
- **`logits_to_keep=1`**: Tells Transformers to compute only the last-position logits, avoiding wasted FLOPs on intermediate positions during decode.

### Profiling with `torch.profiler`

- Produces a **summary table** (sorted by CPU/CUDA time) and a **Chrome trace** (viewable at `ui.perfetto.dev`).
- **Trace anatomy**:
  - CPU thread: nested `aten::` operator bars ending in `cudaLaunchKernel`.
  - GPU stream: actual kernel execution.
  - Healthy trace = both rows densely filled and overlapping.
- **What the trace revealed**:
  - Baseline: `aten::item` 12 times (~45 ms in profiled run), `aten::cat` 120 times, GPU kernels ~80 ms for 12 steps.
  - Optimized: `aten::item` count = 0, GPU kernel total ~12.8 ms, CUDA runtime events dropped from 1565 → 1158.

### Dtype Experiment: bf16 vs fp32

- On H100 with this tiny model, **bf16 was slightly slower than fp32**.
- Likely cause: the model is small enough that the extra dtype conversion overhead outweighs the memory-bandwidth savings of 16-bit.
- For large models or batch inference, bf16 typically wins; the lesson is to **measure, not assume**.

---

## Assignment 3: Mini Inference Engine (Optional, Conceptual)

### Core Design Philosophy

Real inference engines (vLLM, SGLang, TensorRT-LLM, TGI) are primarily about:
1. **Keeping expensive compute busy**
2. **Managing scarce KV memory efficiently**
3. **Serving many concurrent requests fairly**

### Paged KV Memory

- Instead of contiguous KV tensors per request, memory is split into fixed-size **physical blocks**.
- Each request has a `block_table` mapping its logical positions to physical block IDs.
- Benefits: no memory fragmentation, easy preemption/eviction, and prefix sharing.

### Prefix Caching

- Only **complete blocks** are cacheable (e.g., for block size 16, a 40-token prompt caches prefixes of 16 and 32 tokens).
- `match_prefix(tokens)` returns the longest cached prefix without pinning; `lock(handle)` pins blocks for live requests.
- Reusing cached prefix blocks avoids redundant prefill computation, dramatically improving TTFT (time-to-first-token) for repeated prompts.

### Continuous Batching

- Each engine step runs one **phase-pure** batch: either prefill or decode, never both.
- The set of active requests changes over time as new requests arrive, old ones finish, and memory pressure forces preemption.

### CacheManager Ownership Rules

| State | `_ref` | `_cache_ref` | Meaning |
|-------|--------|-------------|---------|
| Free | 0 | 0 | In free pool |
| Live only | 1 | 0 | Owned by one request |
| Cached only | 1 | >0 | Evictable by LRU |
| Pinned + cached | ≥2 | >0 | Do not evict |

### Scheduler Policies

- **PREFILL_FIRST**: Prefers prefill work; good for minimizing TTFT when many new requests are arriving.
- **DECODE_FIRST**: Prefers decode-ready running requests; good for maximizing throughput when decode work dominates.
- Admission is FIFO; if the front request cannot be admitted due to memory pressure, stop admitting for that step.

### Preemption Strategy

1. Try free blocks.
2. Let `CacheManager` evict LRU cached prefixes.
3. If still insufficient, **preempt** a running request (free its blocks, return to waiting queue).

---

## Cross-Cutting Themes (Module 4)

1. **Roofline thinking generalizes**: Any kernel can be classified by its arithmetic intensity. Before optimizing, know whether you are fighting memory bandwidth or compute throughput.
2. **Kernel fusion is transformative**: `torch.compile` can turn a memory-bound loop into a compute-bound fused kernel by keeping intermediates in registers.
3. **Profile first, optimize second**: The naive inference loop looked simple but hid massive repeated work. The trace and `time_generation` numbers, not intuition, revealed the real bottleneck.
4. **KV cache dominates autoregressive inference**: For long contexts, the difference between full-sequence and cached decode is orders of magnitude. Every production inference system builds around this principle.
5. **Measure dtype choices**: bf16 is not always faster than fp32, especially for small models where conversion overhead matters.
6. **Inference engines are memory managers first**: The scheduler, cache manager, and block allocator are as important as the model forward pass. Paged memory, prefix caching, and continuous batching are the core innovations in modern serving systems.
7. **Phase-pure batches**: Prefill and decode use different kernel shapes and attention patterns. Mixing them in one batch complicates kernel selection and hurts efficiency.
8. **Reference counting and LRU are foundational**: The cache manager's `_ref` / `_cache_ref` split ensures blocks are freed exactly once and cached prefixes can be safely evicted when unpinned.
