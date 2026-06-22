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

## Cross-Cutting Themes

1. **Infrastructure as code**: The entire training pipeline — from Docker image to Kubernetes job spec — is version-controlled and reproducible.
2. **Environment-driven configuration**: All tunable parameters live in `train_job.yaml`, not hardcoded in Python, enabling rapid experimentation without code changes.
3. **Observability by default**: NCCL debug logging (`NCCL_DEBUG=INFO`), training metrics every 10 steps, and explicit node IP reporting make distributed debugging tractable.
4. **Container portability**: The same Docker image runs locally (for debugging) and on Nebius MK8S (for scale), eliminating "works on my machine" issues.
5. **SkyPilot abstraction**: Hides Kubernetes complexity — you declare what you need (2 nodes, H100:1, 60GB+ RAM) and SkyPilot handles pod scheduling, networking, and `torchrun` coordination.
6. **bf16 as default**: Modern GPUs (H100, L40S) support bfloat16 natively, offering fp32 dynamic range with fp16 memory footprint — the sweet spot for LLM training.
