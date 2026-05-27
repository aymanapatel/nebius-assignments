import os
from itertools import chain

import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

def setup_distributed():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        print(f"[NCCL] Distributed training initialised")
        print(f"[NCCL] World size: {world_size}")
        print(f"[NCCL] Backend: {dist.get_backend()}")
        print(f"[NCCL] Master addr: {os.environ.get('MASTER_ADDR')}")
        print(f"[NCCL] Master port: {os.environ.get('MASTER_PORT')}")

    # ── Model & tokenizer ──────────────────────────────────────────────
    MODEL_ID = os.environ.get("MODEL_ID", "facebook/opt-1.3b")  # Change via MODEL_ID in train_job.yaml; smaller models run faster.
    print(f"[Rank {rank}] Loading model: {MODEL_ID}")

    HF_TOKEN = os.environ.get("HF_TOKEN")  # Set only if using a gated/private Hugging Face model.
    hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **hf_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **hf_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(local_rank)

    # ── Dataset ────────────────────────────────────────────────────────
    train_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    eval_dataset = load_dataset("wikitext", "wikitext-2-v1", split="validation")
    block_size = min(int(os.environ.get("BLOCK_SIZE", 512)), tokenizer.model_max_length)  # Lower if GPU memory is tight.
    per_device_train_batch_size = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 8))  # Lower if OOM; raise if GPU memory allows.
    per_device_eval_batch_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 8))  # Usually match train batch size or lower.
    gradient_accumulation_steps = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))  # Raise to keep effective batch size when lowering batch size.
    dataloader_num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", 8))  # Lower if CPU/RAM is limited or workers hang.

    def tokenize(examples):
        texts = [text for text in examples["text"] if text and not text.isspace()]
        return tokenizer(texts)

    def group_texts(examples):
        concatenated = {
            key: list(chain.from_iterable(examples[key]))
            for key in examples.keys()
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {key: [] for key in concatenated.keys()}
        return {
            key: [
                values[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }

    tokenized_train = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )
    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        desc=f"Packing tokens into {block_size}-token blocks",
    )
    tokenized_eval = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation dataset",
    )
    tokenized_eval = tokenized_eval.map(
        group_texts,
        batched=True,
        desc=f"Packing validation tokens into {block_size}-token blocks",
    )

    # ── Training ───────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir="/tmp/output",
        max_steps=500,                     # Homework asks for 500 steps; lower only for quick smoke tests.
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-6,                # Tune only if changing model/batch size substantially.
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        bf16=True,                         # Good for H100/L40S; use fp16/no mixed precision only if bf16 is unsupported.
        logging_steps=10,                  # Lower for more frequent logs; higher for less log noise.
        eval_strategy="steps",
        eval_steps=50,                     # Lower for more frequent validation; higher to save time.
        save_steps=250,                    # Change if you need more/fewer checkpoints.
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        report_to="none",                 # Swap to "wandb" if desired and configured.
        ddp_find_unused_parameters=False,  # Keep false for this model unless DDP reports unused parameters.
        local_rank=local_rank,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    if rank == 0:
        print(f"[Done] Training complete — {args.max_steps} steps finished.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
