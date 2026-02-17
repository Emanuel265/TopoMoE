import os
import time
import torch
import deepspeed
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    set_seed
)
from torch.optim import AdamW
import math
from torch.nn.functional import log_softmax

from deepspeed.moe.utils import is_moe_param

from TopoMoE.tests.unit import SimpleMoEModel

@torch.no_grad()
def evaluate(engine, dataloader):
    engine.eval()

    total_loss = 0.0
    total_tokens = 0

    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for batch in dataloader:
        batch = {k: v.to(engine.device) for k, v in batch.items()}
        labels = batch["input_ids"]

        outputs = engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

        preds = torch.argmax(logits, dim=-1)

        mask = labels != -100
        correct += (preds[mask] == labels[mask]).sum().item()

        tp += (preds[mask] == labels[mask]).sum().item()
        fp += (preds[mask] != labels[mask]).sum().item()
        fn += fp  # symmetric for multiclass token prediction

    # Reduce across ranks
    totals = torch.tensor(
        [total_loss, total_tokens, correct, tp, fp, fn],
        device=engine.device
    )
    torch.distributed.all_reduce(totals)

    total_loss, total_tokens, correct, tp, fp, fn = totals.tolist()

    accuracy = correct / total_tokens
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    perplexity = math.exp(total_loss / total_tokens)

    engine.train()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "perplexity": perplexity
    }

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "google/switch-base-16"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

SEQ_LEN = 512
TRAIN_STEPS = 200
WARMUP_STEPS = 10
LOG_INTERVAL = 20
SEED = 42


# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Dataset
    # -------------------------
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length"
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # effective batch size controlled by DeepSpeed
        shuffle=True,
        pin_memory=True
    )

    eval_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="validation"
    )

    eval_dataset = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # -------------------------
    # Model
    # -------------------------
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.bfloat16
    # )
    model = SimpleMoEModel(hidden_dim=768, num_experts=16, ep_size=2)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-5,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )

    print("[CUSTOM DEBUG] Model and optimizer initialized")


    # -------------------------
    # DeepSpeed init
    # -------------------------
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config="ds_config.json"
    )

    engine.train()

    # -------------------------
    # Training loop
    # -------------------------
    step = 0
    total_tokens = 0
    start_time = time.time()

    for batch in dataloader:
        if step >= TRAIN_STEPS:
            break

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[:, :-1] = batch["input_ids"][:, 1:]
        labels[:, -1] = tokenizer.pad_token_id
        batch["labels"] = labels

        batch["labels"][batch["attention_mask"] == 0] = -100


        step_start = time.time()
        outputs = engine(**batch)
        loss = outputs.loss

        engine.backward(loss)
        engine.step()

        step_time = time.time() - step_start
        tokens = batch["input_ids"].numel()
        total_tokens += tokens

        if step >= WARMUP_STEPS and engine.global_rank == 0 and step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            tps = total_tokens * engine.world_size / elapsed

            print(
                f"Step {step:04d} | "
                f"Loss {loss.item():.4f} | "
                f"Step time {step_time:.3f}s | "
                f"Tokens/sec {tps:,.0f}"
            )

        step += 1

    if engine.global_rank == 0:
        total_time = time.time() - start_time
        print("\n==== Benchmark complete ====")
        print(f"Total steps: {step}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg tokens/sec: {total_tokens / total_time:,.0f}")

    metrics = evaluate(engine, eval_dataloader)

    if engine.global_rank == 0:
        print("\n==== Evaluation Results ====")
        print(f"Accuracy     : {metrics['accuracy']:.4f}")
        print(f"F1 score     : {metrics['f1']:.4f}")
        print(f"Perplexity   : {metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
