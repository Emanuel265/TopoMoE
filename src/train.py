import argparse
from dataclasses import dataclass
import os
import time
import json
import math
import torch
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from torch.optim import AdamW

from src.model.model_wrapper import MoEGPT
from src.model.topology import get_topology

@torch.no_grad()
def evaluate(engine, dataloader, max_batches: int = 50):
    """
    Returns perplexity on at most `max_batches` batches.
    `max_batches` keeps evaluation time bounded and consistent
    across runs — use the same value for every comparison.
    """
    engine.eval()

    total_loss   = torch.tensor(0.0, device=engine.device)
    total_tokens = torch.tensor(0,   device=engine.device)

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        batch  = {k: v.to(engine.device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()

        # Same shift the training loop applies
        labels[:, :-1] = batch["input_ids"][:, 1:]
        labels[:, -1]  = -100                           # last position has no target
        labels[batch["attention_mask"] == 0] = -100     # ignore padding

        outputs = engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        # Count only non-ignored tokens so the average is correct
        n_tokens = (labels != -100).sum()
        # outputs.loss is already mean CE; recover the sum for accumulation
        total_loss   += outputs.loss.detach() * n_tokens
        total_tokens += n_tokens

    # Aggregate across all ranks
    totals = torch.stack([total_loss, total_tokens.float()])
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    total_loss, total_tokens = totals[0].item(), totals[1].item()

    perplexity = math.exp(total_loss / total_tokens)

    engine.train()
    return {"perplexity": perplexity}

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Minimal MoE GPT Benchmark")

    p.add_argument("--deepspeed", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default="topomoe/ds_config.json", help="Path to DeepSpeed config file")

    p.add_argument("--model_name", type=str, default="google/switch-base-16", help="HuggingFace model name for tokenizer")
    p.add_argument("--dataset_name", type=str, default="wikitext", help="HuggingFace dataset name")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="HuggingFace dataset configuration")

    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length for training")
    p.add_argument("--hidden", type=int, default=256, help="Hidden size of the model")
    p.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--num_layers", type=int, default=6, help="Number of moe layers")
    p.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size of the model")

    p.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU (must match ds_config)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--train_steps", type=int, default=500, help="Total number of training steps")
    p.add_argument("--aux_weight", type=float, default=0.1, help="Coefficient for MoE auxiliary loss")
    p.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps to exclude from timing")

    p.add_argument("--log_interval", type=int, default=10, help="Steps between logging training metrics")
    p.add_argument("--bench_window", type=int, default=10, help="Number of steps to average for throughput reporting")
    p.add_argument("--eval_interval", type=int, default=500, help="Steps between evaluations")

    p.add_argument("--use-topomoe", action="store_true", help="Whether to use TopoMoE expert placement")
    p.add_argument("--rebalance-step", type=int, help="Step at which to trigger a rebalance of expert placement (TopoMoE only)")

    args = p.parse_args()

    file_path = "src/ds_config.json"
    with open(file_path, "r") as f:
        ds_config = json.load(f)
    print(f"Loaded DeepSpeed config: {ds_config}")

    print(f"Using TopoMoE: {args.use_topomoe}")
    deepspeed.init_distributed()
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ── Tokenizer ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────
    SEQ_LEN = args.seq_len
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )

    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Batch size must be set here, not left at 1 with DeepSpeed controlling
    # it implicitly — otherwise effective batch size is undefined.
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    eval_dataset = load_dataset(args.dataset_name, args.dataset_config, split="validation")
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,          # ← eval dataset, not train
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=False,         # no shuffling for eval
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        pin_memory=True,
        drop_last=False,
    )

    # ── Topology ─────────────────────────────────────────
    topology = get_topology()

    # ── Model ────────────────────────────────────────────
    model = MoEGPT(
        use_topomoe=args.use_topomoe,
        topology=topology,
        token_d=ds_config["topomoe"]["token_d"],
        alpha=ds_config["topomoe"]["alpha"],
        beta=ds_config["topomoe"]["beta"],
        gamma=ds_config["topomoe"]["gamma"],
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden=args.hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=ds_config["moe"]["num_experts"],
        ep_size=ds_config["moe"]["ep_size"],
        k=ds_config["moe"]["k"],
        rank=local_rank,
        aux_weight=args.aux_weight,
    )   

    def create_moe_param_groups(model):
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

        parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}

        return split_params_into_different_moe_groups_for_optimizer(parameters)

    param_groups = create_moe_param_groups(model)
    for group in param_groups:
        count = len(group['params'])
        print(f"Group '{group['name']}': {count} params")

    optimizer = AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )


    # ── DeepSpeed engine ─────────────────────────────────
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        optimizer=optimizer,
        config=file_path,
    )

    engine.timers.log(["moe", "alltoall"])
    print(torch.distributed.get_rank(), engine.wall_clock_breakdown)

    # Verify DeepSpeed agrees on micro batch size
    assert engine.train_micro_batch_size_per_gpu() == args.batch_size, (
        f"Batch size mismatch: BATCH_SIZE={args.batch_size} but DeepSpeed "
        f"train_micro_batch_size_per_gpu={engine.train_micro_batch_size_per_gpu()}. "
        f"Update ds_config train_micro_batch_size_per_gpu to {args.batch_size}."
    )

    engine.train()

    # ── Training loop ────────────────────────────────────
    def make_dataloader(sampler, epoch):
        sampler.set_epoch(epoch)
        return iter(torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
        ))
    
    epoch = 0
    step       = 0
    data_iter = make_dataloader(sampler, epoch)

    # Per-step wall-clock times collected after warmup
    step_times: list[float] = []
    start_time = time.time()

    for batch in dataloader:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = make_dataloader(sampler, epoch)
            batch = next(data_iter)

        batch  = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[:, :-1]                        = batch["input_ids"][:, 1:]
        labels[:, -1]                         = -100
        labels[batch["attention_mask"] == 0]  = -100
        batch["labels"] = labels

        # ── Forward / backward ───────────────────────────
        t0      = time.perf_counter()
        outputs = engine(**batch)
        loss    = outputs.loss
        engine.backward(loss)
        engine.step()
        step_ms = (time.perf_counter() - t0) * 1000

        # Only collect timing after warmup
        if step >= args.warmup_steps:
            step_times.append(step_ms)
        
        # Trigger rebalance at specified step (TopoMoE only)
        if args.use_topomoe and args.rebalance_step is not None and step % args.rebalance_step == 0:
            if engine.global_rank == 0:
                print(f"\n==== Triggering expert placement rebalance at step {step} ====\n")

            model.rebalance_experts()
            

        # ── Logging ──────────────────────────────────────
        if engine.global_rank == 0 and step % args.log_interval == 0:
            if len(step_times) >= 2:
                # Use the most recent BENCH_WINDOW steps for a stable average
                window     = step_times[-args.bench_window:]
                avg_ms     = sum(window) / len(window)
                tokens_per_step = args.batch_size * args.seq_len * engine.world_size
                tps        = tokens_per_step / (avg_ms / 1000)
                tps_str    = f"{tps:,.0f}"
            else:
                avg_ms  = float("nan")
                tps_str = "warming up"

            print(
                f"Step {step:04d} | "
                f"Loss {loss.item():.4f} | "
                f"Aux {outputs.aux_loss.item():.5f} | "
                f"Step {avg_ms:.1f}ms | "
                f"Tokens/sec {tps_str}"
            )

        # ── Evaluation ───────────────────────────────────
        if step > 0 and step % args.eval_interval == 0:
            metrics = evaluate(engine, eval_dataloader)
            if engine.global_rank == 0:
                print(f"  [Eval] Step {step} | Perplexity {metrics['perplexity']:.2f}")

        if step >= args.train_steps:
            break

        if args.use_topomoe:
            engine.timers.log(["moe", "alltoall"], reset=True)

        step += 1

    deepspeed.comm.log_summary()

    # ── Final summary ─────────────────────────────────────
    if engine.global_rank == 0:
        if step_times:
            window          = step_times[-args.bench_window:]
            avg_ms          = sum(window) / len(window)
            std_ms          = (sum((t - avg_ms) ** 2 for t in window) / len(window)) ** 0.5
            tokens_per_step = args.batch_size * args.seq_len * engine.world_size
            tps             = tokens_per_step / (avg_ms / 1000)
            print(f"\n==== Benchmark complete ====")
            print(f"Total time: {time.time() - start_time}")
            print(f"Steps measured (post-warmup): {len(step_times)}")
            print(f"Step time: {avg_ms:.1f} ± {std_ms:.1f} ms  (last {len(window)} steps)")
            print(f"Throughput: {tps:,.0f} tok/s")
        else:
            print("Not enough post-warmup steps to report throughput.")

    metrics = evaluate(engine, eval_dataloader)
    if engine.global_rank == 0:
        print(f"\n==== Final Evaluation ====")
        print(f"Perplexity: {metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()