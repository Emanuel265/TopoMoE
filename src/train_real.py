import os
import time
import json
import math
import torch
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from torch.optim import AdamW

import argparse

from topomoe.model.model_wrapper import ExpertPlacementManager, MoEGPT

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

        labels[:, :-1] = batch["input_ids"][:, 1:]
        labels[:, -1]  = -100
        labels[batch["attention_mask"] == 0] = -100

        outputs = engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        n_tokens = (labels != -100).sum()
        total_loss   += outputs.loss.detach() * n_tokens
        total_tokens += n_tokens

    totals = torch.stack([total_loss, total_tokens.float()])
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    total_loss, total_tokens = totals[0].item(), totals[1].item()

    perplexity = math.exp(total_loss / total_tokens)

    engine.train()
    return {"perplexity": perplexity}

DATASET_NAME   = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

SEED       = 42
SEQ_LEN    = 128
HIDDEN     = 256
NUM_HEADS  = 4
NUM_LAYERS = 6
VOCAB_SIZE = 50257

# Batch size: set explicitly here and must match ds_config
# train_micro_batch_size_per_gpu so DeepSpeed doesn't silently
# override it.  Effective batch = BATCH_SIZE * grad_accum * world_size.
BATCH_SIZE = 32

LR          = 1e-4
TRAIN_STEPS = 500
AUX_WEIGHT  = 0.1   # load-balancing loss coefficient — must be identical
                     # across Standard MoE and TopoMoE runs to isolate the
                     # effect of routing topology

# Throughput measurement:
# Discard the first WARMUP_STEPS steps entirely (CUDA/NCCL init, kernel
# compilation). Then average over BENCH_WINDOW steps for a stable number.
WARMUP_STEPS = 10

LOG_INTERVAL  = 10
BENCH_WINDOW = LOG_INTERVAL
EVAL_INTERVAL = TRAIN_STEPS

def main():
    p = argparse.ArgumentParser(description="Minimal MoE GPT Benchmark")

    p.add_argument("--deepspeed", action="store_true")
    p.add_argument("--deepspeed_config", type=str)

    p.add_argument("--model-name",     type=str, default="google/switch-base-16")
    p.add_argument("--dataset-name",   type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")

    # p.add_argument("--ds-config", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--seq-len",    type=int, default=128)
    p.add_argument("--hidden",     type=int, default=256)
    p.add_argument("--num-heads",  type=int, default=4)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=50257)

    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--aux-weight",  type=float, default=0.0)
    p.add_argument("--lr",  type=float, default=1e-4)

    p.add_argument("--warmup-steps", type=int, default=10)

    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--bench-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=500)

    p.add_argument("--topomoe", action="store_true", help="Whether to use TopoMoE expert placement")

    args = p.parse_args()

    print(f"Using TopoMoE: {args.topomoe}")
    if args.topomoe:
        from deepspeed.moe.expert_placement_integrated import ExpertPlacementConfig

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

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
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    eval_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=False,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────
    placement_manager_config = None
    if args.topomoe:
        placement_manager_config = ExpertPlacementConfig(
            rebalance_steps= [5],         
            max_swap_iterations= 60,        
            min_improvement_frac= 0.005,   
            affinity_decay = 1,          
            migrate_timeout_sec = 60.0,
            alpha = 0.3,                   
            beta = 0.3,
            gamma = 0.4,
            tau_balance = 0.01,
        )
    model = MoEGPT(
        use_topomoe=args.topomoe,
        placement_manager_config=placement_manager_config,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden=args.hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=ds_config["moe"]["num_experts"],
        ep_size=ds_config["moe"]["ep_size"],
        k=ds_config["moe"]["k"],
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

    manager = None 
    if args.topomoe:
        manager = ExpertPlacementManager(
            model, 
            None, 
            ds_config["moe"]["num_experts"] // ds_config["moe"]["ep_size"], 
            ds_config["moe"]["ep_size"], 
            placement_manager_config, 
            device,
            optimizer=optimizer
        )


    # ── DeepSpeed engine ─────────────────────────────────
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        optimizer=optimizer,
        config=args.deepspeed_config,
    )

    if manager is not None:
        ep_group = model.layers[0].moe.deepspeed_moe.ep_group
        manager.set_ep_group(ep_group)

    engine.timers.log(["moe", "alltoall"])
    print(torch.distributed.get_rank(), engine.wall_clock_breakdown)

    # Verify DeepSpeed agrees on micro batch size
    assert engine.train_micro_batch_size_per_gpu() == args.batch_size, (
        f"Batch size mismatch: args.batch_size={args.batch_size} but DeepSpeed "
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

    step_times: list[float] = []
    start_time = time.time()

    # for batch in dataloader:
    #     try:
    #         batch = next(data_iter)
    #     except StopIteration:
    #         epoch += 1
    #         data_iter = make_dataloader(sampler, epoch)
    #         batch = next(data_iter)

    #     batch  = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    #     labels = batch["input_ids"].clone()
    #     labels[:, :-1]                        = batch["input_ids"][:, 1:]
    #     labels[:, -1]                         = -100
    #     labels[batch["attention_mask"] == 0]  = -100
    #     batch["labels"] = labels

    #     # ── Forward / backward ───────────────────────────
    #     t0      = time.perf_counter()
    #     outputs = engine(**batch)
    #     loss    = outputs.loss
    #     engine.backward(loss)
    #     engine.step()
    #     step_ms = (time.perf_counter() - t0) * 1000

    #     if manager is not None:
    #         manager.step(step)

    #     if step >= args.warmup_steps:
    #         step_times.append(step_ms)
    
    #     # ── Logging ──────────────────────────────────────
    #     if engine.global_rank == 0 and step % args.log_interval == 0:
    #         if len(step_times) >= 2:
    #             window     = step_times[-args.bench_interval:]
    #             avg_ms     = sum(window) / len(window)
    #             tokens_per_step = args.batch_size * args.seq_len * engine.world_size
    #             tps        = tokens_per_step / (avg_ms / 1000)
    #             tps_str    = f"{tps:,.0f}"
    #         else:
    #             avg_ms  = float("nan")
    #             tps_str = "warming up"

    #         print(
    #             f"Step {step:04d} | "
    #             f"Loss {loss.item():.4f} | "
    #             f"Aux {outputs.aux_loss.item():.5f} | "
    #             f"Step {avg_ms:.1f}ms | "
    #             f"Tokens/sec {tps_str}"
    #         )

    #     # ── Evaluation ───────────────────────────────────
    #     if step > 0 and step % args.eval_interval == 0:
    #         metrics = evaluate(engine, eval_dataloader)
    #         if engine.global_rank == 0:
    #             print(f"  [Eval] Step {step} | Perplexity {metrics['perplexity']:.2f}")

    #     if step >= args.train_steps:
    #         break

    #     if args.topomoe and step in placement_manager_config.rebalance_steps:
    #         engine.timers.log(["moe", "alltoall"], reset=True)

    #     step += 1
    
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

        if manager is not None:
            manager.step(step)

        # Only collect timing after warmup
        if step >= WARMUP_STEPS:
            step_times.append(step_ms)

        # ── Logging ──────────────────────────────────────
        if engine.global_rank == 0 and step % LOG_INTERVAL == 0:
            if len(step_times) >= 2:
                # Use the most recent BENCH_WINDOW steps for a stable average
                window     = step_times[-BENCH_WINDOW:]
                avg_ms     = sum(window) / len(window)
                tokens_per_step = BATCH_SIZE * SEQ_LEN * engine.world_size
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
        if step > 0 and step % EVAL_INTERVAL == 0:
            metrics = evaluate(engine, eval_dataloader)
            if engine.global_rank == 0:
                print(f"  [Eval] Step {step} | Perplexity {metrics['perplexity']:.2f}")

        if step >= TRAIN_STEPS:
            break

        if True and step in placement_manager_config.rebalance_steps:
            engine.timers.log(["moe", "alltoall"], reset=True)

        step += 1

    deepspeed.comm.log_summary()

    # ── Final summary ─────────────────────────────────────
    if engine.global_rank == 0:
        if step_times:
            window          = step_times[-args.bench_interval:]
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