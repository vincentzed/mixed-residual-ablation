#!/usr/bin/env python3
"""Replay the 24 completed sweep runs into Weights & Biases.

No retraining: the histories are read back out of logs/ (via results.json) and
re-emitted as W&B runs so the sweep is browsable and comparable there.
"""
import json
import os
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results.json").read_text())
PROJECT = os.environ.get("WANDB_PROJECT", "mixed-residual-ablation")
ENTITY = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity

LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
ARMS = ["default", "mixed"]
SEEDS = [0, 1]

# Held fixed across every run; recorded so the sweep is self-describing in W&B.
COMMON = dict(
    architecture="cs336-a1 default, depth 8",
    d_model=512,
    n_layers=8,
    n_heads=16,
    d_ff=1344,
    context_length=256,
    vocab_size=50257,
    params_total=76_400_000,
    params_non_embedding=24_900_000,
    tokens_per_step=65536,
    steps=1200,
    tokens_total=78_643_200,
    optimizer="AdamW",
    lr_schedule="cosine to zero, 100 warmup",
    grad_clip=1.0,
    dataset="allenai/c4 en",
    tokenizer="gpt2",
    framework="torchtitan",
    hardware="1x NVIDIA B300",
)

for arm in ARMS:
    for lr in LRS:
        for seed in SEEDS:
            rec = R[f"{arm}|{lr}|{seed}"]
            run = wandb.init(
                entity=ENTITY,
                project=PROJECT,
                name=f"{arm}-lr{lr}-s{seed}",
                group=arm,
                job_type=f"lr{lr}",
                tags=[arm, f"lr{lr}", f"seed{seed}"],
                config={**COMMON, "residual": arm, "lr": float(lr), "seed": seed},
                reinit=True,
                settings=wandb.Settings(silent=True),
            )
            for step, loss, gnorm in rec["curve"]:
                run.log({"train/loss": loss, "train/grad_norm": gnorm}, step=step)
            if rec["val"] is not None:
                run.log({"val/loss": rec["val"]}, step=1200)
                run.summary["final_val_loss"] = rec["val"]
            run.summary["final_train_loss"] = rec["curve"][-1][1]
            run.finish()
            print(f"  {arm:8s} lr={lr:5s} seed={seed}  val={rec['val']:.4f}", flush=True)

print(f"\nhttps://wandb.ai/{ENTITY}/{PROJECT}")
