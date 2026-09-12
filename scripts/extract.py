#!/usr/bin/env python3
"""Parse the 24 sweep logs into one JSON. Numbers are computed from logs/, never staged."""
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL = re.compile(r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)")
TRAIN = re.compile(
    r"step:\s*(\d+)\s+loss:\s*([\d.]+|nan|inf)\s+grad_norm:\s*([\d.]+|nan|inf)"
)

LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
ARMS = ["default", "mixed"]
SEEDS = [0, 1]
ROOT = Path(__file__).resolve().parent.parent

out = {}
for arm in ARMS:
    for lr in LRS:
        for seed in SEEDS:
            p = ROOT / "logs" / f"mr_{arm}_{lr}_s{seed}.log"
            if not p.exists():
                continue
            curve, val = [], None
            for raw in p.read_text(errors="ignore").splitlines():
                line = ANSI.sub("", raw)
                m = VAL.search(line)
                if m:
                    val = float(m.group(2))
                    continue
                m = TRAIN.search(line)
                if m:
                    try:
                        curve.append(
                            (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                        )
                    except ValueError:
                        pass
            out[f"{arm}|{lr}|{seed}"] = {"val": val, "curve": curve}

(ROOT / "results.json").write_text(json.dumps(out, indent=1))
n = sum(1 for v in out.values() if v["val"] is not None)
print(f"parsed {len(out)} runs, {n} with a final validation loss")
