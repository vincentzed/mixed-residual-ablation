#!/usr/bin/env python3
"""Figures for the mixed-residual ablation.

Charts: vega-lite blog-cards via thread_visuals (humansand light palette).
Table:  rich terminal panel on the dark complement.
"""
import json
import statistics
from pathlib import Path

import thread_visuals as tv

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((ROOT / "results.json").read_text())

LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
SEEDS = [0, 1]


def mean_val(arm, lr):
    return statistics.mean(R[f"{arm}|{lr}|{s}"]["val"] for s in SEEDS)


# ---- Figure 1: final validation loss at every learning rate -----------------
tv.grouped_bar_plot(
    groups=LRS,
    series={
        "mixed residual": [mean_val("mixed", lr) for lr in LRS],
        "default residual": [mean_val("default", lr) for lr in LRS],
    },
    out_png=FIG / "val_loss_by_learning_rate.png",
    unit="final validation loss (mean of 2 seeds)",
    title="Final validation loss at each learning rate",
    xlabel="learning rate",
    label_format=".3f",
    mute_second=True,
)

# ---- Figure 2: the delta, where a zero baseline is the whole point ----------
tv.signed_bar_plot(
    items=[(lr, mean_val("mixed", lr) - mean_val("default", lr)) for lr in LRS],
    out_png=FIG / "loss_difference_by_learning_rate.png",
    unit="mixed minus default, final validation loss (negative = mixed better)",
    title="Change in final validation loss from the mixed residual",
    xlabel="learning rate",
)

# ---- Figure 3: training curves at the shared optimum ------------------------
best_lr = "3e-3"
FROM_STEP = 300  # the 11.3 -> 6.3 warmup drop otherwise flattens the whole range


def avg_curve(arm):
    per_seed = [dict(R[f"{arm}|{best_lr}|{s}"]["curve"]) for s in SEEDS]
    steps = sorted(k for k in per_seed[0] if k >= FROM_STEP)
    return steps, [statistics.mean(c[k] for c in per_seed) for k in steps]


steps, y_m = avg_curve("mixed")
_, y_d = avg_curve("default")
tv.line_plot(
    x=steps,
    series={"mixed residual": y_m, "default residual": y_d},
    out_png=FIG / "training_curve_at_best_lr.png",
    xlabel="training step",
    unit=f"training loss from step {FROM_STEP}, learning rate {best_lr}, mean of 2 seeds",
    title=f"Training loss at learning rate {best_lr}",
    mute_rest=True,
    zero=False,
    y_format=".1f",  # default ~s rounds to integers and prints "5" six times
)

# ---- Figure 3: the numbers, as a terminal panel ----------------------------
rows = []
for lr in LRS:
    d, m = mean_val("default", lr), mean_val("mixed", lr)
    rows.append([lr, f"{d:.4f}", f"{m:.4f}", f"{m - d:+.4f}"])
panel = tv.results_table(
    ["learning rate", "default", "mixed", "mixed - default"],
    rows,
    title="Final validation loss, mean of 2 seeds",
)
tv.save(panel, FIG / "results_table.svg", width=64, title="mixed-residual ablation")
print("figures written to", FIG)
