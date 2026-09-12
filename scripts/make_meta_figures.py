# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "typer", "pydantic", "pillow"]
# ///
"""Mixed-residual ablation figures in the meta-style-charts house style.

Look: skills/meta-style-charts/scripts/hybrid_plot.py, DARK theme -- Meta's card
geometry with humans& tokens mapped onto it (warm near-black, teal hero, rust
pill) rather than the literal Meta black/blue clone.

Honesty rules: skills/bench-graphs -- titles state the finding in words, no
abbreviations anywhere in the figure, the caveat sits ON the figure as a
sentence, and the footnote carries the measurement conditions.

Numbers are read from results.json, which is parsed from logs/.
"""

import json
import statistics
from pathlib import Path

import hybrid_plot as hp

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
hp._register_fonts(ROOT / "fonts")
# The skill's MONO is the Nerd Font build, fetched from ~/Library/Fonts on macOS.
# On this box only the plain face is installed; point at it so numerals do not
# silently fall back to DejaVu.
hp.MONO = "Google Sans Code"

R = json.loads((ROOT / "results.json").read_text())
LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
WORDS = {  # no abbreviations, and no bare arithmetic, inside the figure
    "1e-4": "0.0001", "3e-4": "0.0003", "1e-3": "0.001",
    "3e-3": "0.003", "1e-2": "0.01", "3e-2": "0.03",
}
SEEDS = [0, 1]
THEME = hp.DARK


def mean_val(arm, lr):
    return statistics.mean(R[f"{arm}|{lr}|{s}"]["val"] for s in SEEDS)


spreads = [
    abs(R[f"{a}|{l}|0"]["val"] - R[f"{a}|{l}|1"]["val"]) for a in ("default", "mixed") for l in LRS
]
noise = statistics.median(spreads)

# ---- Card 1: the sweep, hero vs context, delta carried by the pill ----------
groups = []
for lr in LRS:
    d, m = mean_val("default", lr), mean_val("mixed", lr)
    # Full-precision bars; the label format is set on the card via value_fmt.
    groups.append(
        hp.Group(label=WORDS[lr], hero=m, context=d, badge=f"{m - d:+.3f}")
    )

hp.meta_card(
    groups,
    FIG / "meta_val_loss_by_learning_rate.png",
    title_parts=[("The mixed residual wins ", False), ("only", True),
                 (" while training is stable", False)],
    # Single line on purpose: the card places text with va="baseline", so a
    # manual "\n" anchors on the LAST line and rides the first one up into the
    # title's descenders.
    subtitle=("Final validation loss at each learning rate, two seeds. "
              "A negative badge means mixed wins."),
    ylabel="final validation loss (lower is better)",
    hero_name=("Mixed residual", "half of it from two layers back"),
    context_name=("Default residual", "previous layer only"),
    theme=THEME,
    kicker="8-layer transformer, 78.6 million tokens per run",
    footnote=(f"Seed-to-seed noise is {noise:.3f} loss. Each run: 1,200 steps of "
              "65,536 tokens on one B300, AdamW, cosine schedule to zero."),
    legend_loc="bottom",
    value_fmt="{:.2f}",
    show_badge_legend=False,  # the subtitle already defines the badge
)

# ---- Card 2: training curves at the learning rate that was best for both ----
best_lr = "3e-3"
FROM = 300  # the opening drop from 11.3 otherwise flattens the whole range


def avg_curve(arm):
    per_seed = [{r[0]: r[1] for r in R[f"{arm}|{best_lr}|{s}"]["curve"]} for s in SEEDS]
    steps = sorted(k for k in per_seed[0] if k >= FROM)
    return steps, [statistics.mean(c[k] for c in per_seed) for k in steps]


steps, y_mixed = avg_curve("mixed")
_, y_default = avg_curve("default")

hp.meta_line_card(
    [float(s) for s in steps],
    {"Mixed residual": y_mixed, "Default residual": y_default},
    FIG / "meta_training_curve.png",
    title_parts=[("The gap opens ", False), ("late", True), (" in training", False)],
    subtitle=("Training loss at learning rate 0.003, the best setting for both models, "
              "over two seeds."),
    ylabel="training loss (lower is better)",
    xlabel="training step",
    hero="Mixed residual",
    theme=THEME,
    kicker="8-layer transformer, learning rate 0.003",
    footnote=("First 300 steps cut: the opening drop flattens the rest. Logged every "
              "100 steps; same code, data order and seed."),
)
print("wrote", FIG / "meta_val_loss_by_learning_rate.png")
print("wrote", FIG / "meta_training_curve.png")
