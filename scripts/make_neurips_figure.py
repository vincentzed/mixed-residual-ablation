# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Figure for the mixed-residual ablation, technical-report style.

Minimalist chart: axis labels, a terse panel title, a legend, error bars. No
annotations and no narrative -- the figure caption in the surrounding text
carries the interpretation. Numbers are read from results.json (parsed from
logs/).
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((ROOT / "results.json").read_text())

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
KEYS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
SEEDS = [0, 1]
BEST = "3e-3"
FROM_STEP = 300

# Colorblind-safe pair, fixed order (blue = proposed, green = baseline).
BLUE, GREEN = "#2a78d6", "#008300"
INK, GRID = "#1a1a19", "#d9d9d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
})


def vals(arm, k):
    return [R[f"{arm}|{k}|{s}"]["val"] for s in SEEDS]


def curve(arm):
    per_seed = [{r[0]: r[1] for r in R[f"{arm}|{BEST}|{s}"]["curve"]} for s in SEEDS]
    steps = sorted(x for x in per_seed[0] if x >= FROM_STEP)
    return steps, [statistics.mean(c[x] for c in per_seed) for x in steps]


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), dpi=400)

# (a) final validation loss against learning rate; bars span the two seeds
ax = axes[0]
for arm, color, label in (("mixed", BLUE, "Mixed Residual"),
                          ("default", GREEN, "Baseline Residual")):
    means = [statistics.mean(vals(arm, k)) for k in KEYS]
    lo = [m - min(vals(arm, k)) for m, k in zip(means, KEYS)]
    hi = [max(vals(arm, k)) - m for m, k in zip(means, KEYS)]
    ax.errorbar(LRS, means, yerr=[lo, hi], color=color, marker="o",
                capsize=2, elinewidth=0.8, capthick=0.8, label=label, zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Final Validation Loss")
ax.set_title("Final Validation Loss vs. Learning Rate", fontsize=8.5, pad=6)
ax.legend(frameon=False, loc="upper center")

# (b) difference of the two arms; the effect is smaller than the axis range in (a)
ax = axes[1]
diff = [statistics.mean(vals("mixed", k)) - statistics.mean(vals("default", k))
        for k in KEYS]
ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
ax.plot(LRS, diff, color=INK, marker="o", zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Loss Difference")
ax.set_title("Loss Difference (Mixed - Baseline)", fontsize=8.5, pad=6)

# (c) training loss at the learning rate that minimises both arms
ax = axes[2]
steps, y_mixed = curve("mixed")
_, y_default = curve("default")
ax.plot(steps, y_mixed, color=BLUE, label="Mixed Residual", zorder=3)
ax.plot(steps, y_default, color=GREEN, label="Baseline Residual", zorder=3)
ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss at Learning Rate 0.003", fontsize=8.5, pad=6)
ax.legend(frameon=False)

for i, ax in enumerate(axes):
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, width=0.8)
    ax.text(0.5, -0.32, f"({'abc'[i]})", transform=ax.transAxes,
            ha="center", va="top", color=INK)

fig.tight_layout(w_pad=2.0)
out = FIG / "mixed_residual.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("wrote", out, "and", out.with_suffix(".pdf"))
