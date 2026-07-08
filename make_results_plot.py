"""Render the baseline-vs-fine-tuned results chart from results.json.

Small multiples (one panel per metric — BLEU and chrF are on different scales,
so never share an axis). Bars start at zero. Reads real numbers from
results.json; writes assets/results.png.

    python make_results_plot.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# --- palette (validated data-viz reference, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#898781"   # recessive gray
FINETUNED = "#2a78d6"  # highlight blue
GOOD = "#006300"       # positive delta

with open("results.json", "r", encoding="utf-8") as f:
    R = json.load(f)

metrics = [
    {"key": "bleu", "label": "BLEU", "ymax": 0.5, "fmt": "{:.3f}"},
    {"key": "chrf", "label": "chrF", "ymax": 80.0, "fmt": "{:.1f}"},
]

plt.rcParams["font.family"] = "DejaVu Sans"

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor=SURFACE)
fig.suptitle(
    "Medical PL→EN translation — baseline vs fine-tuned (test set, 10k rows)",
    fontsize=12, fontweight="bold", color=INK, y=0.98,
)

for ax, m in zip(axes, metrics):
    b = R["baseline"][m["key"]]
    ft = R["finetuned"][m["key"]]
    d = R["delta"][m["key"]]

    ax.set_facecolor(SURFACE)
    bars = ax.bar(
        ["Baseline", "Fine-tuned"], [b, ft],
        color=[BASELINE, FINETUNED], width=0.6, zorder=3,
    )
    ax.set_ylim(0, m["ymax"])
    ax.set_title(m["label"], fontsize=11, color=INK, fontweight="bold", pad=10)

    # value labels above bars
    for rect, val in zip(bars, [b, ft]):
        ax.text(
            rect.get_x() + rect.get_width() / 2, val, "  " + m["fmt"].format(val),
            ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold",
            rotation=0,
        )

    # delta annotation
    ax.text(
        0.5, 0.92, "Δ +{}".format(m["fmt"].format(d)),
        transform=ax.transAxes, ha="center", va="top",
        fontsize=11, color=GOOD, fontweight="bold",
    )

    # chrome: recessive grid, no top/right spines
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

fig.tight_layout(rect=(0, 0, 1, 0.94))
os.makedirs("assets", exist_ok=True)
fig.savefig("assets/results.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
print("Wrote assets/results.png  |  BLEU +{:.3f}  chrF +{:.2f}".format(
    R["delta"]["bleu"], R["delta"]["chrf"]))
