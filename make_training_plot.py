"""Render the training curves from training_log.json.

Two single-measure panels (never a shared/dual axis):
  - left:  train loss (step-level) + validation loss (per epoch) — same unit
  - right: validation BLEU per epoch, best epoch marked

Writes assets/training_curve.png.

    python make_training_plot.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette (validated data-viz reference, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
TRAIN = "#2a78d6"   # blue
VAL = "#eb6834"     # orange
BEST = "#006300"    # good/highlight

with open("training_log.json", "r", encoding="utf-8") as f:
    LOG = json.load(f)

hist = LOG["log_history"]
train = [(e["epoch"], e["loss"]) for e in hist if "loss" in e and "epoch" in e]
ev_loss = [(e["epoch"], e["eval_loss"]) for e in hist if "eval_loss" in e]
ev_bleu = [(e["epoch"], e["eval_bleu"]) for e in hist if "eval_bleu" in e]

tx, ty = zip(*train)
lx, ly = zip(*ev_loss)
bx, by = zip(*ev_bleu)
best_i = max(range(len(by)), key=lambda i: by[i])

plt.rcParams["font.family"] = "DejaVu Sans"
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.2), facecolor=SURFACE)
fig.suptitle(
    "Training curves — 3 epochs, full fine-tune",
    fontsize=12, fontweight="bold", color=INK, y=0.98,
)


def chrome(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlabel("Epoch", color=INK, fontsize=10)
    ax.set_xticks([1, 2, 3])


# --- left: loss ---
axL.plot(tx, ty, color=TRAIN, linewidth=1.6, zorder=3, label="Train loss (per 100 steps)")
axL.plot(lx, ly, color=VAL, linewidth=2, marker="o", markersize=8, zorder=4,
         label="Validation loss (per epoch)")
axL.set_ylim(0, max(ty) * 1.05)
axL.set_title("Loss", fontsize=11, color=INK, fontweight="bold", pad=8)
chrome(axL)
axL.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)

# --- right: validation BLEU ---
axR.plot(bx, by, color=TRAIN, linewidth=2, marker="o", markersize=8, zorder=3)
axR.scatter([bx[best_i]], [by[best_i]], s=150, facecolor="none",
            edgecolor=BEST, linewidth=2.2, zorder=5)
axR.annotate(
    "best: {:.4f}\n(epoch {:g})".format(by[best_i], bx[best_i]),
    xy=(bx[best_i], by[best_i]), xytext=(0, -34), textcoords="offset points",
    ha="center", fontsize=9.5, color=BEST, fontweight="bold",
)
pad = (max(by) - min(by)) * 0.6 + 0.002
axR.set_ylim(min(by) - pad, max(by) + pad)
axR.set_title("Validation BLEU", fontsize=11, color=INK, fontweight="bold", pad=8)
chrome(axR)

fig.tight_layout(rect=(0, 0, 1, 0.94))
os.makedirs("assets", exist_ok=True)
fig.savefig("assets/training_curve.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
print("Wrote assets/training_curve.png  |  best val BLEU {:.4f} @ epoch {:g}".format(
    by[best_i], bx[best_i]))
