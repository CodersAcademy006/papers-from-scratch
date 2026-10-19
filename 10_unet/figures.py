"""
Figures for JOURNAL.md / the repo's docs site. Static PNGs (matches
every prior paper's figures/ convention), built per dataviz skill rules
that still apply to a static chart: one hue per series picked by what
it IS (not position), no dual-axis, legible direct labels, status colors
from the repo's validated reference palette rather than eyeballed hex.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from weight_map import compute_weight_map
from elastic import elastic_deform
from data import make_samplers, PATCH_IN, PATCH_OUT, CROP_OFFSET

FIG_DIR = Path(__file__).parent / "figures"

# fixed categorical slots (train vs. val is an identity distinction, not
# a magnitude ramp) -- blue/orange pair, consistent across every figure
# in this folder rather than re-picked per chart.
C_TRAIN = "#3b6fb6"
C_VAL = "#d98324"
STATUS_GOOD = "#0ca30c"


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_training_curve(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(history["epoch"], history["train_loss"], color=C_TRAIN, linewidth=2, label="train")
    ax1.plot(history["epoch"], history["val_loss"], color=C_VAL, linewidth=2, label="val")
    ax1.set_ylabel("weighted pixel CE loss (Eqn.1)")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title("10_unet training -- loss and pixel accuracy per epoch")

    ax2.plot(history["epoch"], history["val_acc"], color=C_VAL, linewidth=2)
    baseline = 0.78  # real-data majority-class fraction, JOURNAL.md entry 3
    ax2.axhline(baseline, color="#888888", linestyle="--", linewidth=1)
    ax2.text(history["epoch"][-1], baseline, " majority-class baseline", fontsize=7,
              color="#888888", va="bottom", ha="right")
    ax2.set_ylabel("val pixel accuracy")
    ax2.set_xlabel("epoch")
    ax2.spines[["top", "right"]].set_visible(False)

    save(fig, "training_curve.png")


def plot_sample_predictions(model, val_sampler, n=4, seed=123):
    rng = np.random.default_rng(seed)
    saved_rng = val_sampler.rng
    val_sampler.rng = rng
    imgs, lbls = val_sampler.sample_batch(n, augment=False)
    val_sampler.rng = saved_rng

    logits = model.forward(imgs)
    preds = logits.argmax(axis=1)

    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    col_titles = ["input patch (68x68)", "ground truth (28x28)", "prediction (28x28)"]
    for i in range(n):
        axes[i, 0].imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(lbls[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(preds[i], cmap="gray", vmin=0, vmax=1)
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=9)
    fig.suptitle("10_unet sample predictions on held-out real EM patches (epoch 30)")
    save(fig, "sample_predictions.png")


def plot_weight_map_example():
    # Hand-picked patch, not a random draw: JOURNAL.md entry 3 verified
    # this exact region (label 0, rows/cols 200:268) has 8 separate cell
    # components with real touching-membrane gaps (median d1+d2 ~5.2px).
    # A random patch frequently lands on a single uninterrupted cell
    # interior, where Eqn.2's second term is correctly ~0 everywhere --
    # a less useful illustration of the mechanism this figure exists to
    # show, so the example is chosen for being representative of what
    # the weight map is *for*, not cherry-picked for a flattering result.
    train_sampler, _ = make_samplers(seed=0)
    img = train_sampler.images[0][200:268, 200:268]
    lbl = train_sampler.labels[0][200:268, 200:268]
    w = compute_weight_map(lbl.astype(bool))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("EM patch (68x68)", fontsize=9)
    axes[1].imshow(lbl, cmap="gray")
    axes[1].set_title("ground truth (1=cell, 0=membrane)", fontsize=9)
    im = axes[2].imshow(w, cmap="viridis")
    axes[2].set_title("Eqn.2 weight map w(x)", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Eqn.2 border-weighting on a real membrane patch -- "
                  "high weight exactly where two cells nearly touch")
    save(fig, "weight_map_example.png")


def plot_elastic_example():
    _, val_sampler = make_samplers(seed=1)
    imgs, lbls_out = val_sampler.sample_batch(1, augment=False)
    # need the full 68x68 label (not the cropped 28x28 target) to show
    # the warp consistently on both image and label
    img_full = val_sampler.images[val_sampler.indices[0]]
    lbl_full = val_sampler.labels[val_sampler.indices[0]]
    r, c = 100, 150
    img_patch = img_full[r:r + PATCH_IN, c:c + PATCH_IN]
    lbl_patch = lbl_full[r:r + PATCH_IN, c:c + PATCH_IN]
    rng = np.random.default_rng(7)
    warped_img, warped_lbl = elastic_deform(img_patch, lbl_patch, rng=rng)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes[0, 0].imshow(img_patch, cmap="gray")
    axes[0, 0].set_title("original image", fontsize=9)
    axes[0, 1].imshow(warped_img, cmap="gray")
    axes[0, 1].set_title("elastically deformed", fontsize=9)
    axes[1, 0].imshow(lbl_patch, cmap="gray")
    axes[1, 0].set_title("original label", fontsize=9)
    axes[1, 1].imshow(warped_lbl, cmap="gray")
    axes[1, 1].set_title("deformed label (nearest-sampled)", fontsize=9)
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Sec.3.1 elastic deformation (3x3 grid, std=4px) on a real patch")
    save(fig, "elastic_deformation_example.png")


if __name__ == "__main__":
    plot_weight_map_example()
    plot_elastic_example()
