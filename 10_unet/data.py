"""
Real-data patch loader for the ISBI-2012 EM membrane set (JOURNAL.md
entry 1/3 -- fetched from the zhixuhao/unet mirror into
data/membrane/train/{image,label}/{0..29}.png, gitignored like every
other paper's data/ dir in this repo).

Label convention (verified directly, not assumed -- see JOURNAL.md entry
3): pixel value 255 = cell interior (foreground, class 1), 0 = membrane
(background, class 0); confirmed by checking the foreground fraction is
~78% (majority, consistent with large cell interiors) vs. ~22% for the
thin membrane network.

Patch geometry: input patch 68x68, network output (and therefore the
training target) is the *center* 28x28 of that same patch -- offset 20
on every side, following directly from the valid-conv/pool/upconv
arithmetic worked out in JOURNAL.md entry 4 (total spatial shrink 68->28
= 40, split symmetrically by this architecture's construction).

Train/val split is by *whole image*, not by patch, even though patches
are drawn randomly per-call -- held-out images 25-29 are never sampled
from during training. A random-patch split without this would let two
overlapping or adjacent patches from the same micrograph land in both
train and val, leaking the same membrane structures across the split.
"""
import numpy as np
from pathlib import Path
from PIL import Image

from elastic import elastic_deform

DATA_DIR = Path(__file__).parent / "data" / "membrane" / "train"
PATCH_IN = 68
PATCH_OUT = 28
CROP_OFFSET = (PATCH_IN - PATCH_OUT) // 2  # 20
N_VAL_IMAGES = 5


def load_dataset():
    images, labels = [], []
    for i in range(30):
        img = np.array(Image.open(DATA_DIR / "image" / f"{i}.png"), dtype=np.float64) / 255.0
        lbl = np.array(Image.open(DATA_DIR / "label" / f"{i}.png"))
        images.append(img)
        labels.append((lbl == 255).astype(np.int64))  # 1=cell, 0=membrane
    return images, labels


class PatchSampler:
    def __init__(self, images, labels, indices, rng):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.rng = rng

    def sample_batch(self, batch_size, augment):
        imgs = np.empty((batch_size, 1, PATCH_IN, PATCH_IN), dtype=np.float64)
        lbls = np.empty((batch_size, PATCH_OUT, PATCH_OUT), dtype=np.int64)
        for n in range(batch_size):
            idx = self.indices[self.rng.integers(len(self.indices))]
            img_full, lbl_full = self.images[idx], self.labels[idx]
            H, W = img_full.shape
            r = self.rng.integers(0, H - PATCH_IN + 1)
            c = self.rng.integers(0, W - PATCH_IN + 1)
            img_patch = img_full[r:r + PATCH_IN, c:c + PATCH_IN]
            lbl_patch = lbl_full[r:r + PATCH_IN, c:c + PATCH_IN]
            if augment:
                img_patch, lbl_patch = elastic_deform(img_patch, lbl_patch, rng=self.rng)
            imgs[n, 0] = img_patch
            lbls[n] = lbl_patch[CROP_OFFSET:CROP_OFFSET + PATCH_OUT, CROP_OFFSET:CROP_OFFSET + PATCH_OUT]
        return imgs, lbls


def make_samplers(seed=0):
    images, labels = load_dataset()
    rng = np.random.default_rng(seed)
    all_idx = np.arange(30)
    val_idx = all_idx[-N_VAL_IMAGES:]
    train_idx = all_idx[:-N_VAL_IMAGES]
    train = PatchSampler(images, labels, train_idx, np.random.default_rng(seed))
    val = PatchSampler(images, labels, val_idx, np.random.default_rng(seed + 1))
    return train, val
