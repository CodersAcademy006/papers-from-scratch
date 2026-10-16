"""
CIFAR-10 loader, duplicated from 08_vggnet/cifar_data.py (self-contained-
folder convention), plus train-time augmentation this repo hasn't needed
before. Preprocessing (per-pixel mean subtraction) matches Section 4.2.
"""
import os
import pickle
import subprocess
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_PATH = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
EXTRACT_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")


def _ensure_downloaded():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(TAR_PATH):
            subprocess.run(["curl", "-sL", "-f", URL, "-o", TAR_PATH], check=True)
        with tarfile.open(TAR_PATH) as tf:
            tf.extractall(DATA_DIR)


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float64)
    y = np.array(d[b"labels"], dtype=np.int64)
    return X, y


def one_hot(labels, n_classes=10):
    out = np.zeros((labels.shape[0], n_classes))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def load_cifar10():
    _ensure_downloaded()
    X_train = []
    y_train = []
    for i in range(1, 6):
        X, y = _load_batch(os.path.join(EXTRACT_DIR, f"data_batch_{i}"))
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = _load_batch(os.path.join(EXTRACT_DIR, "test_batch"))

    mean_image = X_train.mean(axis=0)
    X_train -= mean_image
    X_test -= mean_image
    return X_train, y_train, X_test, y_test


def augment_batch(X, rng):
    """Section 4.2: 'pad 4 pixels on each side ... a 32x32 crop is
    randomly sampled from the padded image or its horizontal flip.'
    Zero-pad (paper doesn't specify the pad mode; [24]'s own augmentation
    recipe, which Section 4.2 cites, uses zero-padding -- JOURNAL.md
    entry 3). Train-time only -- Section 4.2: test uses the single
    original 32x32 view, unaugmented."""
    N, C, H, W = X.shape
    padded = np.pad(X, ((0, 0), (0, 0), (4, 4), (4, 4)))
    out = np.empty_like(X)
    tops = rng.integers(0, 2 * 4 + 1, size=N)
    lefts = rng.integers(0, 2 * 4 + 1, size=N)
    flips = rng.random(N) < 0.5
    for i in range(N):
        crop = padded[i, :, tops[i]:tops[i] + H, lefts[i]:lefts[i] + W]
        out[i] = crop[:, :, ::-1] if flips[i] else crop
    return out
