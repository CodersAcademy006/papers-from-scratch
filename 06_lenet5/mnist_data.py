"""MNIST loader, duplicated from 05_layernorm's copy (self-contained-folder
convention, see 02_dropout/JOURNAL.md entry 5), with LeNet-5's own
32x32 padding and pixel normalization (JOURNAL.md entry 2): background
-0.1, foreground 1.175, per the paper's own stated values."""
import gzip
import os
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

BACKGROUND = -0.1
FOREGROUND = 1.175


def _download(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        subprocess.run(["curl", "-sL", "-f", BASE_URL + name, "-o", path], check=True)
    return path


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic number {magic} for images file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        raw = data.reshape(n, rows, cols).astype(np.float64) / 255.0  # 0=white, 1=black
        padded = np.full((n, 1, 32, 32), BACKGROUND, dtype=np.float64)
        padded[:, 0, 2:30, 2:30] = BACKGROUND + raw * (FOREGROUND - BACKGROUND)
        return padded


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic number {magic} for labels file"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def one_hot(labels, n_classes=10):
    out = np.zeros((labels.shape[0], n_classes))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def load_mnist():
    paths = {k: _download(v) for k, v in FILES.items()}
    X_train = _read_images(paths["train_images"])
    y_train = _read_labels(paths["train_labels"])
    X_test = _read_images(paths["test_images"])
    y_test = _read_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test
