"""
Section 6.6's permutation-invariant MNIST comparison: Baseline / BatchNorm
/ LayerNorm, at batch size 128 AND batch size 4 -- the paper's own direct
test of "layer normalization is robust to the batch-sizes" (quoted in
JOURNAL.md entry 3).

Uses a training subset rather than the full 55000 examples (see EPOCHS/
TRAIN_SUBSET below): batch size 4 means ~14x more optimizer steps per
epoch than batch size 128 at the same data volume, and this is 6 full
training runs (3 norm modes x 2 batch sizes), not 1.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp_norm import NormMLP
from mnist_data import load_mnist, one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ARCH = [784, 1000, 1000, 10]
TRAIN_SUBSET = 5000  # scope deviation from the paper's 55000, see JOURNAL.md entry 7
EPOCHS = 10
LR = 0.01


def train(norm_mode, batch_size, X_train, D_train, y_train_int, X_test, y_test_int, seed, log_path):
    net = NormMLP(ARCH, norm_mode=norm_mode, weight_std=0.01, seed=seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    history = []  # (epoch, train_nll, test_err)

    with open(log_path, "w") as log:
        log.write("epoch,train_nll,test_err\n")
        for epoch in range(EPOCHS):
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, Db = X_train[idx], D_train[idx]
                activations, caches = net.forward_train(Xb)
                loss = net.loss(activations[-1], Db)
                grads_W, grads_b, grads_norm = net.backward(activations, caches, Db)
                net.step(grads_W, grads_b, grads_norm, LR)
                total_loss += loss / Xb.shape[0]
                n_batches += 1
            avg_nll = total_loss / n_batches
            preds = net.forward_eval(X_test)
            test_err = float(np.mean(np.argmax(preds, axis=1) != y_test_int))
            history.append((epoch, avg_nll, test_err))
            log.write(f"{epoch},{avg_nll:.6f},{test_err:.6f}\n")
            print(f"[{norm_mode} bz{batch_size}] epoch {epoch:2d} train_nll={avg_nll:.4f} test_err={test_err:.4f}")

    return net, history


if __name__ == "__main__":
    X_train_full, y_train_full, X_test, y_test = load_mnist()
    rng = np.random.default_rng(0)
    idx = rng.choice(X_train_full.shape[0], size=TRAIN_SUBSET, replace=False)
    X_train = X_train_full[idx]
    y_train_int = y_train_full[idx]
    D_train = one_hot(y_train_int)

    results = {}
    for batch_size in [128, 4]:
        for norm_mode in ["none", "batchnorm", "layernorm"]:
            key = (norm_mode, batch_size)
            _, hist = train(norm_mode, batch_size, X_train, D_train, y_train_int, X_test, y_test,
                             seed=1, log_path=os.path.join(LOG_DIR, f"run_{norm_mode}_bz{batch_size}.log"))
            results[key] = hist

    colors = {"none": "#C44E52", "batchnorm": "#55A868", "layernorm": "#4C72B0"}
    labels = {"none": "Baseline", "batchnorm": "BatchNorm", "layernorm": "LayerNorm"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, batch_size in enumerate([128, 4]):
        for norm_mode in ["none", "batchnorm", "layernorm"]:
            hist = results[(norm_mode, batch_size)]
            epochs = [h[0] for h in hist]
            nll = [h[1] for h in hist]
            err = [h[2] for h in hist]
            axes[0, col].plot(epochs, nll, label=labels[norm_mode], color=colors[norm_mode])
            axes[1, col].plot(epochs, err, label=labels[norm_mode], color=colors[norm_mode])
        axes[0, col].set_title(f"batch size {batch_size}")
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylabel("train NLL" if col == 0 else "")
        axes[1, col].set_ylabel("test error" if col == 0 else "")
        axes[1, col].set_xlabel("epoch")
        axes[0, col].legend()
    fig.suptitle(f"Permutation-invariant MNIST, {ARCH} ReLU MLP, {TRAIN_SUBSET}-example subset")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig6_batchsize_robustness.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)

    print("\nfinal test errors:")
    for batch_size in [128, 4]:
        for norm_mode in ["none", "batchnorm", "layernorm"]:
            print(f"  {labels[norm_mode]:10s} bz={batch_size:3d}: {results[(norm_mode, batch_size)][-1][2]:.4f}")
