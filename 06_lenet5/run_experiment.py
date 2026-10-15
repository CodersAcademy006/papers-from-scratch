"""
Train LeNet5 on the full 60k MNIST training set for 20 epochs (matching
the paper's own pass count, Section III.B), compare final test error
against the paper's headline 0.95% (no distortion, Fig. 5).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lenet5 import LeNet5
from optimizer import Adam
from mnist_data import load_mnist, one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 128


def test_error(net, X_test, y_test, batch_size=500):
    wrong = 0
    for start in range(0, X_test.shape[0], batch_size):
        xb = X_test[start:start + batch_size]
        logits = net.forward(xb)
        preds = np.argmax(logits, axis=1)
        wrong += np.sum(preds != y_test[start:start + batch_size])
    return wrong / X_test.shape[0]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    D_train = one_hot(y_train)
    n = X_train.shape[0]
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    net = LeNet5(seed=0)
    opt = Adam(net.params(), alpha=0.001)
    rng = np.random.default_rng(0)

    history = []
    log_path = os.path.join(LOG_DIR, "run_lenet5.log")
    with open(log_path, "w") as log:
        log.write("epoch,train_nll,test_err,epoch_seconds\n")
        for epoch in range(EPOCHS):
            t0 = time.time()
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                xb, db = X_train[idx], D_train[idx]
                logits = net.forward(xb)
                loss, dlogits, _ = net.softmax_ce(logits, db)
                _, grads = net.backward(dlogits)
                opt.step(net.params(), grads)
                total_loss += loss
                n_batches += 1
            avg_loss = total_loss / n_batches
            err = test_error(net, X_test, y_test)
            elapsed = time.time() - t0
            history.append((epoch, avg_loss, err, elapsed))
            log.write(f"{epoch},{avg_loss:.6f},{err:.6f},{elapsed:.1f}\n")
            log.flush()
            print(f"epoch {epoch:2d}  train_nll={avg_loss:.4f}  test_err={err:.4f} ({err*100:.2f}%)  {elapsed:.1f}s")

    epochs = [h[0] for h in history]
    nll = [h[1] for h in history]
    err = [h[2] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(epochs, nll, color="#4C72B0")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train NLL")
    ax1.set_yscale("log")
    ax1.set_title("training loss")

    ax2.plot(epochs, err, color="#C44E52", marker="o", markersize=3)
    ax2.axhline(0.95, color="gray", linestyle="--", label="paper: 0.95% (no distortion)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test error (%)")
    ax2.set_title("test error vs. paper's Fig. 5")
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_training_curve.png")
    plt.savefig(out, dpi=150)
    print("wrote", out)
    print(f"\nfinal test error: {history[-1][2]*100:.2f}% (paper, no distortion: 0.95%)")
