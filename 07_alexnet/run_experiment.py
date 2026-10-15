"""
Train AlexNetCIFAR on the real CIFAR-10 training set, SGD+momentum+
weight-decay (Section 5, as specified -- see optimizer.py), with a
patience-based LR step standing in for the paper's own manual "divide
by 10 when validation error stops improving" rule (entry 2).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alexnet_cifar import AlexNetCIFAR
from optimizer import SGDMomentum
from cifar_data import load_cifar10, one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 128
PATIENCE = 3
LR_DECAY_FACTOR = 10.0


def test_error(net, X_test, y_test, rng, batch_size=500):
    wrong = 0
    for start in range(0, X_test.shape[0], batch_size):
        xb = X_test[start:start + batch_size]
        logits = net.forward(xb, train=False, rng=rng)
        preds = np.argmax(logits, axis=1)
        wrong += np.sum(preds != y_test[start:start + batch_size])
    return wrong / X_test.shape[0]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10()
    D_train = one_hot(y_train)
    n = X_train.shape[0]
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    net = AlexNetCIFAR(seed=0)
    opt = SGDMomentum(net.params(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    rng = np.random.default_rng(0)

    best_err = 1.0
    bad_epochs = 0
    history = []
    log_path = os.path.join(LOG_DIR, "run_alexnet.log")
    with open(log_path, "w") as log:
        log.write("epoch,train_nll,test_err,lr,epoch_seconds\n")
        for epoch in range(EPOCHS):
            t0 = time.time()
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                xb, db = X_train[idx], D_train[idx]
                logits = net.forward(xb, train=True, rng=rng)
                loss, dlogits, _ = net.softmax_ce(logits, db)
                _, grads = net.backward(dlogits)
                opt.step(net.params(), grads)
                total_loss += loss
                n_batches += 1
            avg_loss = total_loss / n_batches
            err = test_error(net, X_test, y_test, rng)
            elapsed = time.time() - t0

            if err < best_err - 1e-4:
                best_err = err
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    opt.lr /= LR_DECAY_FACTOR
                    bad_epochs = 0
                    print(f"  -> validation error plateaued, lr divided by {LR_DECAY_FACTOR} to {opt.lr}")

            history.append((epoch, avg_loss, err, opt.lr, elapsed))
            log.write(f"{epoch},{avg_loss:.6f},{err:.6f},{opt.lr:.6f},{elapsed:.1f}\n")
            log.flush()
            print(f"epoch {epoch:2d}  train_nll={avg_loss:.4f}  test_err={err:.4f} ({err*100:.2f}%)  lr={opt.lr:.5f}  {elapsed:.1f}s")

    epochs = [h[0] for h in history]
    nll = [h[1] for h in history]
    err_pct = [h[2] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(epochs, nll, color="#4C72B0")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train NLL")
    ax1.set_yscale("log")
    ax1.set_title("training loss")

    ax2.plot(epochs, err_pct, color="#C44E52", marker="o", markersize=3)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test error (%)")
    ax2.set_title("CIFAR-10 test error (rescaled AlexNet)")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "training_curve.png")
    plt.savefig(out, dpi=150)
    print("wrote", out)
    print(f"\nfinal test error: {history[-1][2]*100:.2f}%  best: {best_err*100:.2f}%")
