"""
Reproduces Section 4.1 of the paper: 784-100-100-100-10 sigmoid MLP, BN vs.
no-BN, 50000 steps, batch size 60, binarized MNIST ("28x28 binary image").

Three things come out of this, matching JOURNAL.md entry 3's scope:
  1. Fig. 1(a): test accuracy vs. training step, same architecture/LR, with
     vs. without BN.
  2. Fig. 1(b,c): {15,50,85}th-percentile evolution of the last hidden
     sigmoid's input distribution over training, for a fixed unit.
  3. An extra ablation (not in the paper's MNIST section, testing its
     Section 3.3 claim directly): both networks at a deliberately large
     learning rate, to see which one survives.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp import BNMLP
from mnist_data import load_mnist, one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ARCH = [784, 100, 100, 100, 10]
STEPS = 50000
BATCH = 60
EVAL_EVERY = 500
PROBE_UNIT = 0  # which of the 100 last-hidden-layer units to chart percentiles for
PROBE_BATCH = 256  # fixed snapshot size for the percentile-evolution probe


def binarize(X, threshold=0.5):
    """Paper: '28x28 binary image as input'. MNIST pixels are loaded as
    [0,1] floats (mnist_data.py); threshold to {0,1} to match."""
    return (X > threshold).astype(np.float64)


def accuracy(net, X, y_int):
    preds = net.forward_eval(X)
    return float(np.mean(np.argmax(preds, axis=1) == y_int))


def run(use_bn, lr, X_train, D_train, y_train_int, X_test, y_test_int,
        probe_X, seed, label, log_path):
    net = BNMLP(ARCH, use_bn=use_bn, seed=seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]

    history = []       # (step, test_acc)
    percentiles = []    # (step, p15, p50, p85)
    diverged_at = None

    with open(log_path, "w") as log:
        log.write("step,test_acc,p15,p50,p85\n")
        for step in range(STEPS):
            idx = rng.integers(0, n, size=BATCH)  # paper says "steps" not epochs; sample with replacement
            Xb, Db = X_train[idx], D_train[idx]

            activations, bn_caches = net.forward_train(Xb)
            if not np.all(np.isfinite(activations[-1])):
                diverged_at = step
                print(f"[{label}] DIVERGED at step {step} (non-finite output)")
                break
            grads_W, grads_b, grads_bn = net.backward(activations, bn_caches, Db)
            net.step(grads_W, grads_b, grads_bn, lr)

            if step % EVAL_EVERY == 0 or step == STEPS - 1:
                acc = accuracy(net, X_test, y_test_int)
                probe_vals = net.probe_last_hidden_sigmoid_input(probe_X)[:, PROBE_UNIT]
                p15, p50, p85 = np.percentile(probe_vals, [15, 50, 85])
                history.append((step, acc))
                percentiles.append((step, p15, p50, p85))
                log.write(f"{step},{acc:.6f},{p15:.6f},{p50:.6f},{p85:.6f}\n")
                if step % 5000 == 0:
                    print(f"[{label}] step {step:6d} test_acc={acc:.4f}")

    return net, history, percentiles, diverged_at


if __name__ == "__main__":
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()
    X_train = binarize(X_train_full)
    X_test = binarize(X_test_full)
    D_train = one_hot(y_train_full)
    y_train_int, y_test_int = y_train_full, y_test_full

    rng = np.random.default_rng(0)
    probe_idx = rng.choice(X_test.shape[0], size=PROBE_BATCH, replace=False)
    probe_X = X_test[probe_idx]

    # Paper doesn't state a learning rate for this toy experiment (only for
    # the much bigger Inception/ImageNet one, out of scope -- entry 3).
    # Swept {0.01, 0.05, 0.1, 0.3, 1.0} at 2000 steps first (journal entry 6):
    # 0.01 is where BOTH networks train without collapsing (fair same-LR
    # comparison, matching the spirit of Fig. 1a); 0.3 is where no-BN
    # collapses to chance accuracy but BN still reaches ~86% -- used below
    # as the Section 3.3 stress test, not picked to avoid an inconvenient
    # result, picked BECAUSE it's the cleanest demonstration of that claim.
    LR = 0.01
    LR_STRESS = 0.3

    print("=== main comparison, lr =", LR, "===")
    net_nobn, hist_nobn, pct_nobn, _ = run(
        False, LR, X_train, D_train, y_train_int, X_test, y_test_int,
        probe_X, seed=1, label="no-BN", log_path=os.path.join(LOG_DIR, "run_nobn.log"))
    net_bn, hist_bn, pct_bn, _ = run(
        True, LR, X_train, D_train, y_train_int, X_test, y_test_int,
        probe_X, seed=1, label="BN", log_path=os.path.join(LOG_DIR, "run_bn.log"))

    final_nobn = hist_nobn[-1][1]
    final_bn = hist_bn[-1][1]
    print(f"FINAL test accuracy: no-BN={final_nobn:.4f} BN={final_bn:.4f}")

    # Fig 1(a): test accuracy vs step
    plt.figure(figsize=(7, 5))
    plt.plot([h[0] for h in hist_nobn], [h[1] for h in hist_nobn], label="without BN", color="#C44E52")
    plt.plot([h[0] for h in hist_bn], [h[1] for h in hist_bn], label="with BN", color="#4C72B0")
    plt.xlabel("training step")
    plt.ylabel("test accuracy")
    plt.title(f"MNIST, 784-100-100-100-10 sigmoid MLP, lr={LR}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1a_test_accuracy.png"), dpi=150)
    plt.close()

    # Fig 1(b,c): percentile evolution of last-hidden sigmoid input, unit PROBE_UNIT
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, pct, title in [(axes[0], pct_nobn, "Without BN"), (axes[1], pct_bn, "With BN")]:
        steps = [p[0] for p in pct]
        p15 = [p[1] for p in pct]
        p50 = [p[2] for p in pct]
        p85 = [p[3] for p in pct]
        ax.plot(steps, p50, color="#333333", label="50th pct")
        ax.fill_between(steps, p15, p85, alpha=0.3, color="#4C72B0", label="15th-85th pct")
        ax.set_title(title)
        ax.set_xlabel("training step")
    axes[0].set_ylabel(f"input to last hidden sigmoid, unit {PROBE_UNIT}")
    axes[0].legend()
    fig.suptitle("Fig. 1(b,c) reproduction: sigmoid input distribution over training")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1bc_activation_percentiles.png"), dpi=150)
    plt.close()

    print("wrote figures/fig1a_test_accuracy.png, figures/fig1bc_activation_percentiles.png")

    # Section 3.3 stress test: same architecture, deliberately large LR.
    # "Batch Normalization allows us to use much higher learning rates...
    # without the risk of divergence." Run both networks at LR_STRESS and
    # compare collapse/survival, not final accuracy curves.
    print("\n=== LR stress test, lr =", LR_STRESS, "===")
    _, hist_nobn_stress, _, div_nobn = run(
        False, LR_STRESS, X_train, D_train, y_train_int, X_test, y_test_int,
        probe_X, seed=1, label="no-BN-stress", log_path=os.path.join(LOG_DIR, "run_nobn_stress.log"))
    _, hist_bn_stress, _, div_bn = run(
        True, LR_STRESS, X_train, D_train, y_train_int, X_test, y_test_int,
        probe_X, seed=1, label="BN-stress", log_path=os.path.join(LOG_DIR, "run_bn_stress.log"))

    print(f"no-BN  @ lr={LR_STRESS}: final test_acc={hist_nobn_stress[-1][1]:.4f} diverged={div_nobn}")
    print(f"BN     @ lr={LR_STRESS}: final test_acc={hist_bn_stress[-1][1]:.4f} diverged={div_bn}")

    plt.figure(figsize=(7, 5))
    plt.plot([h[0] for h in hist_nobn_stress], [h[1] for h in hist_nobn_stress],
              label="without BN", color="#C44E52")
    plt.plot([h[0] for h in hist_bn_stress], [h[1] for h in hist_bn_stress],
              label="with BN", color="#4C72B0")
    plt.xlabel("training step")
    plt.ylabel("test accuracy")
    plt.title(f"Section 3.3 stress test: same net, lr={LR_STRESS} (vs. {LR} in the main comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lr_stress_test.png"), dpi=150)
    plt.close()
    print("wrote figures/lr_stress_test.png")
