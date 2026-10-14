"""
Reproduces the paper's own stated overfitting scenario ("when a large
feedforward neural network is trained on a small training set, it
typically performs poorly on held-out test data") rather than its
full-scale 60k/3000-epoch MNIST run (see JOURNAL.md entry 3 for why).

Two runs, identical in every respect except dropout itself:
  (a) no dropout  (hidden_keep=1.0, input_keep=1.0)
  (b) dropout     (hidden_keep=0.5, input_keep=0.8, matching the paper)
Both use the same architecture, same max-norm constraint (l=15, the
paper's own MNIST value), and the same Appendix A.1 scheduled-momentum
optimizer -- so any difference in the train/test gap is attributable to
dropout alone, not a confound from also changing the optimizer.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp import DropoutMLP
from mnist_data import load_mnist, one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_SUBSET = 2000
ARCH = [784, 256, 128, 10]
EPOCHS = 300
BATCH = 100
MAXNORM_SQ = 15.0  # paper's own MNIST value (Appendix A.1: "l = 15")

# Appendix A.1 schedule. EPSILON_0 is NOT the paper's literal 10.0 -- see
# JOURNAL.md entry 6: that value caused immediate, irrecoverable sigmoid
# saturation (hidden layer 2 activations collapse to ~0 after the very
# first update) at this smaller architecture/dataset scale, even after
# fixing the momentum-direction and gradient-averaging bugs. 10.0 was
# presumably calibrated for the paper's own 784-800-800-10/60k-example
# setup; 1.0 was found empirically (by sweeping 10.0/1.0/0.5/0.2 and
# checking whether the hidden layer stayed alive) to train stably here.
EPSILON_0 = 1.0
F_DECAY = 0.998
P_INITIAL = 0.5
P_FINAL = 0.99
T_MOMENTUM = 500


def lr_and_momentum(epoch):
    epsilon = EPSILON_0 * (F_DECAY ** epoch)
    if epoch < T_MOMENTUM:
        # Paper's prose (not the possibly mis-OCR'd formula fraction order --
        # see JOURNAL.md entry 6): momentum starts at p_i=0.5 and rises
        # linearly to p_f=0.99 over the first T epochs, continuous at t=T.
        p = P_INITIAL + (epoch / T_MOMENTUM) * (P_FINAL - P_INITIAL)
    else:
        p = P_FINAL
    return epsilon, p


def error_rate(net, X, y_int):
    preds = net.predict(X)
    return float(np.mean(np.argmax(preds, axis=1) != y_int))


def train_run(X_train, D_train, y_train_int, X_test, y_test_int,
              hidden_keep, input_keep, seed, label):
    net = DropoutMLP(ARCH, hidden_keep=hidden_keep, input_keep=input_keep,
                      maxnorm_sq=MAXNORM_SQ, seed=seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    history = []  # (epoch, train_error, test_error)

    for epoch in range(EPOCHS):
        epsilon, p = lr_and_momentum(epoch)
        perm = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            Xb, Db = X_train[idx], D_train[idx]
            activations, masks = net.forward_train(Xb, rng)
            grads_W, grads_b = net.backward(activations, masks, Db)
            # backward() sums over the batch (matching eq (3)'s sum_c convention,
            # kept that way so test_gradcheck.py's already-verified check stays
            # valid); Appendix A.1's optimizer formula explicitly wants the
            # AVERAGE gradient over the minibatch ("<grad_w L>_t"), so divide
            # here, at the point where this paper's convention actually applies.
            batch_n = Xb.shape[0]
            grads_W = [g / batch_n for g in grads_W]
            grads_b = [g / batch_n for g in grads_b]
            net.step(grads_W, grads_b, epsilon, p)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            tr_err = error_rate(net, X_train, y_train_int)
            te_err = error_rate(net, X_test, y_test_int)
            history.append((epoch, tr_err, te_err))
            print(f"[{label}] epoch {epoch:4d} eps={epsilon:.4f} p={p:.3f} "
                  f"train_err={tr_err:.4f} test_err={te_err:.4f}")

    return net, history


if __name__ == "__main__":
    X_train_full, y_train_full, X_test, y_test = load_mnist()

    rng = np.random.default_rng(0)
    idx = rng.choice(X_train_full.shape[0], size=TRAIN_SUBSET, replace=False)
    X_train = X_train_full[idx]
    y_train_int = y_train_full[idx]
    D_train = one_hot(y_train_int)

    print(f"training subset: {X_train.shape}, test set: {X_test.shape}")

    net_nodrop, hist_nodrop = train_run(
        X_train, D_train, y_train_int, X_test, y_test,
        hidden_keep=1.0, input_keep=1.0, seed=1, label="no-dropout")

    net_drop, hist_drop = train_run(
        X_train, D_train, y_train_int, X_test, y_test,
        hidden_keep=0.5, input_keep=0.8, seed=1, label="dropout")

    with open(os.path.join(LOG_DIR, "run_nodropout.log"), "w") as f:
        f.write("epoch,train_error,test_error\n")
        for e, tr, te in hist_nodrop:
            f.write(f"{e},{tr:.6f},{te:.6f}\n")
    with open(os.path.join(LOG_DIR, "run_dropout.log"), "w") as f:
        f.write("epoch,train_error,test_error\n")
        for e, tr, te in hist_drop:
            f.write(f"{e},{tr:.6f},{te:.6f}\n")

    final_nodrop = hist_nodrop[-1]
    final_drop = hist_drop[-1]
    print(f"FINAL no-dropout: train_err={final_nodrop[1]:.4f} test_err={final_nodrop[2]:.4f} "
          f"gap={final_nodrop[2]-final_nodrop[1]:.4f}")
    print(f"FINAL dropout:    train_err={final_drop[1]:.4f} test_err={final_drop[2]:.4f} "
          f"gap={final_drop[2]-final_drop[1]:.4f}")

    # figure: train/test error curves, both runs
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, hist, title in [(axes[0], hist_nodrop, "No dropout"),
                              (axes[1], hist_drop, "Dropout (50% hidden, 20% input)")]:
        epochs = [h[0] for h in hist]
        tr = [h[1] for h in hist]
        te = [h[2] for h in hist]
        ax.plot(epochs, tr, label="train error", color="#4C72B0")
        ax.plot(epochs, te, label="test error", color="#C44E52")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("classification error rate")
    fig.suptitle(f"MNIST, {TRAIN_SUBSET}-example training subset, {ARCH} network")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "overfitting_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("wrote", out)

    # figure: generalization gap over time, both runs on one axis
    plt.figure(figsize=(6, 4))
    gap_nodrop = [te - tr for _, tr, te in hist_nodrop]
    gap_drop = [te - tr for _, tr, te in hist_drop]
    plt.plot([h[0] for h in hist_nodrop], gap_nodrop, label="no dropout", color="#4C72B0")
    plt.plot([h[0] for h in hist_drop], gap_drop, label="dropout", color="#C44E52")
    plt.xlabel("epoch")
    plt.ylabel("test error - train error (generalization gap)")
    plt.title("Generalization gap: dropout vs. no dropout")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "generalization_gap.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print("wrote", out2)
