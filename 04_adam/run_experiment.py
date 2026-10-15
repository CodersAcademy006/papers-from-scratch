"""
Section 6.2's deterministic multi-layer network comparison: Adam vs.
momentum-SGD vs. AdaGrad, 784-1000-1000-10 ReLU MLP, minibatch 128, full
MNIST, cross-entropy + L2 weight decay. See JOURNAL.md entry 3 for scope
(full 200-epoch run not reproduced -- see EPOCHS below).

Also runs entry 2/3's bias-correction ablation: Adam with vs. without the
bias-correction division, at beta2=0.999, on this same network -- testing
Section 6.4's claim directly rather than needing the paper's own VAE setup.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp_relu import ReLUMLP
from mnist_data import load_mnist, one_hot
from optimizer import Adam, MomentumSGD, AdaGrad

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ARCH = [784, 1000, 1000, 10]
BATCH = 128
# Paper's Fig. 2 runs to 200 "iterations over entire dataset" (epochs).
# Measured ~36s/epoch for this network on full 60k-example MNIST in pure
# numpy (smoke-tested before committing to a run length) -- 200 epochs x 3
# optimizers would be ~6 hours. Scoped to 8 here, enough to show the
# convergence-speed ordering the paper reports within a few minutes per
# optimizer, not a full run.
EPOCHS = 8


def make_net(seed):
    return ReLUMLP(ARCH, weight_std=0.01, l2=1e-4, seed=seed)


def train(opt_name, make_opt, X_train, D_train, epochs, seed, log_path):
    net = make_net(seed)
    opt = make_opt(net.params())
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    history = []  # (epoch_fraction, loss) once per epoch

    with open(log_path, "w") as log:
        log.write("epoch,avg_loss\n")
        for epoch in range(epochs):
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, BATCH):
                idx = perm[start:start + BATCH]
                Xb, Db = X_train[idx], D_train[idx]
                activations, pre_act = net.forward(Xb)
                loss = net.loss(activations[-1], Db)
                grads = net.backward(activations, pre_act, Db)
                opt.step(net.params(), grads)
                total_loss += loss / Xb.shape[0]
                n_batches += 1
            avg_loss = total_loss / n_batches
            history.append((epoch, avg_loss))
            log.write(f"{epoch},{avg_loss:.6f}\n")
            print(f"[{opt_name}] epoch {epoch:3d} avg_loss={avg_loss:.4f}")

    return net, history


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    D_train = one_hot(y_train)

    print("=== main comparison: Adam vs MomentumSGD vs AdaGrad ===")
    _, hist_adam = train("Adam", lambda p: Adam(p), X_train, D_train, EPOCHS, seed=1,
                          log_path=os.path.join(LOG_DIR, "run_adam.log"))
    _, hist_momentum = train("MomentumSGD", lambda p: MomentumSGD(p, lr=0.01, momentum=0.9),
                              X_train, D_train, EPOCHS, seed=1,
                              log_path=os.path.join(LOG_DIR, "run_momentum.log"))
    _, hist_adagrad = train("AdaGrad", lambda p: AdaGrad(p, alpha=0.01),
                             X_train, D_train, EPOCHS, seed=1,
                             log_path=os.path.join(LOG_DIR, "run_adagrad.log"))

    plt.figure(figsize=(7, 5))
    for hist, label, color in [(hist_adam, "Adam", "#4C72B0"),
                                 (hist_momentum, "MomentumSGD", "#C44E52"),
                                 (hist_adagrad, "AdaGrad", "#55A868")]:
        plt.plot([h[0] for h in hist], [h[1] for h in hist], label=label, color=color, marker="o", markersize=3)
    plt.xlabel("iterations over entire dataset (epoch)")
    plt.ylabel("training cost (avg cross-entropy + L2)")
    plt.yscale("log")
    plt.title("MNIST, 784-1000-1000-10 ReLU MLP, deterministic objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2b_optimizer_comparison.png"), dpi=150)
    plt.close()
    print("wrote figures/fig2b_optimizer_comparison.png")

    print("\n=== bias-correction ablation, beta2=0.999 (Section 6.4) ===")
    EPOCHS_ABLATION = 3
    _, hist_with_bc = train("Adam-with-bias-correction",
                             lambda p: Adam(p, beta2=0.999, bias_correction=True),
                             X_train, D_train, EPOCHS_ABLATION, seed=2,
                             log_path=os.path.join(LOG_DIR, "run_bias_correction_on.log"))
    _, hist_without_bc = train("Adam-without-bias-correction",
                                lambda p: Adam(p, beta2=0.999, bias_correction=False),
                                X_train, D_train, EPOCHS_ABLATION, seed=2,
                                log_path=os.path.join(LOG_DIR, "run_bias_correction_off.log"))

    plt.figure(figsize=(7, 5))
    plt.plot([h[0] for h in hist_with_bc], [h[1] for h in hist_with_bc],
              label="with bias correction", color="#4C72B0", marker="o", markersize=4)
    plt.plot([h[0] for h in hist_without_bc], [h[1] for h in hist_without_bc],
              label="without bias correction (= RMSProp+momentum)", color="#C44E52", marker="o", markersize=4)
    plt.xlabel("epoch")
    plt.ylabel("training cost (avg cross-entropy + L2)")
    plt.yscale("log")
    plt.title("Section 6.4 ablation: bias correction at beta2=0.999")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_bias_correction_ablation.png"), dpi=150)
    plt.close()
    print("wrote figures/fig4_bias_correction_ablation.png")

    print(f"\nfinal losses: Adam={hist_adam[-1][1]:.4f} MomentumSGD={hist_momentum[-1][1]:.4f} AdaGrad={hist_adagrad[-1][1]:.4f}")
    print(f"bias-correction ablation final losses: with_bc={hist_with_bc[-1][1]:.4f} without_bc={hist_without_bc[-1][1]:.4f}")
