"""
Full experiment run for the symmetry-detection task (paper's Fig. 1):
1. Sweep many random seeds to characterise the local-minima behaviour
   the paper itself warns about ("We have only encountered this
   undesirable behaviour in networks that have just enough connections
   to perform the task").
2. Re-train the best converging seed with per-sweep error logged, to
   plot a real learning curve.
3. Save everything to logs/ and figures/ as actual artifacts, not
   numbers transcribed by hand into the journal.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backprop import Network, make_symmetry_dataset, sigmoid

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = 100
EPOCHS = 1425
EPSILON = 0.1
ALPHA = 0.9


def train_logged(seed, epochs=EPOCHS, epsilon=EPSILON, alpha=ALPHA, n_hidden=2):
    X, D = make_symmetry_dataset()
    net = Network(n_in=X.shape[1], n_hidden=n_hidden, n_out=1, seed=seed)
    errors = []
    for sweep in range(1, epochs + 1):
        y1, y2 = net.forward(X)
        grads = net.backward(X, y1, y2, D)
        net.step(grads, epsilon, alpha)
        errors.append(net.error(y2, D))
    y1, y2 = net.forward(X)
    final_error = net.error(y2, D)
    accuracy = float(np.mean((y2 >= 0.5).astype(float) == D))
    return net, errors, final_error, accuracy


def seed_sweep():
    rows = []
    for seed in range(N_SEEDS):
        _, _, final_error, accuracy = train_logged(seed)
        rows.append((seed, final_error, accuracy))

    log_path = os.path.join(LOG_DIR, "seed_sweep.log")
    with open(log_path, "w") as f:
        f.write("seed,final_error,accuracy\n")
        for seed, err, acc in rows:
            f.write(f"{seed},{err:.6f},{acc:.4f}\n")

    converged = [r for r in rows if r[2] >= 0.999]
    stuck = [r for r in rows if r[2] < 0.999]
    print(f"seed sweep: {len(converged)}/{N_SEEDS} converged to accuracy>=0.999, "
          f"{len(stuck)}/{N_SEEDS} stuck below that")
    return rows, converged, stuck


def plot_local_minima_histogram(rows):
    errors = [r[1] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30, color="#4C72B0", edgecolor="black")
    plt.xlabel("final error E (eq. 3) after 1425 sweeps")
    plt.ylabel(f"count out of {len(rows)} random seeds")
    plt.title("Local minima in the minimal 2-hidden-unit symmetry network")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "local_minima_histogram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_loss_curve(errors, seed):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(errors) + 1), errors, color="#C44E52")
    plt.yscale("log")
    plt.xlabel("sweep")
    plt.ylabel("error E (eq. 3), log scale")
    plt.title(f"Training curve, seed={seed} (converged run)")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"loss_curve_seed{seed}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_hidden_weights(net, seed):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(net.W1.shape[1])
    width = 0.35
    ax.bar(x - width / 2, net.W1[0], width, label="hidden unit 1")
    ax.bar(x + width / 2, net.W1[1], width, label="hidden unit 2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("input unit (position in the 6-bit vector)")
    ax.set_ylabel("weight")
    ax.set_title(f"Learned hidden-unit weights, seed={seed}\n"
                 f"(paper: equal magnitude, opposite sign, ~1:2:4 outward ratio)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"hidden_weights_seed{seed}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


if __name__ == "__main__":
    rows, converged, stuck = seed_sweep()

    hist_path = plot_local_minima_histogram(rows)
    print("wrote", hist_path)

    best_seed = min(converged, key=lambda r: r[1])[0] if converged else rows[0][0]
    net, errors, final_error, accuracy = train_logged(best_seed)
    print(f"best converged seed={best_seed} final_error={final_error:.6f} accuracy={accuracy:.4f}")

    curve_path = plot_loss_curve(errors, best_seed)
    print("wrote", curve_path)
    weights_path = plot_hidden_weights(net, best_seed)
    print("wrote", weights_path)

    with open(os.path.join(LOG_DIR, f"training_curve_seed{best_seed}.log"), "w") as f:
        f.write("sweep,error\n")
        for i, e in enumerate(errors, start=1):
            if i % 25 == 0 or i == 1 or i == len(errors):
                f.write(f"{i},{e:.6f}\n")

    ratio_check = net.W1[0] / net.W1[0][0] if net.W1[0][0] != 0 else None
    print("W1 row 0:", np.round(net.W1[0], 3))
    print("W1 row 1:", np.round(net.W1[1], 3))
    print("row0 + row1 (should be ~0):", np.round(net.W1[0] + net.W1[1], 3))
