"""
HISTORICAL. This diagnostic is what caused bn_layer.py's running-average
bookkeeping to change (see JOURNAL.md entry 7) -- kept here as the record
of why, not as a live alternative to the real implementation.

run_experiment.py's main comparison at lr=0.01 first came back with no-BN
slightly BEATING BN (0.9662 vs 0.9651) and training faster early on -- the
opposite of Fig. 1(a)'s shape. Hypothesis tested here: the "literal running
average" choice for inference stats (entry 2's first implementation)
weighs every training mini-batch equally, including the very noisy early
ones (before the network has learned anything), dragging the FINAL
running_mean/running_var away from where the network's activations
actually are by the end of training -- hurting forward_eval() specifically,
even though training-time (batch-stats-based) behavior was fine.

EMABatchNorm1d below was a throwaway local subclass used to test an
exponential moving average against the then-current plain-average
BatchNorm1d, without touching the main implementation until the result
justified it. Result: EMA reached 0.9757 final test accuracy vs. the plain
average's 0.9651, and converged much faster (0.9588 vs 0.8515 by step
5000) -- confirming the hypothesis. bn_layer.py now implements this
exponential-average logic directly, so this subclass is redundant with the
real class today; it's kept only as the evidence trail.
"""
import os
import sys
import numpy as np

PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT)  # bn_layer.py/mlp.py/etc. live one level up, in 03_batchnorm/

from bn_layer import BatchNorm1d
from mlp import BNMLP
from mnist_data import load_mnist, one_hot
from run_experiment import binarize, accuracy, ARCH, BATCH, STEPS, EVAL_EVERY

HERE = PARENT  # logs/ it writes to belong with the rest of the experiment's logs


class EMABatchNorm1d(BatchNorm1d):
    """Same forward/backward math, different running-stat bookkeeping."""

    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        super().__init__(n_features, eps)
        self.momentum = momentum

    def forward_train(self, x):
        y, cache = self._batch_normalize(x)
        _, _, mu_B, sigma_B_sq, _, m = cache
        unbiased_var = sigma_B_sq * m / (m - 1) if m > 1 else sigma_B_sq
        if self.n_batches_seen == 0:
            self.running_mean = mu_B
            self.running_var = unbiased_var
        else:
            mom = self.momentum
            self.running_mean = mom * self.running_mean + (1 - mom) * mu_B
            self.running_var = mom * self.running_var + (1 - mom) * unbiased_var
        self.n_batches_seen += 1
        return y, cache


def make_ema_net(seed):
    net = BNMLP(ARCH, use_bn=True, seed=seed)
    for l in range(net.n_layers):
        if net.bn[l] is not None:
            n_features = net.bn[l].gamma.shape[0]
            ema = EMABatchNorm1d(n_features)
            ema.gamma = net.bn[l].gamma
            ema.beta = net.bn[l].beta
            net.bn[l] = ema
    return net


if __name__ == "__main__":
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()
    X_train = binarize(X_train_full)
    X_test = binarize(X_test_full)
    D_train = one_hot(y_train_full)
    y_train_int, y_test_int = y_train_full, y_test_full

    LR = 0.01
    rng = np.random.default_rng(1)
    n = X_train.shape[0]

    net = make_ema_net(seed=1)
    history = []
    for step in range(STEPS):
        idx = rng.integers(0, n, size=BATCH)
        Xb, Db = X_train[idx], D_train[idx]
        activations, bn_caches = net.forward_train(Xb)
        grads_W, grads_b, grads_bn = net.backward(activations, bn_caches, Db)
        net.step(grads_W, grads_b, grads_bn, LR)
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            acc = accuracy(net, X_test, y_test_int)
            history.append((step, acc))
            if step % 5000 == 0:
                print(f"[BN-EMA] step {step:6d} test_acc={acc:.4f}")

    with open(os.path.join(HERE, "logs", "run_bn_ema.log"), "w") as f:
        f.write("step,test_acc\n")
        for s, a in history:
            f.write(f"{s},{a:.6f}\n")

    print(f"FINAL BN-EMA test accuracy: {history[-1][1]:.4f}")
    print("(compare against logs/run_bn.log's plain-running-average final accuracy, and logs/run_nobn.log)")
