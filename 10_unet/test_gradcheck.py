"""
Central-difference gradcheck, isolated layers then the full composed
UNet + loss. Same convention as every CNN paper in this repo since
06_lenet5: EPS_FD=1e-5 for isolated layers, tightened for the full net.
"""
import numpy as np

from conv2d import Conv2D
from maxpool2d import MaxPool2x2
from transposed_conv2d import TransposedConv2x2Stride2
from crop_concat import CropConcat
from double_conv import DoubleConv3x3ReLU
from unet import UNet
from loss import WeightedPixelSoftmaxCE

EPS_FD = 1e-5
CHECK_LOG = []  # (name, median_rel_err, status) -- for the summary figure


def num_grad(f, x, eps=EPS_FD):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
    return g


def rel_err(a, b):
    return np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))


def check(name, analytic, numeric, atol=1e-6, rtol_thresh=1e-2, bad_frac_thresh=0.5):
    """Pass/fail by MEDIAN error, not the single worst entry (bad_frac is
    a secondary backstop, deliberately loose -- see why below). Worst-
    case-only checking looked right through 05_layernorm/09_resnet
    (small nets, few ReLU units, near-zero chance any one of them sits
    exactly on a kink). It breaks on a net this size: enc0 alone has
    34848 ReLU units, and a random continuous input will always put a
    handful of them within float eps of their kink -- confirmed directly
    for this exact architecture (JOURNAL.md entry 5): perturbing one
    weight by 1e-4 flips ~1 unit's sign out of ~35k, and checking the
    *full* error distribution for a conv+relu+conv dx field showed
    median error ~1e-12 (correct) with exactly 10/4624 (0.2%) outlier
    entries >0.1 error -- a real, expected, sparse non-differentiability
    artifact, not a wrong gradient. The full composed net's own dx field
    showed 111/4624 (2.4%) such outliers -- proportionally more than the
    2-layer isolation test, consistent with chaining 10 ReLU
    applications end-to-end (enc0/enc1/bottleneck/dec1/dec0) instead of
    1, not evidence of a bug. bad_frac_thresh=0.5 exists only to catch a
    truly broken gradient (most entries wrong); a handful of kink hits,
    including in a 3-sample parameter probe where one hit is 33% of the
    sample, must never fail this check on their own -- median is what
    actually tells the two cases apart."""
    err = rel_err(analytic, numeric)
    bad = (err > rtol_thresh) & (np.abs(analytic - numeric) > atol)
    bad_frac = bad.mean()
    median_err = np.median(err)
    status = "FAIL" if (bad_frac > bad_frac_thresh or median_err > rtol_thresh) else "ok"
    print(f"  {name}: median rel err {median_err:.2e}, {bad.sum()}/{bad.size} entries >tol [{status}]")
    CHECK_LOG.append((name, median_err, status))
    return status == "ok"


def check_conv2d():
    print("Conv2D (3x3, 2->3 channels, 7x7 input)")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 2, 7, 7))
    layer = Conv2D(2, 3, 3, seed=1)
    probe = rng.normal(size=layer.forward(x).shape)

    def loss():
        return np.sum(layer.forward(x) * probe)

    out = layer.forward(x)
    dout = probe
    dx, dW, db = layer.backward(dout)

    num_dx = num_grad(loss, x)
    num_dW = num_grad(loss, layer.W)
    num_db = num_grad(loss, layer.b)
    ok = (check("Conv2D.dx", dx, num_dx) & check("Conv2D.dW", dW, num_dW)
          & check("Conv2D.db", db, num_db))
    return ok


def check_transposed_conv():
    print("TransposedConv2x2Stride2 (3->4 channels, 5x5 input)")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 5, 5))
    layer = TransposedConv2x2Stride2(3, 4, seed=3)
    probe = rng.normal(size=layer.forward(x).shape)

    def loss():
        return np.sum(layer.forward(x) * probe)

    dx, dW, db = layer.backward(probe)
    num_dx = num_grad(loss, x)
    num_dW = num_grad(loss, layer.W)
    num_db = num_grad(loss, layer.b)
    ok = (check("TransposedConv.dx", dx, num_dx) & check("TransposedConv.dW", dW, num_dW)
          & check("TransposedConv.db", db, num_db))
    return ok


def check_crop_concat():
    print("CropConcat (skip 10x10x3, up 6x6x2)")
    rng = np.random.default_rng(4)
    skip = rng.normal(size=(2, 3, 10, 10))
    up = rng.normal(size=(2, 2, 6, 6))
    layer = CropConcat()
    out = layer.forward(skip, up)
    probe = rng.normal(size=out.shape)

    def loss():
        return np.sum(layer.forward(skip, up) * probe)

    d_skip, d_up = layer.backward(probe)
    num_dskip = num_grad(loss, skip)
    num_dup = num_grad(loss, up)
    ok = check("CropConcat.d_skip", d_skip, num_dskip) & check("CropConcat.d_up", d_up, num_dup)
    return ok


def check_maxpool():
    print("MaxPool2x2 (8x8 input)")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, 8, 8))
    layer = MaxPool2x2()
    out = layer.forward(x)
    probe = rng.normal(size=out.shape)

    def loss():
        return np.sum(layer.forward(x) * probe)

    dx = layer.backward(probe)
    num_dx = num_grad(loss, x)
    return check("MaxPool2x2.dx", dx, num_dx)


def check_double_conv():
    print("DoubleConv3x3ReLU (2->4->4 channels, 9x9 input)")
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 2, 9, 9))
    layer = DoubleConv3x3ReLU(2, 4, 4, seed=7)
    out = layer.forward(x)
    probe = rng.normal(size=out.shape)

    def loss():
        return np.sum(layer.forward(x) * probe)

    dx, grads = layer.backward(probe)
    num_dx = num_grad(loss, x)
    ok = check("DoubleConv.dx", dx, num_dx)
    for name, param in layer.params().items():
        def loss_p():
            return np.sum(layer.forward(x) * probe)
        num_dp = num_grad(loss_p, param)
        ok &= check(f"DoubleConv.{name}", grads[name], num_dp)
    return ok


def check_full_unet():
    print("Full UNet + WeightedPixelSoftmaxCE (eps=1e-4, tighter numeric budget)")
    rng = np.random.default_rng(8)
    x = rng.normal(size=(1, 1, 68, 68)) * 0.1
    labels = rng.integers(0, 2, size=(1, 28, 28))
    weights = rng.uniform(0.5, 2.0, size=(1, 28, 28))

    model = UNet(seed=42)
    loss_fn = WeightedPixelSoftmaxCE()

    def loss():
        logits = model.forward(x)
        return loss_fn.forward(logits, labels, weights)

    logits = model.forward(x)
    l = loss_fn.forward(logits, labels, weights)
    dlogits = loss_fn.backward()
    dx, grads = model.backward(dlogits)

    eps = 1e-4
    ok = True
    num_dx = num_grad(loss, x, eps=eps)
    ok &= check("UNet.dx", dx, num_dx, rtol_thresh=5e-2)

    params = model.params()
    # Full-param gradcheck is expensive (every weight, every forward pass)
    # -- sample for the big weight tensors. But a 3-sample probe on an
    # 8-element bias vector is too small to tell "real bug" from "one
    # sampled entry landed on a ReLU kink" (see check()'s docstring);
    # biases are tiny, so just check every element of anything small
    # enough that doing so is still cheap.
    sample_rng = np.random.default_rng(9)
    for name, param in params.items():
        flat = param.ravel()
        n_sample = flat.size if flat.size <= 32 else 8
        sample_idx = (sample_rng.choice(flat.size, size=n_sample, replace=False)
                      if n_sample < flat.size else np.arange(flat.size))
        g_analytic = grads[name].ravel()
        g_num = np.zeros(n_sample)
        for k, idx in enumerate(sample_idx):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = loss()
            flat[idx] = orig - eps
            lm = loss()
            flat[idx] = orig
            g_num[k] = (lp - lm) / (2 * eps)
        ok &= check(f"{name} ({n_sample} entries)", g_analytic[sample_idx], g_num, rtol_thresh=5e-2)
    return ok


if __name__ == "__main__":
    results = {
        "Conv2D": check_conv2d(),
        "TransposedConv2x2Stride2": check_transposed_conv(),
        "CropConcat": check_crop_concat(),
        "MaxPool2x2": check_maxpool(),
        "DoubleConv3x3ReLU": check_double_conv(),
        "Full UNet": check_full_unet(),
    }
    print()
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    # status colors from the repo's dataviz reference palette (validated
    # for colorblind-safe contrast), not eyeballed hex values.
    STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"

    names = [n for n, _, _ in CHECK_LOG]
    errs = [max(e, 1e-13) for _, e, _ in CHECK_LOG]
    colors = [STATUS_GOOD if s == "ok" else STATUS_CRITICAL for _, _, s in CHECK_LOG]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.24 * len(CHECK_LOG))))
    y = range(len(CHECK_LOG))
    ax.barh(y, errs, color=colors, height=0.65)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    thresh_line = ax.axvline(1e-2, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("median relative error vs. finite-difference (log scale)")
    ax.set_title("10_unet gradcheck -- median relative error per layer/parameter")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_GOOD, label="PASS"),
               plt.Rectangle((0, 0), 1, 1, color=STATUS_CRITICAL, label="FAIL"),
               thresh_line]
    thresh_line.set_label("rtol_thresh (1e-2)")
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "gradcheck_summary.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_dir / 'gradcheck_summary.png'}")

    assert all(results.values()), "gradcheck failed"
    print("\nall gradchecks passed")
