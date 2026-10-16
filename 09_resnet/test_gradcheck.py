"""Isolated gradchecks for conv2d.py (stride=2 case, new here since
08_vggnet never exercised stride!=1), bn2d.py, and residual_block.py
(IdentityShortcut + BasicBlock) before composing the full ResNet-20.
dense.py is an unchanged duplicate already gradchecked in 07_alexnet/
08_vggnet, not re-checked here. bn_layer.py (BatchNorm1d) already
gradchecked in 03_batchnorm/test_gradcheck.py -- bn2d.py only reshapes
and delegates to it, checked here through the 2D wrapper instead."""
import numpy as np
from conv2d import Conv2D
from bn2d import BatchNorm2d
from residual_block import IdentityShortcut, BasicBlock

EPS_FD = 1e-5
REL_TOL = 1e-3
ABS_TOL = 1e-4
rng = np.random.default_rng(0)


def check(name, x, f, df_dx, n_checks=8):
    worst = 0.0
    flat_idx = rng.choice(x.size, size=min(n_checks, x.size), replace=False)
    for idx in flat_idx:
        multi = np.unravel_index(idx, x.shape)
        orig = x[multi]
        x[multi] = orig + EPS_FD
        lp = f()
        x[multi] = orig - EPS_FD
        lm = f()
        x[multi] = orig
        numeric = (lp - lm) / (2 * EPS_FD)
        analytic = df_dx[multi]
        e = abs(analytic - numeric) / max(1e-12, abs(analytic) + abs(numeric))
        worst = max(worst, e)
        assert e < REL_TOL or abs(analytic - numeric) < ABS_TOL, \
            f"{name}{multi}: analytic={analytic:.6f} numeric={numeric:.6f} rel_err={e:.2e}"
    return worst


if __name__ == "__main__":
    # Conv2D, stride=2 padding=1 (the downsampling convs 08_vggnet never exercised)
    x = rng.standard_normal((2, 4, 8, 8)) * 0.3
    conv = Conv2D(4, 8, 3, stride=2, padding=1, seed=1)
    y_target = rng.standard_normal(conv.forward(x).shape)

    def loss_conv():
        out = conv.forward(x)
        return 0.5 * np.sum((out - y_target) ** 2)

    out = conv.forward(x)
    dout = out - y_target
    dx, dW, db = conv.backward(dout)

    w_x = check("x", x, loss_conv, dx)
    w_W = check("W", conv.W, loss_conv, dW)
    w_b = check("b", conv.b, loss_conv, db)
    print(f"Conv2D (stride=2, padding=1, 3x3) gradcheck passed: worst rel err dx={w_x:.2e} dW={w_W:.2e} db={w_b:.2e}")

    # BatchNorm2d, spatial BN over conv feature maps (N,C,H,W)
    x2 = rng.standard_normal((4, 6, 5, 5)) * 0.5
    bn = BatchNorm2d(6)
    bn.gamma = rng.standard_normal(6) + 1.0
    bn.beta = rng.standard_normal(6)
    y_target2 = rng.standard_normal(x2.shape)

    def loss_bn():
        out, _ = bn.forward_train(x2)
        return 0.5 * np.sum((out - y_target2) ** 2)

    out2, cache2 = bn.forward_train(x2)
    dout2 = out2 - y_target2
    dx2, dgamma2, dbeta2 = bn.backward(dout2, cache2)

    w_bx = check("x", x2, loss_bn, dx2)
    w_bg = check("gamma", bn.gamma, loss_bn, dgamma2)
    w_bb = check("beta", bn.beta, loss_bn, dbeta2)
    print(f"BatchNorm2d gradcheck passed: worst rel err dx={w_bx:.2e} dgamma={w_bg:.2e} dbeta={w_bb:.2e}")

    # IdentityShortcut (option A): stride-2 subsample + zero-pad channels, parameter-free
    x3 = rng.standard_normal((2, 8, 8, 8)) * 0.3
    shortcut = IdentityShortcut(8, 16, stride=2)
    y_target3 = rng.standard_normal(shortcut.forward(x3).shape)

    def loss_shortcut():
        out = shortcut.forward(x3)
        return 0.5 * np.sum((out - y_target3) ** 2)

    out3 = shortcut.forward(x3)
    dout3 = out3 - y_target3
    dx3 = shortcut.backward(dout3)

    w_sx = check("x", x3, loss_shortcut, dx3)
    print(f"IdentityShortcut (stride=2, 8->16ch) gradcheck passed: worst rel err dx={w_sx:.2e}")

    # BasicBlock, stride=2 + channel change (crosses a stage boundary, exercises the shortcut)
    x4 = rng.standard_normal((2, 8, 8, 8)) * 0.3
    block = BasicBlock(8, 16, stride=2, seed=2)
    y_target4 = rng.standard_normal(block.forward(x4, train=True).shape)

    def loss_block():
        out = block.forward(x4, train=True)
        return 0.5 * np.sum((out - y_target4) ** 2)

    out4 = block.forward(x4, train=True)
    dout4 = out4 - y_target4
    dx4, grads4 = block.backward(dout4, prefix='block')

    w_4x = check("x", x4, loss_block, dx4)
    w_4w1 = check("conv1.W", block.conv1.W, loss_block, grads4['block.conv1.W'])
    w_4w2 = check("conv2.W", block.conv2.W, loss_block, grads4['block.conv2.W'])
    w_4g1 = check("bn1.gamma", block.bn1.gamma, loss_block, grads4['block.bn1.gamma'])
    w_4g2 = check("bn2.gamma", block.bn2.gamma, loss_block, grads4['block.bn2.gamma'])
    print(f"BasicBlock (stride=2, 8->16ch, option-A shortcut) gradcheck passed: "
          f"worst rel err dx={w_4x:.2e} dW1={w_4w1:.2e} dW2={w_4w2:.2e} "
          f"dgamma1={w_4g1:.2e} dgamma2={w_4g2:.2e}")
