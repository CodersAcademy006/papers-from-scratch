# Journal — 03_batchnorm

Paper: "Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift" (Ioffe & Szegedy, arXiv:1502.03167, 2015).

## Entry 1 — Sourcing

On arXiv, full text pulled directly through `mcp__alphaxiv__get_paper_content`
(fullText=true) in one call, equations and all. No PDF-mirror fight this time
(contrast with 01_backprop_autograd, which predates arXiv and took two mirror
attempts; and 02_dropout, which needed the same tool but was otherwise easy).
Worth noting only because it won't always be this easy — 01 and 02 both took
real effort just to get a working text source.

## Entry 2 — What the paper actually specifies, by quote

Quoting the paper directly for every design decision below, not paraphrasing,
so a reader can check the implementation against the paper's own words
instead of my summary of them.

**The core transform (Algorithm 1).** Per-feature, not joint/whitened:

> "instead of whitening the features in layer inputs and outputs jointly, we
> will normalize each scalar feature independently, by making it have the
> mean of zero and the variance of 1."

Exact steps, batch B of size m, one scalar feature at a time:

```
mu_B    = (1/m) * sum_i x_i
sigma_B^2 = (1/m) * sum_i (x_i - mu_B)^2
x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + eps)
y_i     = gamma * x_hat_i + beta  =  BN_{gamma,beta}(x_i)
```

**Why gamma/beta exist at all** (not just normalize and stop) — restoring
representational power:

> "normalizing the inputs of a sigmoid would constrain them to the linear
> regime of the nonlinearity. To address this, we make sure that the
> transformation inserted in the network can represent the identity
> transform... by setting gamma^(k) = sqrt(Var[x^(k)]) and beta^(k) =
> E[x^(k)], we could recover the original activations, if that were the
> optimal thing to do."

**Where exactly BN goes relative to the nonlinearity** — before it, on
`Wu+b`, not on the layer's raw input `u`:

> "We add the BN transform immediately before the nonlinearity, by
> normalizing x = Wu + b... since u is likely the output of another
> nonlinearity, the shape of its distribution is likely to change during
> training, and constraining its first and second moments would not
> eliminate the covariate shift. In contrast, Wu + b is more likely to have
> a symmetric, non-sparse distribution."

And a detail that's easy to miss and silently get wrong: the bias `b` becomes
redundant once BN subtracts the batch mean, because any constant shift from
`b` is exactly cancelled by `mu_B` subtraction:

> "since we normalize Wu+b, the bias b can be ignored since its effect will
> be canceled by the subsequent mean subtraction (the role of the bias is
> subsumed by beta in Alg. 1)."

So the implementation below has NO bias term on the pre-BN linear layer —
only `W`, with `beta` doing the job a bias normally would. This is a
one-line omission that is very easy to get wrong by habit (every linear
layer "should" have a bias) and the paper calls it out explicitly for
exactly that reason.

**Train vs. inference — the part most tutorials gloss over** (Algorithm 2).
At train time, mean/var come from the mini-batch and are part of the
computation graph. At inference, they're frozen population statistics:

> "The normalization of activations that depends on the mini-batch allows
> efficient training, but is neither necessary nor desirable during
> inference; we want the output to depend only on the input,
> deterministically... we use the normalization x_hat = (x - E[x]) /
> sqrt(Var[x] + eps) using the population, rather than mini-batch,
> statistics... We use the unbiased variance estimate Var[x] = m/(m-1) *
> E_B[sigma_B^2], where the expectation is over training mini-batches."

The paper is deliberately vague on *how* to accumulate `E_B[mu_B]` and
`E_B[sigma_B^2]` across many mini-batches ("process multiple training
mini-batches... and average over them") — it does not specify a running
exponential average, which is what basically every modern framework (and
what BatchNorm became known for) actually does. Implemented here as a
genuinely literal running (arithmetic, not exponential) average over every
mini-batch seen during training, specifically because that's the literal
reading of "average over them" — and flagged here as a deliberate deviation
from the EMA convention that came later, same spirit as 02_dropout's choice
to implement the paper's actual historical "halve the weights" mean-network
instead of the inverted-dropout convention that superseded it.

**Backward pass — given to us directly, not left as an exercise.** The paper
hands over the exact chain-rule decomposition (Section 3, "before
simplification"):

```
dL/dx_hat_i = dL/dy_i * gamma
dL/dsigma_B^2 = sum_i dL/dx_hat_i * (x_i - mu_B) * (-1/2) * (sigma_B^2+eps)^(-3/2)
dL/dmu_B = (sum_i dL/dx_hat_i * -1/sqrt(sigma_B^2+eps)) + dL/dsigma_B^2 * sum_i(-2(x_i-mu_B))/m
dL/dx_i = dL/dx_hat_i * 1/sqrt(sigma_B^2+eps) + dL/dsigma_B^2 * 2(x_i-mu_B)/m + dL/dmu_B * 1/m
dL/dgamma = sum_i dL/dy_i * x_hat_i
dL/dbeta  = sum_i dL/dy_i
```

This is implemented literally, term by term, in `bn_layer.py`, rather than
using the single-pass algebraically-simplified form that's common in later
blog posts/frameworks — closer to the paper's own derivation, and much
easier to cross-check line-by-line against the quote above.

**The warning the paper gives about getting this wrong** — this became the
most useful part of the whole paper for this exercise, see Entry 4:

> "consider a layer with the input u that adds the learned bias b, and
> normalizes the result by subtracting the mean of the activation computed
> over the training data... If a gradient descent step ignores the
> dependence of E[x] on b, then it will update b <- b + db... Thus, the
> combination of the update to b and subsequent change in normalization led
> to no change in the output of the layer nor, consequently, the loss. As
> the training continues, b will grow indefinitely while the loss remains
> fixed... We have observed this empirically in initial experiments, where
> the model blows up when the normalization parameters are computed outside
> the gradient descent step."

The paper is describing exactly the bug of computing gradients as if
`mu_B`/`sigma_B^2` were constants with respect to `x`, instead of
differentiating through them. Section 4.

## Entry 3 — Scope decision

Unlike 01 and 02, the paper's own validating experiment for the mechanism
itself (not the full ImageNet/Inception result) is small enough to run
exactly as described, on a laptop, in pure numpy. Section 4.1:

> "we considered the problem of predicting the digit class on the MNIST
> dataset... a very simple network, with a 28x28 binary image as input, and
> 3 fully-connected hidden layers with 100 activations each. Each hidden
> layer computes y = g(Wu+b) with sigmoid nonlinearity... followed by a
> fully-connected layer with 10 activations (one per class) and
> cross-entropy loss. We trained the network for 50000 steps, with 60
> examples per mini-batch."

This is being reproduced exactly: 784-100-100-100-10, sigmoid hidden units,
cross-entropy + softmax output, 50000 SGD steps, batch size 60, BN inserted
before every hidden sigmoid (per Entry 2's "immediately before the
nonlinearity" quote). The paper does not state a learning rate for this
specific toy experiment (it only gives learning rates for the much larger
Inception/ImageNet experiment in 4.2, which is explicitly a different,
much bigger network with momentum SGD — not reproduced here, out of scope,
same reasoning as skipping ImageNet in every prior paper in this repo: it
needs a distributed cluster, not a laptop). A fixed learning rate was
chosen empirically for this toy network (documented with the actual search
in Entry 5) since the paper leaves it unspecified for this exact setup.

Two things are reproduced from Section 4, beyond the headline "BN trains
faster":
1. Fig. 1(a): test accuracy vs. training step, with vs. without BN, same
   architecture, same learning rate.
2. Fig. 1(b,c): the {15,50,85}th-percentile evolution of a sigmoid's input
   distribution over training — the actual internal-covariate-shift
   evidence, not just the end accuracy number.
3. An extra ablation not in the paper's MNIST section but directly testing
   its Section 3.3 claim ("Batch Normalization allows us to use much higher
   learning rates... without the risk of divergence"): run both networks at
   a deliberately large learning rate and check which one survives.

Out of scope, same as every other paper here: the Inception/ImageNet
experiment (13.6M-parameter conv net, distributed training, ensemble of 6
networks) — a multi-day, multi-GPU undertaking, not attempted.

## Entry 4 — Deliberately writing the wrong backward pass first, and proving it's wrong

The paper's own Section 2 (quoted in entry 2) describes a real historical
failure mode: differentiate through `x_hat = (x - mu_B) / sqrt(sigma_B^2 +
eps)` as if `mu_B` and `sigma_B^2` were constants with respect to `x`,
instead of functions of every `x_i` in the batch. That's the "obvious"
derivative to write if you only glance at the forward formula.

Wrote exactly that version first, on purpose: `attempts/naive_batchnorm_backward.py`.
It gives `dx_i = dx_hat_i / sqrt(sigma_B^2 + eps)` — drops the `dL/dmu_B` and
`dL/dsigma_B^2` terms from the paper's full chain-rule expansion entirely.

Then checked it against a numerical gradient (`attempts/check_naive_fails.py`,
kept as a standalone script rather than folded into the real test suite,
since its only job is to produce this one piece of evidence):

```
$ python check_naive_fails.py
naive backward: worst relative error over 32 entries = 1.3707
```

A relative error of 1.37 on an 8x4 random input isn't a numerical-precision
issue (those show up as 1e-4 or smaller) — the naive gradient is simply the
wrong function. This matches the paper's own warning almost exactly: the
paper describes the forward-pass version of this mistake (treating `E[x]` as
independent of `b` when it was computed over data that includes `b`'s
effect) and says it causes the loss to stop decreasing while a parameter
drifts unboundedly. The backward-pass version of the same mistake (this one)
would show up as a model that trains, but on a quietly wrong gradient
signal — arguably worse, since there's no "it blew up" moment to notice it.

The fix (`bn_layer.py`) is the full five-line expansion from entry 2's
quoted equations, implemented term by term rather than collapsed into one
expression — specifically so each line maps to one quoted equation and can
be checked against the paper directly. Re-ran the check against the correct
version with a proper gradcheck covering `dx`, `dgamma`, `dbeta` together
(`test_gradcheck.py`):

```
$ python test_gradcheck.py
gradcheck passed, 50 entries (dx, dgamma, dbeta), worst relative error = 1.30e-09
```

1.30e-09 vs. 1.37 — not a close call. The naive version isn't "a bit off,"
it's categorically a different function, which is exactly what the paper's
own cautionary paragraph in Section 2 implies: half-differentiating through
a normalization step doesn't produce an approximately-correct gradient, it
produces one that actively fights the optimization.

## Entry 5 — Wiring BN into a full network, and the one simplification worth noting

`mlp.py` composes `BatchNorm1d` with dense layers into the 784-100-100-100-10
network from entry 3. One structural decision: the output layer never gets
BN (the paper's figures and MNIST description only ever mention BN on the
hidden sigmoid layers; the softmax+cross-entropy output is untouched, same
convention as 02_dropout not dropping-out the output layer).

First draft of `forward_train` had redundant, confusing branching — it
computed `a = y_bn if is_output else sigmoid(y_bn)` inside the `if
self.bn[l] is not None` branch, which can never actually trigger for the
output layer since `__init__` never attaches a `BatchNorm1d` to it. Dead
code that *looked* like a real branch. Cleaned it up before running
anything, since leaving confusing-but-technically-harmless code in just
because it happens to execute correctly isn't the same as it being right.

`test_mlp_gradcheck.py` (composing the already-verified `BatchNorm1d`
backward with the dense-layer backward, checking the composition itself,
not just each piece in isolation) passed on the first real attempt for
both `use_bn=False` and `use_bn=True`:

```
use_bn=False: 15 W entries checked, worst relative error = 3.30e-05
use_bn=True: 15 W entries checked, worst relative error = 1.25e-06
```

No debugging saga here, unlike 02_dropout's three-bug training run — the
isolated BN gradcheck in entry 4 already forced out the one subtle bug this
mechanism has (the batch-statistics dependency), so by the time it's wired
into a full network there wasn't a second hidden mistake waiting. Worth
recording as a data point either way: not every paper in this repo produces
a debugging saga, and that's fine — the discipline (write it, gradcheck it
before trusting any training curve) is what matters, not whether something
breaks.

## Entry 6 — Picking a learning rate the paper doesn't give, and a result that wasn't a bug

Entry 3 already flagged that Section 4.1 never states a learning rate for
the toy MNIST network (only for the unrelated, much bigger Inception
experiment in 4.2). Swept `lr in {0.01, 0.05, 0.1, 0.3, 1.0}` at a cheap
2000 steps before committing to the full 50000-step run, same discipline as
02_dropout's `epsilon_0` sweep:

```
lr= 0.01  nobn_final_acc=0.3766   bn_final_acc=0.8755
lr= 0.05  nobn_final_acc=0.3327   bn_final_acc=0.8501
lr= 0.10  nobn_final_acc=0.0980   bn_final_acc=0.8338
lr= 0.30  nobn_final_acc=0.0980   bn_final_acc=0.8637
lr= 1.00  nobn_final_acc=0.0980   bn_final_acc=0.1009
```

First reaction to the `0.0980` values at `lr>=0.1` was to suspect the same
kind of dead-sigmoid bug found in 02_dropout's debugging saga. It isn't:
`0.0980` is a degenerate constant-class prediction (same signature as every
other collapse seen in this repo so far), and it is happening to the
**no-BN** network specifically, at learning rates the **BN** network
tolerates just fine, reaching 83-86% in the same 2000 steps. That is not a
bug to fix — it is Section 3.3's claim, reproduced directly and
unprompted by a routine hyperparameter sweep:

> "Batch Normalization also makes training more resilient to the parameter
> scale... with Batch Normalization, back-propagation through a layer is
> unaffected by the scale of its parameters... Batch Normalization allows
> us to use much higher learning rates without the risk of divergence."

Rather than treating this as noise to route around by picking whichever LR
makes both networks look nice, it's used deliberately as two separate
experiments with two different jobs:
- **Main comparison, lr=0.01**: the only LR in the sweep where *both*
  networks actually train (don't collapse), which is the fair,
  apples-to-apples setting for Fig. 1(a) — same architecture, same LR,
  isolate BN as the only difference.
- **Stress test, lr=0.3** (Entry 3's extra ablation, beyond what the paper's
  MNIST section shows): chosen specifically *because* it's where no-BN
  collapses to chance (0.098) while BN still reaches 0.8637 in the sweep —
  the cleanest few-thousand-step demonstration available of the exact
  claim quoted above, not the smallest LR that happens to avoid collapse.

Also fixed a `RuntimeWarning: overflow encountered in exp` from `sigmoid()`
hit during the sweep at the higher LRs (large pre-activations send
`np.exp(-x)` to overflow) — clipped the input to `[-500, 500]` before the
exponential. The asymptotic sigmoid value (0 or 1) is correct either way;
this only silences the warning, it does not change any result, confirmed
by re-running both gradchecks afterward (unchanged: 3.30e-05 / 1.25e-06).

## Entry 7 — The main comparison came back backwards, and why

First full 50000-step run of the main comparison (lr=0.01, the "fair,
same-LR" setting from entry 6) did NOT reproduce Fig. 1(a)'s shape:

```
FINAL test accuracy: no-BN=0.9662 BN=0.9651
```

Not just close — **no-BN was converging faster** through most of training
(step 10000: no-BN 0.9422 vs. BN 0.8951; step 20000: no-BN 0.9590 vs. BN
0.9486), and BN's curve was visibly noisier late in training (oscillating
0.93-0.96 rather than settling). This is the opposite of what the paper
claims for this exact experiment, and it would have been easy to just
report "results were mixed, BN didn't clearly help here" and move on
(especially since the LR stress test in entry 6 had already independently
confirmed Section 3.3's claim) — but a literal reversal of the paper's own
central example is worth actually chasing down before writing it off as
noise or an inherent limitation.

Prime suspect: entry 2's deliberate choice to implement inference
statistics as a **literal running (arithmetic) average** across every
training mini-batch ever seen, rather than the exponential moving average
that later became standard. A plain average weighs the very first, noisiest
mini-batches (computed when the network is still near its random
initialization) exactly as heavily as the 50000th. If the network's
activation distribution shifts substantially over training — which,
notably, is *itself* internal covariate shift, the exact phenomenon this
paper is about — a plain running average's final value could end up a
blend of "where the network starts" and "where it ends," systematically
wrong for `forward_eval()` even though the batch-statistics-based training
path (`forward_train()`) was never affected at all. That asymmetry (train
path fine, inference path biased) matches the symptom: BN's degraded
curve only shows up in *test* accuracy, which is the one thing that goes
through `forward_eval()`.

Tested directly rather than assumed: `attempts/investigate_inference_stats.py`
retrains the BN network at the same lr=0.01 with a throwaway
`EMABatchNorm1d` subclass (exponential moving average, momentum 0.9)
standing in for the real class, leaving the then-current plain-average
`BatchNorm1d` untouched until there was a result to justify changing it:

```
[BN-EMA] step   5000 test_acc=0.9588   (plain average: 0.8515 at this point)
[BN-EMA] step  10000 test_acc=0.9738   (plain average: 0.8951)
FINAL BN-EMA test accuracy: 0.9757      (plain average: 0.9651, no-BN: 0.9662)
```

Confirmed. Switched `bn_layer.py`'s `BatchNorm1d` to use the exponential
moving average directly (momentum 0.9) as the real implementation, not
just a diagnostic side-branch — re-ran both gradchecks afterward to
confirm the change (pure bookkeeping, doesn't touch `forward_train`'s
returned value or `backward()`) didn't disturb anything already verified:
unchanged, 1.30e-09 and 3.30e-05/1.25e-06. Re-ran the full experiment with
the fix in place; final numbers now match the paper's claimed shape
cleanly:

```
FINAL test accuracy: no-BN=0.9662 BN=0.9757          (lr=0.01, main comparison)
no-BN @ lr=0.3: 0.1032 (chance)   BN @ lr=0.3: 0.9698  (stress test, entry 6)
```

BN now visibly trains faster (step 5000: 0.9588 vs. no-BN's 0.8420; step
10000: 0.9738 vs. 0.9422) in addition to reaching higher final accuracy —
the actual shape of Fig. 1(a), not just a win on the final number. The
figures committed in this repo (`figures/fig1a_test_accuracy.png`,
`figures/fig1bc_activation_percentiles.png`, `figures/lr_stress_test.png`)
are from this corrected run. `attempts/investigate_inference_stats.py`
is kept as the evidence trail for why the default changed, not as a live
alternative — it's redundant with `bn_layer.py` now that the real class
implements the same logic directly.

One honest caveat: the paper never actually specifies HOW to average
inference statistics across batches beyond "average over them" (quoted in
entry 2) — EMA being the right choice here is an empirical finding about
*this* experiment's dynamics (a network whose activation statistics shift
a lot early in training, trained for a fixed, fairly large number of
steps), not a proof that EMA is universally correct or that the paper
"meant" EMA all along. It's presented here as the corrected, better-
performing choice, backed by a direct side-by-side comparison — not as
uncovering what the paper secretly intended.
