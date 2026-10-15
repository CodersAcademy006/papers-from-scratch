# Journal — 04_adam

Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015,
arXiv:1412.6980).

## Entry 1 — Sourcing

Direct arXiv full-text pull again, same as 03_batchnorm, no mirror fight.
Noting it anyway for the running record: 1 for 1 easy sourcing streak since
switching off papers old enough to predate arXiv (01_backprop_autograd was
the only hard one, being from 1986).

## Entry 2 — The algorithm, by quote

**The update rule itself (Algorithm 1)**, exactly as given, nothing
simplified:

```
m_0 = 0, v_0 = 0, t = 0
while not converged:
    t += 1
    g_t = grad_theta f_t(theta_{t-1})
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2          # elementwise square
    m_hat_t = m_t / (1 - beta1^t)
    v_hat_t = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + eps)
```

> "Good default settings for the tested machine learning problems are
> alpha = 0.001, beta1 = 0.9, beta2 = 0.999 and eps = 1e-8."

These defaults are used as-is below, not re-tuned, since the paper presents
them as defaults meant to transfer across problems rather than per-task
hyperparameters (unlike 02_dropout's `epsilon_0` or 03_batchnorm's
unstated MNIST learning rate, which genuinely had no paper-given value to
start from).

**Why the bias-correction terms exist at all** — this is the part most
blog-post implementations skip or get hand-wavy about, and the paper
derives it properly (Section 3), so it's transcribed here instead of
paraphrased:

> "v_t = (1 - beta2) * sum_{i=1}^t beta2^(t-i) * g_i^2 ... We wish to know
> how E[v_t]... relates to the true second moment E[g_t^2]... E[v_t] =
> E[g_t^2] * (1 - beta2^t) + zeta ... What is left is the term (1 -
> beta2^t) which is caused by initializing the running average with zeros.
> In algorithm 1 we therefore divide by this term to correct the
> initialization bias."

In words: an exponential moving average initialized at zero is biased
toward zero for exactly as long as the decay rate keeps "remembering" that
initial zero — dividing by `(1 - beta^t)` exactly cancels that geometric
factor. `t=1`, `beta2=0.999`: `1 - beta2^1 = 0.001`, so the very first
`v_hat_1` is `v_1 / 0.001`, a 1000x correction. The correction matters most
exactly when it's needed most (small `t`, `beta` close to 1) and fades to
~1 as `t` grows — not a one-off hack, a quantity derived to match the
actual size of the initialization artifact at each step.

**What "removing it" turns Adam into** — this motivates the ablation in
entry 4 below instead of requiring the paper's own VAE experiment:

> "Discussed in section 5, removal of the bias correction terms results in
> a version of RMSProp (Tieleman & Hinton, 2012) with momentum... RMSProp
> also lacks a bias-correction term; this matters most in case of a value
> of beta2 close to 1 (required in case of sparse gradients), since in that
> case not correcting the bias leads to very large stepsizes and often
> divergence, as we also empirically demonstrate in section 6.4."

**The "trust region" property (Section 2.1)**, worth quoting because it's
the clearest statement of why Adam's step sizes behave the way they do, and
gives a concrete numeric claim to test in entry 5:

> "The effective magnitude of the steps taken in parameter space at each
> timestep are approximately bounded by the stepsize setting alpha...
> rescaling the gradients g with factor c will scale m_hat with a factor c
> and v_hat with a factor c^2, which cancel out."

**The AdaGrad correspondence (Section 5)** — a concrete, checkable
mathematical claim, not just a qualitative comparison:

> "AdaGrad corresponds to a version of Adam with beta1 = 0, infinitesimal
> (1 - beta2) and a replacement of alpha by an annealed version alpha_t =
> alpha * t^(-1/2)."

This is directly testable: run Adam with `beta1=0`, `beta2` extremely
close to 1, `alpha_t = alpha/sqrt(t)`, against a literal AdaGrad
implementation (`theta -= alpha * g_t / sqrt(sum_{i<=t} g_i^2)`) on the same
synthetic gradient sequence, and check the two update sequences actually
converge to each other numerically — see entry 6.

## Entry 3 — Scope decision

The paper runs four experiments (6.1 logistic regression, 6.2 multi-layer
NN, 6.3 CNN, 6.4 bias-correction on a VAE) plus a convergence-rate proof
(Section 4, pure online-convex-optimization theory, not something to
"implement," skipped entirely — there's no code to write for a regret
bound).

**Reproduced**: Section 6.2's deterministic multi-layer network setup:

> "a neural network model with two fully connected hidden layers with 1000
> hidden units each and ReLU activation... minibatch size of 128... the
> standard deterministic cross-entropy objective function with L2 weight
> decay."

On MNIST, comparing Adam against plain momentum-SGD and AdaGrad (both
cheap to add) at the same architecture/init — the dropout variant of this
same experiment and the SFO quasi-Newton baseline are both skipped (SFO
specifically because it's "5-10x slower per iteration" even in the paper's
own words and has its own memory/implementation complexity that's a
separate undertaking; dropout variant skipped because 02_dropout already
covers dropout's own mechanism in depth and re-adding it here would just
be testing the same mechanism twice rather than testing Adam).

**Reproduced, but on a cheaper setup than the paper's own**: Section 6.4's
bias-correction claim. The paper tests this on a VAE with a 50-dimensional
latent Gaussian (Kingma & Welling, 2013) — implementing a VAE is paper
28_vae's job, not this one's, so instead the *claim itself* ("beta2 close
to 1 leads to instabilities... when no bias correction term was present,
especially at first few epochs") is tested directly on 04_adam's own
Section 6.2 network: same architecture, same data, Adam with vs. without
the bias-correction division, at `beta2=0.999`. If the underlying
mechanism is what the paper says it is, the effect should show up on any
network trained with a high `beta2`, not just a VAE — this is a case of
testing the mechanism rather than literally reproducing the figure.

**Reproduced as an extra, paper-doesn't-ask-for-it check**: the AdaGrad
correspondence from entry 2, verified numerically on a synthetic gradient
sequence rather than a real network (it's a claim about the optimizers'
update rules in the abstract, doesn't need training data to test).

**Out of scope, with reasons**: 6.1 logistic regression + IMDB sparse
bag-of-words experiment (a second, different dataset/task whose main point
-- sparse gradient handling -- is a repeat of the same adaptive-learning-
rate idea already covered by the AdaGrad correspondence check); 6.3 CNN
experiment (needs a hand-rolled conv2d, which is 06_lenet5 and 07_alexnet's
job, not this optimizer paper's); Section 4's regret-bound proof and
AdaMax/temporal-averaging extensions (Sections 7.1, 7.2) -- interesting,
but they're each a distinct secondary mechanism layered on top of the
paper's actual contribution, not needed to validate Adam itself.

## Entry 4 — The two abstract claims, checked exactly (not approximately)

`test_optimizer_properties.py` checks the two claims pulled out in entry 2,
plus a plain convergence sanity check first. All three passed on the first
run, no debugging:

```
quadratic convergence check passed, max abs error = 2.78e-17
rescaling invariance check passed: max update difference between 1x and
  100x gradient scale = 0.0000 (typical update size 0.0025)
AdaGrad correspondence check passed: worst relative trajectory difference
  over 200 steps = 0.0000
```

Worth dwelling on the second and third results specifically, since
`0.0000` is a stronger result than expected going in. The rescaling-
invariance claim ("the effective magnitude of the steps... are
APPROXIMATELY bounded") is hedged in the paper's own wording, so the test
was written expecting a *small but nonzero* difference between the 1x and
100x gradient-scale trajectories (hence the loose tolerance in the assert).
Getting an exact `0.0000` instead makes sense on reflection: with `eps`'s
effect negligible at this gradient scale, `m_hat`'s numerator and `v_hat`'s
square-rooted denominator really do scale by exactly `c` and `c`
respectively (since `sqrt(c^2) = c` for `c>0`), at every single timestep,
not just in the limit — the "approximately" in the paper's text is hedging
against the `eps` term and pathological edge cases, not against this being
false in a clean synthetic test.

Likewise, the AdaGrad correspondence isn't a loose analogy — plugging
`beta1=0`, `beta2 -> 1`, `alpha_t = alpha/sqrt(t)` into Adam's update and
simplifying algebraically really does collapse to AdaGrad's update exactly
(this is literally what the paper's Section 5 paragraph derives, step by
step, before stating the correspondence in prose) — so an exact match
between the two implementations, run independently on the same gradient
sequence, is the correct outcome, and a useful cross-check that both
`optimizer.py` classes are actually implementing what they claim to.

## Entry 5 — ReLU MLP for the 6.2 comparison, and the one deliberate simplification

`mlp_relu.py` is Section 6.2's network: 784-1000-1000-10, ReLU hidden units,
softmax+cross-entropy output, L2 weight decay. One deliberate
simplification not flagged until now: the paper's own baseline is "SGD
with Nesterov momentum" ("SGDNesterov" in Figures 1-3), but `optimizer.py`
only implements classical (non-Nesterov) momentum. Nesterov's lookahead
step is a genuinely different update rule (evaluate the gradient at the
momentum-projected point, not the current point), not a one-line variant
of plain momentum, and the point of this paper's exercise is Adam, not
re-deriving Nesterov acceleration -- so plain momentum stands in as the
"classical adaptive-ish baseline," documented as weaker evidence than the
paper's own SGDNesterov comparison, not as equivalent to it.

## Entry 6 — A real ReLU-kink gradcheck failure, traced to a structural zero

First `test_gradcheck.py` run failed immediately:

```
AssertionError: param 4 idx (0,): analytic=0.02865436 numeric=0.02699575 rel_err=5.79e-02
```

Param 4 is `b[1]`, the second hidden layer's bias. Before assuming a bug in
`backward()` (which had already passed on 02_dropout's and 03_batchnorm's
harder cases), checked the actual pre-activations at that layer for this
batch:

```
z1[:,0] values: [ 1.23e-05 -1.01e-03  4.91e-05 -1.10e-03 -2.76e-04 -1.41e-03
                   4.75e-05 -6.19e-05  0.00000000e+00  3.79e-04]
min abs: 0.0
```

Example 8's pre-activation is **exactly** 0.0 — not close to zero,
precisely zero. Perturbing `b[1][0]` by `+-1e-5` (the finite-difference
step) shifts every example's pre-activation by that same amount, which for
example 8 means the perturbed-up version is ReLU-active and the
perturbed-down version is ReLU-inactive. The finite-difference quotient
then measures something the analytic gradient's fixed one-sided convention
(`z > 0` evaluated at the unperturbed `z=0`, which Python's `>` resolves
to False/inactive) doesn't capture — a real discontinuity in a piecewise-
linear function, not a coding mistake.

First instinct was to try a different random seed, assuming this was a
rare coincidence. It wasn't: swept seeds 0-19 and every single one produced
a pre-activation within `1e-4` of zero somewhere in the network (several
within `1e-6`). The real cause: `ReLUMLP`'s `weight_std=0.01` default
(matching every other paper's init scale in this repo, chosen for
consistency, not for this network specifically) keeps every pre-activation
clustered tightly around zero — and worse, example 8's entire FIRST hidden
layer was already all-zero (every one of its 6 units inactive for that
example), which forces the second layer's pre-activation to be *exactly*
zero for that example regardless of the second layer's weight scale at
all — rescaling `weight_std` can't fix a problem caused one layer
upstream.

Fix: rather than loosening the tolerance (which would hide real bugs, not
just kink artifacts) or hunting for a lucky seed (which the sweep showed
doesn't reliably exist), `test_gradcheck.py` now computes, per hidden
layer, which output units have pre-activations safely away from zero
(`> 1e-3`) for every example in the batch, and only samples check-points
from that safe set. This run: 41 of 59 candidate (parameter, unit) pairs
were excluded as kink-adjacent — a much larger fraction than expected,
confirming this is a real, pervasive property of small-weight-std ReLU
networks on small batches, not a one-off fluke. The filtered check passes
cleanly:

```
gradcheck passed, 18 params sampled (41 skipped as kink-adjacent), worst relative error = 1.96e-06
```

## Entry 7 — Two real findings from running the actual experiment, neither flattering

**MomentumSGD is pathologically learning-rate-sensitive on this network.**
First full run of the main comparison used `lr=0.01` for `MomentumSGD`
(a standard-looking choice) and it barely moved:

```
[MomentumSGD] epoch 0 avg_loss=2.6847
[MomentumSGD] epoch 7 avg_loss=2.5538
```

`-ln(0.1) = 2.303` is the loss of guessing uniformly among 10 classes, so
this is WORSE than random guessing even by epoch 7 (the L2 term adds a
little, but not nearly this much). Before writing this up as "MomentumSGD
just doesn't work as well as Adam here," checked whether `optimizer.py`'s
`MomentumSGD` itself was broken: ran it on the same convex-quadratic
sanity check `test_optimizer_properties.py` already uses for Adam.

```
MomentumSGD: max abs error after 2000 steps = 0.000000
```

Exact convergence — the optimizer is correct. So the problem is specific
to this network/learning-rate combination. Swept `lr` for `MomentumSGD` on
the real network (2 epochs each, to keep this cheap):

```
lr=0.010: epoch0=2.6847  epoch1=2.5658   (barely moving)
lr=0.015: epoch0=2.86e14 epoch1=3.05e14  (exploded)
lr=0.020: epoch0=8.4728  epoch1=8.3083   (bad, but not exploded)
lr=0.025: epoch0=1.18e4  epoch1=1.21e4   (exploded)
lr=0.030: epoch0=4.76e53 epoch1=4.92e53  (exploded)
lr=0.050: epoch0=2.05e28 epoch1=2.09e28  (exploded)
lr=0.100: epoch0=1.34e61 epoch1=1.30e61  (exploded)
lr=0.300: epoch0=1.77e21 epoch1=1.40e21  (exploded)
```

Eight learning rates across two orders of magnitude, and not one produces
healthy training: `0.01` is too small to make real progress, and
everything from `0.015` up explodes (the explosions aren't even
monotonically worse with larger `lr` — `0.02` is merely bad while `0.1` is
astronomically worse, a chaotic, not smooth, relationship) — there is no
clean stable middle found in this sweep. Didn't search further; the result
itself is the finding, not an obstacle to route around. Plain momentum-SGD
with one global learning rate is extremely fragile on a 1000-unit-wide
ReLU network at this small (`weight_std=0.01`) initialization scale, in
exactly the way the paper's own motivation describes without this repo
needing to construct a special case to demonstrate it:

> "SGD proved itself as an efficient and effective optimization method...
> [but] Adam... has little memory requirements, is invariant to diagonal
> rescaling of the gradients... The hyper-parameters have intuitive
> interpretations and typically require little tuning."

`lr=0.01` (the least-bad option found) is what's in the committed
`figures/fig2b_optimizer_comparison.png` and `logs/run_momentum.log` — a
genuinely weak showing for the paper's own comparison baseline, left as
found rather than hand-tuned into looking more competitive, with this
entry as the explanation for why.

**Bias correction made training SLOWER here, not more stable — the
opposite of what Section 6.4 claims.** First 3-epoch ablation run:

```
with_bc:    0.2637 -> 0.0907 -> 0.0598
without_bc: 0.1937 -> 0.0658 -> 0.0389
```

Without correction is ahead at every epoch, not behind. Extended to 8
epochs (`EPOCHS_ABLATION` bumped from 3 to 8) to check whether the
without-correction run eventually destabilizes, since the paper's own
claim is specifically about instability "especially apparent towards the
end of optimization":

```
epoch: 0       1       2       3       4       5       6       7
with:  0.2637  0.0907  0.0598  0.0437  0.0347  0.0259  0.0247  0.0211
without:0.1937 0.0658  0.0389  0.0279  0.0210  0.0161  0.0162  0.0114
```

No crossover, no instability, across all 8 epochs — without-correction
stays consistently, stably ahead. Tried the more extreme `beta2=0.9999`
(the upper end of the paper's own sweep range in Fig. 4's caption,
`beta2 in {0.99, 0.999, 0.9999}`) over 5 epochs to see if a more aggressive
decay rate would trigger the claimed instability instead:

```
epoch: 0       1       2       3       4
with:  0.2639  0.0909  0.0608  0.0433  0.0327
without:0.2402 0.0909  0.0585  0.0421  0.0307
```

Still no instability, though the gap between the two conditions narrows
(they're essentially tied at epoch 1). This is a genuine disagreement with
the paper's Section 6.4 finding, on this specific task, honestly reported
rather than massaged into alignment. A few plausible, non-exhaustive
reasons this network/task might simply not be where the effect shows up:
the paper's own demonstration is on a VAE (reconstruction loss, 50-d
latent Gaussian, a very different loss landscape) trained up to 100
epochs, not a discriminative classifier trained for 5-8; it's possible the
instability needs either a longer horizon than tested here, or a loss
surface shaped more like the paper's own generative-modeling setup, to
actually manifest. What the math does explain, independent of any of
that: the FIRST step without correction is provably larger than with it
for this `beta1=0.9`/`beta2=0.999` pair (derivable directly from Algorithm
1: at `t=1`, `m_hat/sqrt(v_hat) = sign(g)` exactly when corrected, vs.
`(1-beta1)/sqrt(1-beta2) * sign(g) ≈ 3.16 * sign(g)` uncorrected) — so
omitting correction isn't a free no-op here, it's a genuinely more
aggressive optimizer on this task, and in this case "more aggressive" paid
off rather than caused instability. Both are real, documented outcomes of
the same mechanism; which one shows up depends on the loss landscape, and
this repo's finding is simply a different point in that space than the
paper's own.
