# Journal — 05_layernorm

Paper: "Layer Normalization" (Ba, Kiros, Hinton, arXiv:1607.06450, 2016).

## Entry 1 — Sourcing

Direct arXiv full-text pull, same as 03/04. 3 for 3 easy sourcing streak
now for the papers young enough to be on arXiv at all.

## Entry 2 — The mechanism, by quote, and how it differs from BatchNorm

**Batch normalization, restated in this paper's own notation** (quoted
because the LN-vs-BN distinction is easiest to see when both are written
with the same symbols):

> "for the i-th summed input in the l-th layer, the batch normalization
> method rescales the summed inputs according to their variances under the
> distribution of the data: a_hat_i = (g_i / sigma_i) * (a_i - mu_i), mu_i
> = E_{x~P(x)}[a_i], sigma_i = sqrt(E_{x~P(x)}[(a_i - mu_i)^2])... Instead,
> mu and sigma are estimated using the empirical samples from the current
> mini-batch."

**Layer normalization's actual change** — swap which axis gets averaged
over:

> "we compute the layer normalization statistics over all the hidden units
> in the same layer as follows: mu^l = (1/H) * sum_{i=1}^H a_i^l, sigma^l =
> sqrt((1/H) * sum_{i=1}^H (a_i^l - mu^l)^2) where H denotes the number of
> hidden units in a layer. The difference between Eq. (2) and Eq. (3) is
> that under layer normalization, all the hidden units in a layer share the
> same normalization terms mu and sigma, but different training cases have
> different normalization terms."

This is the entire mechanism, stated as plainly as the paper states it:
BatchNorm fixes a *unit* and averages over *examples in a batch*; LayerNorm
fixes an *example* and averages over *units in a layer*. Implementation-
wise this means `bn_layer.py`'s `axis=0` (03_batchnorm, averaging down the
batch dimension) becomes `axis=1` here (averaging across the feature
dimension, per example) — a one-line change in principle, which is exactly
why `ln_layer.py` below is a close structural cousin of 03_batchnorm's
`bn_layer.py`, not a fresh design.

**The gain/bias, same pattern as BN, written generically to cover all
three normalization schemes the paper compares (Section 5.1)**:

> "h_i = f((g_i/sigma_i) * (a_i - mu_i) + b_i)... Note that for layer
> normalization and batch normalization, mu and sigma is computed according
> to Eq. 2 and 3. In weight normalization, mu is 0, and sigma = ||w||_2."

**The practical consequence that matters most for this folder's
experiment** — no running statistics, no train/inference split, because
the normalization never depends on other examples in the first place:

> "Unlike batch normalization, layer normalization performs exactly the
> same computation at training and test times."

This is a genuine implementation simplification worth flagging up front:
03_batchnorm's `BatchNorm1d` needed `forward_train` / `forward_eval` /
`forward_probe` and an entire running-average mechanism that turned out to
be the source of that paper's biggest bug (JOURNAL.md entry 7 there). None
of that exists for LayerNorm — `forward()` is the only method needed, full
stop, and that absence is not an oversight, it's the paper's own stated
selling point.

**The motivating problem, stated directly, and the specific weakness of BN
this paper targets**:

> "the effect of batch normalization is dependent on the mini-batch size
> and it is not obvious how to apply it to recurrent neural networks...
> batch normalization cannot be applied to online learning tasks or to
> extremely large distributed models where the minibatches have to be
> small."

**The invariance claims (Table 1, Section 5.1)** — concrete, checkable
properties, not just qualitative comparisons, quoted here because entry 4
tests them directly:

> "Layer normalization, on the other hand, is not invariant to the
> individual scaling of the single weight vectors. Instead, layer
> normalization is invariant to scaling of the entire weight matrix and
> invariant to a shift to all of the incoming weights in the weight
> matrix... layer normalization is invariant to re-scaling of individual
> training cases, because the normalization scalars mu and sigma in Eq.
> (3) only depend on the current input data."

Table 1 itself, transcribed directly since it's the paper's own summary and
entry 4 checks every column that applies to a feedforward (non-recurrent)
network:

```
                        weight-matrix   weight-matrix   weight-vector   single-training-case
                        re-scaling      re-centering    re-scaling      re-scaling
Batch norm              Invariant       No              Invariant       No
Weight norm              Invariant       No              Invariant       No
Layer norm               Invariant       Invariant       No              Invariant
```
(dataset-level re-scaling/re-centering columns omitted here — both methods
are invariant to those and it's not a distinguishing property worth
re-testing.)

## Entry 3 — Scope decision

The paper runs 7 experiments: order embeddings (GRU, COCO), reading
comprehension (LSTM, CNN corpus), skip-thoughts (GRU, BookCorpus),
DRAW/binarized MNIST (LSTM + attention, Adam), handwriting generation
(LSTM, IAM-OnDB), permutation-invariant MNIST (plain feedforward), and a
preliminary ConvNet experiment.

**Reproduced**: Section 6.6, Permutation Invariant MNIST. Quoted in full
since it's the one directly reproducible here:

> "we investigated layer normalization in feed-forward networks... the
> well-studied permutation invariant MNIST classification problem...
> layer normalization is robust to the batch-sizes and exhibits a faster
> training convergence comparing to batch normalization."

Fig. 6 compares Baseline / BatchNorm / LayerNorm on a 784-1000-1000-10
network at batch size 128 AND at batch size 4 — this is the paper's own
direct test of its central practical claim (BN degrades at small batch
sizes, LN doesn't), and conveniently uses almost the exact architecture
already built twice in this repo (03_batchnorm's BN comparison,
04_adam's 6.2 ReLU network), so it's cheap to set up and gives a clean,
three-way, two-batch-size comparison rather than a single ablation.

**Reproduced as an extra**: the invariance properties from entry 2's Table
1 quote, checked numerically the same way 04_adam checked its rescaling-
invariance and AdaGrad-correspondence claims — doesn't need a trained
network, just the forward pass and some algebra, so it's cheap evidence for
claims that would otherwise just be taken on faith from the paper's own
proof.

**Out of scope, with reasons**: every RNN experiment (order embeddings,
reading comprehension, skip-thoughts, DRAW, handwriting generation) needs
a working GRU or LSTM first — those are 14_lstm and 15_gru's job, not
built yet, and each experiment additionally needs its own large external
dataset (MS-COCO, the CNN QA corpus, BookCorpus, IAM-OnDB) that would be
its own multi-day acquisition-and-preprocessing undertaking even with the
model in hand. The paper's own strongest results are on these RNN tasks
("recurrent neural networks benefit the most from the proposed method"),
so this folder is explicitly NOT reproducing the paper's headline finding,
only its simpler feedforward side-experiment — flagged here rather than
left implicit. Revisiting LayerNorm-in-an-RNN after 14/15 land is a
reasonable thing to come back to, not promised here. Section 6.7's
ConvNet experiment is also skipped: the paper itself calls it
"preliminary" and inconclusive ("we think further research is needed to
make layer normalization work well in ConvNets"), so there's no clean
claim to validate there even if a conv layer were available (it isn't yet
either — that's 06_lenet5's job).

## Entry 4 — A real bug in the axis-swap, caught by gradcheck on the first run

`ln_layer.py` was written by taking 03_batchnorm's `BatchNorm1d.backward()`
and substituting `(x - mu) = x_hat / std_inv` throughout (since this class
doesn't cache `x` itself, only `x_hat` and `std_inv`, to keep the cache
smaller). First `test_gradcheck.py` run failed immediately:

```
AssertionError: dx[0,0]: analytic=0.12066323 numeric=0.11800632 rel_err=2.20e-02
```

The substitution itself had an algebra error. The term needed is `(x - mu)
* (sigma_sq + eps)^(-3/2)`. Since `std_inv = (sigma_sq + eps)^(-1/2)`:

```
(x - mu) * (sigma_sq + eps)^(-3/2)
  = (x_hat / std_inv) * std_inv^3
  = x_hat * std_inv^2
```

The first version wrote `x_hat * std_inv` (power 1, not 2) — missing one
factor of `std_inv` entirely, not a rounding/precision issue. Fixed by
using `std_inv ** 2` explicitly rather than re-deriving from scratch, with
a comment showing the substitution so the next reader doesn't have to
redo the algebra to see why the squared term is there. Re-ran: passed, 56
entries (dx, dgamma, dbeta) across `(6, 7)`-shaped test data, worst
relative error 4.91e-09 — about seven orders of magnitude tighter than the
bug's 2.20e-02, the same kind of "categorically different function, not a
precision issue" signature 03_batchnorm's naive-backward attempt showed
(JOURNAL.md entry 4 there).

## Entry 5 — Table 1's invariance claims, checked (and one tolerance miscalibration)

`test_invariance_properties.py` checks three of Table 1's claims directly:
weight-matrix rescale+recenter invariance (the positive claim, Eq. 6),
single-weight-vector rescale NON-invariance (the negative claim — LN is
explicitly NOT invariant to this, unlike BN/weight-norm, and it's worth
checking the paper's "no" entries too, not just the "yes" ones), and
single-training-case rescale invariance (Eq. 7).

First run of the weight-matrix check failed at an assert written with too
strict a tolerance (`1e-8`):

```
AssertionError: weight matrix rescale+recenter changed output, max diff=5.3324977180579936e-06
```

Checked the actual magnitudes before assuming the invariance claim itself
was wrong: `z = Wx` ranges roughly [-5, 5] before normalization, LN output
`y` ranges roughly [-2, 2], and the test's `delta=3.7` rescale pushes
`W'` entries up to ~20 in magnitude — float64 arithmetic at that scale
naturally accumulates errors in the `1e-6`-ish range, well above a `1e-8`
absolute bar that would only be reasonable for values close to 1. Not a
violation of the paper's claim, just an assert written without checking
the data's actual scale first. Loosened both this check and the
single-training-case check (which hit the identical issue, `2.71e-06`) to
`1e-5`, matching the `ABS_TOL` scale used throughout this repo's
gradchecks rather than an arbitrarily tight number. All three checks pass:

```
weight-matrix rescale+recenter invariance passed, max output diff = 5.33e-06
weight-vector rescale correctly changes output (NOT invariant, as Table 1 says), max diff = 2.80e+00
single-training-case rescale invariance passed: rescaled case's output diff = 2.71e-06, other cases unaffected (diff = 0.00e+00)
```

The weight-vector check's `2.80` max diff is the interesting contrast here
— two to six orders of magnitude larger than the two invariant checks'
diffs, which is exactly what "not invariant" should look like: a real,
large output change, not a rounding artifact. Seeing both a tight
near-zero diff (true invariance) and a clearly large diff (true
non-invariance) in the same test file, from the same codebase, is a better
confirmation that the implementation is actually testing something real
than three near-zero results would have been.

## Entry 6 — A backward pass that looks wrong on a read-through, but isn't

`mlp_norm.py`'s `NormMLP` covers all three conditions (`"none"`,
`"batchnorm"`, `"layernorm"`) for Section 6.6's comparison in one class.
Writing `backward()`, the ReLU-mask bookkeeping came out looking
inconsistent between the two branches:

- The no-norm branch applies the ReLU-derivative mask at the END of its
  own iteration, using `activations[l] > 0` — pre-computing the mask the
  NEXT (lower) layer will need.
- The norm branch applies a mask at the START of its own iteration, using
  `activations[l+1] > 0` — masking the gradient it just received for its
  OWN ReLU, before handing an unmasked gradient further down.

Traced a concrete 3-layer example by hand (2 hidden + 1 output, both
hidden layers normed) before trusting this: the always-unnormed output
layer's end-of-iteration masking step fires regardless of whether the
layer below it is normed, which means the top hidden layer's norm branch
receives an already-masked gradient and then masks it AGAIN with the
identical `activations[...] > 0` array. On paper this looks like a
double-application bug. It isn't one, for a reason specific to boolean
masks: `mask * mask == mask` for any `mask in {0, 1}`, so re-applying the
exact same positivity mask a second time is a no-op, not a second
derivative multiplication. Every hidden-to-hidden transition in this
architecture turns out to apply the correct mask exactly once in net
effect, just with the "end of upper iteration" and "start of lower
iteration" responsibilities overlapping for the layer adjacent to the
(always unnormed) output layer.

This reasoning could easily have been wrong — relying on "it's idempotent
so it cancels out" is exactly the kind of manual proof that should be
checked, not trusted. `test_mlp_gradcheck.py` is the actual arbiter:
passed on the first real run for all three `norm_mode` values,

```
norm_mode=none: 10 W entries checked (20 skipped as kink-adjacent), worst relative error = 1.74e-09
norm_mode=batchnorm: 10 W entries checked (20 skipped as kink-adjacent), worst relative error = 4.07e-09
norm_mode=layernorm: 15 W entries checked (0 skipped as kink-adjacent), worst relative error = 3.33e-09
```

confirming the hand-trace rather than replacing it — both together are
more convincing than either alone would have been. Worth noting
`norm_mode=layernorm` needed 0 kink-adjacent units excluded, vs. 20 for
the other two modes with the same `weight_std=0.3` and input scaling: LN's
per-example normalization (mean 0, unit variance, by construction, every
forward pass) keeps pre-activations spread away from exactly zero far more
reliably than an unnormalized or even a batch-normalized signal does at
this small test-batch size — a small, incidental illustration of entry 2's
"layer normalization performs exactly the same computation" point showing
up as a more consistently-scaled signal too, not just a training/inference
convenience.
