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
