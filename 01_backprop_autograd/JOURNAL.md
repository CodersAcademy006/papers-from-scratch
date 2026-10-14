# Journal — 01: Learning Representations by Back-Propagating Errors

Rumelhart, D.E., Hinton, G.E. & Williams, R.J. *Nature* 323, 533-536 (1986). DOI: 10.1038/323533a0.

This file is written as the work happens, not reconstructed afterward. Entries are appended in chronological order.

## Entry 1 — Sourcing the paper

The paper predates arXiv (1986, published in *Nature*), so it is not in alphaxiv's index. Had to find a free mirror.

- First hit: `mlanthology.org` listing pointed to `https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf`. Fetching it returned an Anubis bot-protection error page (HTTP 403, error code `9e4edb5b6b850c41`), not the paper. Dead end, noted and moved on rather than fighting a bot wall.
- Second try: Hinton's own paper archive, `https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf`. This one worked at the HTTP level (binary PDF, 474.5KB, actually downloaded), but the WebFetch tool's HTML-to-markdown conversion step choked on it and reported back "corrupted/unreadable binary" instead of the extracted text — WebFetch is built for HTML pages run through a markdown converter, not for PDFs that need real text-layer extraction.
- WebFetch had, however, saved the raw PDF bytes to a local scratch path as a side effect of the fetch. Read directly from that saved file with the `Read` tool instead (which does proper PDF text-layer extraction), and got the full 4-page paper cleanly, equations and figures included. Lesson for future papers in this repo: if `WebFetch` on a PDF URL reports "binary/corrupted," check whether it still saved the file locally and `Read` it directly rather than giving up or re-fetching.

## Entry 2 — What the paper actually says

The paper describes **one specific algorithm** for one specific architecture: a layered, feed-forward network of logistic units, trained by full-batch gradient descent on squared error, optionally with a momentum term. It is not a description of a general reverse-mode automatic differentiation engine (that abstraction came later, and is a modern generalization of the same chain-rule idea) — the paper's own gradients are hand-derived for this exact architecture, and that is what gets implemented here, faithfully, rather than building a generic `Value`/`Tensor` autograd graph and calling it "the backprop paper." The folder name (`01_backprop_autograd`) is kept since this is where reverse-mode differentiation starts historically, but the implementation matches what equations (1)-(9) of the paper actually specify.

Forward pass, per unit j receiving from units i:

- Eq (1): total input `x_j = sum_i y_i * w_ji` (linear, bias is just a weight from a constant-1 unit)
- Eq (2): activation `y_j = 1 / (1 + exp(-x_j))` (logistic sigmoid)

Error measure, over cases c and output units j:

- Eq (3): `E = 1/2 * sum_c sum_j (y_{j,c} - d_{j,c})^2` (summed squared error, the 1/2 is there purely to cancel the exponent when differentiated)

Backward pass (derived via the chain rule, case index suppressed):

- Eq (4): `dE/dy_j = y_j - d_j` — only valid at **output** units, where the target `d_j` exists directly.
- Eq (5): `dE/dx_j = dE/dy_j * y_j * (1 - y_j)` — chain rule through the sigmoid's own derivative, `y(1-y)`.
- Eq (6): `dE/dw_ji = dE/dx_j * y_i` — the gradient w.r.t. a specific weight is just the upstream error-signal at the receiving unit times the sending unit's activation.
- Eq (7): `dE/dy_i = sum_j dE/dx_j * w_ji` — this is the actual "back-propagation" step: it reconstructs the same quantity used in (4) but for a **hidden** unit, by pushing every downstream unit's `dE/dx_j` back through the weight that connects them. This is what lets (5)-(6) be reapplied one layer down, recursively, with no separate target needed for hidden units.

Weight update:

- Eq (8): plain gradient descent, `delta_w = -epsilon * dE/dw`, accumulated over all cases before a single update ("full-batch").
- Eq (9): with momentum, `delta_w(t) = -epsilon * dE/dw(t) + alpha * delta_w(t-1)`, `alpha` an exponential decay in [0, 1) controlling how much of the previous step's direction carries over.

## Entry 3 — Choosing which experiment to reproduce, and why

The paper demonstrates three things: (a) mirror-symmetry detection on 6-bit binary vectors with a 2-hidden-unit network (Fig. 1), (b) an encoder for two isomorphic family trees expressed as relation triples (Figs. 2-4), (c) an equivalence between iterative (recurrent) nets and layered nets via weight-averaging (Fig. 5), presented as a sketch, not a worked experiment with numbers to check against.

Decision: implement and faithfully reproduce **(a), the symmetry detector**, and explicitly skip (b) and (c) for this pass.

- **(a) is fully specified and checkable.** The paper gives the exact architecture (6 input units, 2 hidden units, 1 output unit), the exact hyperparameters (`epsilon = 0.1`, `alpha = 0.9`), the exact training length (1425 sweeps over all 64 possible input vectors), and a specific, falsifiable claim about the learned solution: the two hidden units converge to identical-magnitude, opposite-sign weight patterns, symmetric about the input midpoint, with weights in the paper's Fig. 1 appearing in a 1:2:4 magnitude ratio moving outward from the center. That gives something concrete to verify the from-scratch implementation against, not just "does the loss go down."
- **(b), the family-tree encoder, is a data-engineering task bolted onto the same algorithm, not a new mechanism.** It needs 104 triples built from two specific isomorphic trees (Fig. 2) and a specific multi-group input encoding (24 + 12 one-hot-style input units across two groups, Fig. 3), none of which tests anything about backprop that (a) doesn't already test. Reproducing it would mostly be about faithfully transcribing two family trees from a figure, which is busywork, not signal. Left out by scope decision, not because it's hard.
- **(c), the recurrent/iterative-net equivalence, is a conceptual extension** (average the gradients of weight-tied layers to simulate a recurrent net as an unrolled layered net), presented with no concrete worked numbers in the paper to validate against. Out of scope for "implement the mechanism," since there is no single mechanism here beyond "tie weights and average their gradients," which is a trivial extension of what (a) already implements, with nothing new to verify.

**Why full-batch instead of per-case (online) updates:** the paper explicitly describes both options in the text around Eq. (8) and states full-batch accumulation is "the research reported here." Matching that, rather than the more common modern default of per-example or mini-batch SGD, since the point is to reproduce what the paper did, not to modernize it.

**Why no existing autodiff/deep-learning library:** this is the entire point of the exercise — equations (4)-(7) are implemented by hand as explicit matrix operations, not via `.backward()` on any framework. numpy is used only as a vectorized array library (matrix multiply, elementwise ops), carrying no differentiation machinery of its own.
