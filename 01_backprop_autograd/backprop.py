"""
Learning Representations by Back-Propagating Errors
Rumelhart, Hinton & Williams, Nature 323, 533-536 (1986).

Hand-derived forward/backward pass for a single hidden layer of logistic
units, matching equations (1)-(9) of the paper exactly. No autodiff
library is used anywhere in this file; numpy is only a vectorized array
library here.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Network:
    """Two-layer (one hidden layer) feed-forward net of logistic units.

    Bias is treated as in the paper: "equivalent to a threshold ... can be
    treated just like the other weights" -- implemented here as a separate
    additive vector rather than literally augmenting the input with a
    constant-1 unit, which is algebraically identical but avoids
    reshaping every input matrix.
    """

    def __init__(self, n_in, n_hidden, n_out, init_range=0.3, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-init_range, init_range, size=(n_hidden, n_in))
        self.b1 = rng.uniform(-init_range, init_range, size=(n_hidden,))
        self.W2 = rng.uniform(-init_range, init_range, size=(n_out, n_hidden))
        self.b2 = rng.uniform(-init_range, init_range, size=(n_out,))

        # momentum terms, eq (9); start at zero (no previous step yet)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        """X: (n_cases, n_in). Returns (y1, y2): hidden and output activations."""
        x1 = X @ self.W1.T + self.b1       # eq (1)
        y1 = sigmoid(x1)                    # eq (2)
        x2 = y1 @ self.W2.T + self.b2       # eq (1)
        y2 = sigmoid(x2)                    # eq (2)
        return y1, y2

    def backward(self, X, y1, y2, D):
        """D: (n_cases, n_out) desired outputs. Returns summed-over-cases
        gradients dE/dW1, dE/db1, dE/dW2, dE/db2, matching eqs (4)-(7).
        The sum over cases is exactly eq (3)'s sum_c, done before the
        weight update per eq (8)/(9) ("full-batch").
        """
        dE_dy2 = y2 - D                              # eq (4), output units only
        dE_dx2 = dE_dy2 * y2 * (1.0 - y2)             # eq (5)
        dE_dW2 = dE_dx2.T @ y1                        # eq (6)
        dE_db2 = dE_dx2.sum(axis=0)                   # eq (6), bias "sender" y_i = 1

        dE_dy1 = dE_dx2 @ self.W2                     # eq (7): propagate to hidden layer
        dE_dx1 = dE_dy1 * y1 * (1.0 - y1)              # eq (5), applied one layer down
        dE_dW1 = dE_dx1.T @ X                          # eq (6)
        dE_db1 = dE_dx1.sum(axis=0)                    # eq (6)

        return dE_dW1, dE_db1, dE_dW2, dE_db2

    def step(self, grads, epsilon, alpha):
        """Momentum update, eq (9): delta_w(t) = -eps*dE/dw(t) + alpha*delta_w(t-1)."""
        dW1, db1, dW2, db2 = grads
        self.vW1 = -epsilon * dW1 + alpha * self.vW1
        self.vb1 = -epsilon * db1 + alpha * self.vb1
        self.vW2 = -epsilon * dW2 + alpha * self.vW2
        self.vb2 = -epsilon * db2 + alpha * self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def error(self, y2, D):
        """Eq (3): E = 1/2 * sum_c sum_j (y - d)^2."""
        return 0.5 * np.sum((y2 - D) ** 2)


def make_symmetry_dataset(n_bits=6):
    """All 2**n_bits binary vectors, labelled 1 if mirror-symmetric about
    the vector's centre, 0 otherwise. Matches Fig. 1's "64 possible input
    vectors" for n_bits=6.
    """
    n = 2 ** n_bits
    X = np.array([[(i >> b) & 1 for b in range(n_bits)] for i in range(n)], dtype=float)
    D = np.array([[1.0 if np.array_equal(row, row[::-1]) else 0.0] for row in X])
    return X, D


def train(epochs=1425, epsilon=0.1, alpha=0.9, n_hidden=2, seed=0, log_every=25):
    X, D = make_symmetry_dataset()
    net = Network(n_in=X.shape[1], n_hidden=n_hidden, n_out=1, seed=seed)

    history = []
    for sweep in range(1, epochs + 1):
        y1, y2 = net.forward(X)
        grads = net.backward(X, y1, y2, D)
        net.step(grads, epsilon, alpha)
        if sweep % log_every == 0 or sweep == 1:
            e = net.error(y2, D)
            history.append((sweep, e))

    y1, y2 = net.forward(X)
    final_error = net.error(y2, D)
    preds = (y2 >= 0.5).astype(float)
    accuracy = float(np.mean(preds == D))
    return net, X, D, history, final_error, accuracy


if __name__ == "__main__":
    net, X, D, history, final_error, accuracy = train()
    print(f"final error E = {final_error:.6f}")
    print(f"classification accuracy on all 64 patterns = {accuracy:.4f}")
    print("hidden layer weights (2 x 6):")
    print(net.W1)
    print("hidden layer biases:", net.b1)
    print("output layer weights (1 x 2):", net.W2)
    print("output layer bias:", net.b2)
