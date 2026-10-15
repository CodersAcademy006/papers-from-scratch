"""
AlexNet, rescaled for CIFAR-10 (JOURNAL.md entry 4): same 5-conv+3-FC
shape, ReLU everywhere, LRN after conv1/conv2 (before pooling),
overlapping max-pool, dropout on the first two FC layers, softmax+CE
output (10-way instead of 1000-way).
"""
import numpy as np
from conv2d import Conv2D
from lrn import LRN
from maxpool import MaxPool
from dropout import Dropout
from dense import Dense


class AlexNetCIFAR:
    def __init__(self, seed=0):
        s = iter(range(seed * 100, seed * 100 + 10))
        # Section 5's bias=1 (conv2/4/5, hidden FC) was meant to accelerate
        # ReLUs under the paper's own flat std=0.01 init. With He init
        # (conv2d.py, dense.py; JOURNAL.md entry 6) that rationale no longer
        # applies -- bias=1 would just reintroduce the same bias-dominates-
        # the-signal problem at a different layer, so bias is 0 everywhere.
        self.conv1 = Conv2D(3, 32, 5, stride=1, padding=2, bias_init=0.0, seed=next(s))
        self.lrn1 = LRN()
        self.pool1 = MaxPool(3, 2)

        self.conv2 = Conv2D(32, 64, 5, stride=1, padding=2, bias_init=0.0, seed=next(s))
        self.lrn2 = LRN()
        self.pool2 = MaxPool(3, 2)

        self.conv3 = Conv2D(64, 96, 3, stride=1, padding=1, bias_init=0.0, seed=next(s))
        self.conv4 = Conv2D(96, 96, 3, stride=1, padding=1, bias_init=0.0, seed=next(s))
        self.conv5 = Conv2D(96, 64, 3, stride=1, padding=1, bias_init=0.0, seed=next(s))
        self.pool5 = MaxPool(3, 2)

        self.fc6 = Dense(576, 384, bias_init=0.0, activation=True, seed=next(s))
        self.drop6 = Dropout(0.5)
        self.fc7 = Dense(384, 192, bias_init=0.0, activation=True, seed=next(s))
        self.drop7 = Dropout(0.5)
        self.fc8 = Dense(192, 10, bias_init=0.0, activation=False, seed=next(s))

    def params(self):
        return {
            'conv1.W': self.conv1.W, 'conv1.b': self.conv1.b,
            'conv2.W': self.conv2.W, 'conv2.b': self.conv2.b,
            'conv3.W': self.conv3.W, 'conv3.b': self.conv3.b,
            'conv4.W': self.conv4.W, 'conv4.b': self.conv4.b,
            'conv5.W': self.conv5.W, 'conv5.b': self.conv5.b,
            'fc6.W': self.fc6.W, 'fc6.b': self.fc6.b,
            'fc7.W': self.fc7.W, 'fc7.b': self.fc7.b,
            'fc8.W': self.fc8.W, 'fc8.b': self.fc8.b,
        }

    def forward(self, x, train=True, rng=None):
        rng = rng or np.random.default_rng()
        self._relu1 = np.maximum(0, self.conv1.forward(x))
        h = self.pool1.forward(self.lrn1.forward(self._relu1))

        self._relu2 = np.maximum(0, self.conv2.forward(h))
        h = self.pool2.forward(self.lrn2.forward(self._relu2))

        self._relu3 = np.maximum(0, self.conv3.forward(h))
        self._relu4 = np.maximum(0, self.conv4.forward(self._relu3))
        self._relu5 = np.maximum(0, self.conv5.forward(self._relu4))
        h = self.pool5.forward(self._relu5)

        self._flat_shape = h.shape
        flat = h.reshape(h.shape[0], -1)
        h = self.fc6.forward(flat)
        h = self.drop6.forward(h, train, rng)
        h = self.fc7.forward(h)
        h = self.drop7.forward(h, train, rng)
        logits = self.fc8.forward(h)
        return logits

    @staticmethod
    def softmax_ce(logits, y_onehot):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        dlogits = (probs - y_onehot) / N
        return loss, dlogits, probs

    def backward(self, dlogits):
        dh, dW8, db8 = self.fc8.backward(dlogits)
        dh = self.drop7.backward(dh)
        dh, dW7, db7 = self.fc7.backward(dh)
        dh = self.drop6.backward(dh)
        dh, dW6, db6 = self.fc6.backward(dh)
        dh = dh.reshape(self._flat_shape)

        dh = self.pool5.backward(dh)
        dh = dh * (self._relu5 > 0)
        dh, dW5, db5 = self.conv5.backward(dh)

        dh = dh * (self._relu4 > 0)
        dh, dW4, db4 = self.conv4.backward(dh)

        dh = dh * (self._relu3 > 0)
        dh, dW3, db3 = self.conv3.backward(dh)

        dh = self.pool2.backward(dh)
        dh = self.lrn2.backward(dh)
        dh = dh * (self._relu2 > 0)
        dh, dW2, db2 = self.conv2.backward(dh)

        dh = self.pool1.backward(dh)
        dh = self.lrn1.backward(dh)
        dh = dh * (self._relu1 > 0)
        dx, dW1, db1 = self.conv1.backward(dh)

        grads = {
            'conv1.W': dW1, 'conv1.b': db1,
            'conv2.W': dW2, 'conv2.b': db2,
            'conv3.W': dW3, 'conv3.b': db3,
            'conv4.W': dW4, 'conv4.b': db4,
            'conv5.W': dW5, 'conv5.b': db5,
            'fc6.W': dW6, 'fc6.b': db6,
            'fc7.W': dW7, 'fc7.b': db7,
            'fc8.W': dW8, 'fc8.b': db8,
        }
        return dx, grads
