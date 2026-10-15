"""
VGG configuration A, rescaled for CIFAR-10 (JOURNAL.md entry 3):
11 weight layers (8 conv + 3 FC), channel counts unchanged from the
paper (64,128,256,256,512,512,512,512) since five 2x2/stride-2
max-pools take 32x32 -> 1x1 exactly as the paper's own design takes
224x224 -> 7x7. Dropout 0.5 on the first two FC layers, softmax+CE
output (10-way instead of 1000-way).
"""
import numpy as np
from conv2d import Conv2D
from maxpool import MaxPool
from dropout import Dropout
from dense import Dense


class VGG_A_CIFAR:
    def __init__(self, seed=0):
        s = iter(range(seed * 100, seed * 100 + 15))
        self.conv1_1 = Conv2D(3, 64, 3, stride=1, padding=1, seed=next(s))
        self.pool1 = MaxPool(2, 2)

        self.conv2_1 = Conv2D(64, 128, 3, stride=1, padding=1, seed=next(s))
        self.pool2 = MaxPool(2, 2)

        self.conv3_1 = Conv2D(128, 256, 3, stride=1, padding=1, seed=next(s))
        self.conv3_2 = Conv2D(256, 256, 3, stride=1, padding=1, seed=next(s))
        self.pool3 = MaxPool(2, 2)

        self.conv4_1 = Conv2D(256, 512, 3, stride=1, padding=1, seed=next(s))
        self.conv4_2 = Conv2D(512, 512, 3, stride=1, padding=1, seed=next(s))
        self.pool4 = MaxPool(2, 2)

        self.conv5_1 = Conv2D(512, 512, 3, stride=1, padding=1, seed=next(s))
        self.conv5_2 = Conv2D(512, 512, 3, stride=1, padding=1, seed=next(s))
        self.pool5 = MaxPool(2, 2)

        self.fc6 = Dense(512, 4096, activation=True, seed=next(s))
        self.drop6 = Dropout(0.5)
        self.fc7 = Dense(4096, 4096, activation=True, seed=next(s))
        self.drop7 = Dropout(0.5)
        self.fc8 = Dense(4096, 10, activation=False, seed=next(s))

        self._convs = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv3_2,
                        self.conv4_1, self.conv4_2, self.conv5_1, self.conv5_2]
        self._conv_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv3_2',
                             'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2']

    def params(self):
        p = {}
        for name, c in zip(self._conv_names, self._convs):
            p[f'{name}.W'] = c.W
            p[f'{name}.b'] = c.b
        for name, d in [('fc6', self.fc6), ('fc7', self.fc7), ('fc8', self.fc8)]:
            p[f'{name}.W'] = d.W
            p[f'{name}.b'] = d.b
        return p

    def forward(self, x, train=True, rng=None):
        rng = rng or np.random.default_rng()
        self._r1 = np.maximum(0, self.conv1_1.forward(x))
        h = self.pool1.forward(self._r1)

        self._r2 = np.maximum(0, self.conv2_1.forward(h))
        h = self.pool2.forward(self._r2)

        self._r3a = np.maximum(0, self.conv3_1.forward(h))
        self._r3b = np.maximum(0, self.conv3_2.forward(self._r3a))
        h = self.pool3.forward(self._r3b)

        self._r4a = np.maximum(0, self.conv4_1.forward(h))
        self._r4b = np.maximum(0, self.conv4_2.forward(self._r4a))
        h = self.pool4.forward(self._r4b)

        self._r5a = np.maximum(0, self.conv5_1.forward(h))
        self._r5b = np.maximum(0, self.conv5_2.forward(self._r5a))
        h = self.pool5.forward(self._r5b)

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
        dh = dh * (self._r5b > 0)
        dh, dW5b, db5b = self.conv5_2.backward(dh)
        dh = dh * (self._r5a > 0)
        dh, dW5a, db5a = self.conv5_1.backward(dh)

        dh = self.pool4.backward(dh)
        dh = dh * (self._r4b > 0)
        dh, dW4b, db4b = self.conv4_2.backward(dh)
        dh = dh * (self._r4a > 0)
        dh, dW4a, db4a = self.conv4_1.backward(dh)

        dh = self.pool3.backward(dh)
        dh = dh * (self._r3b > 0)
        dh, dW3b, db3b = self.conv3_2.backward(dh)
        dh = dh * (self._r3a > 0)
        dh, dW3a, db3a = self.conv3_1.backward(dh)

        dh = self.pool2.backward(dh)
        dh = dh * (self._r2 > 0)
        dh, dW2, db2 = self.conv2_1.backward(dh)

        dh = self.pool1.backward(dh)
        dh = dh * (self._r1 > 0)
        dx, dW1, db1 = self.conv1_1.backward(dh)

        grads = {
            'conv1_1.W': dW1, 'conv1_1.b': db1,
            'conv2_1.W': dW2, 'conv2_1.b': db2,
            'conv3_1.W': dW3a, 'conv3_1.b': db3a,
            'conv3_2.W': dW3b, 'conv3_2.b': db3b,
            'conv4_1.W': dW4a, 'conv4_1.b': db4a,
            'conv4_2.W': dW4b, 'conv4_2.b': db4b,
            'conv5_1.W': dW5a, 'conv5_1.b': db5a,
            'conv5_2.W': dW5b, 'conv5_2.b': db5b,
            'fc6.W': dW6, 'fc6.b': db6,
            'fc7.W': dW7, 'fc7.b': db7,
            'fc8.W': dW8, 'fc8.b': db8,
        }
        return dx, grads
