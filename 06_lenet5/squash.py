"""
The squashing function used throughout LeNet-5 (Eq. 5-6, Appendix A):
f(a) = A tanh(Sa), A=1.7159, S=2/3, chosen so f(1)=1, f(-1)=-1.
"""
import numpy as np

A = 1.7159
S = 2.0 / 3.0


def squash(a):
    return A * np.tanh(S * a)


def dsquash(a):
    t = np.tanh(S * a)
    return A * S * (1 - t ** 2)
