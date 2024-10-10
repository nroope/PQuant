from keras_core import ops
import numpy as np
pi = ops.convert_to_tensor(np.pi)
L0 = ops.convert_to_tensor(-6.0)
L1 = ops.convert_to_tensor(6.0)

def cosine_decay(i, T):
    return (1 + ops.cos(pi * i / T)) / 2

def sigmoid_decay(i, T):
    return 1 - ops.sigmoid(L0 + (L1 - L0) * i / T)

def cosine_sigmoid_decay(i, T):
    return ops.maximum(cosine_decay(i, T), sigmoid_decay(i, T))