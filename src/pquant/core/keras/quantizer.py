import keras
from keras.initializers import Constant

from pquant.core.quantizer_functions import create_quantizer


class Quantizer(keras.layers.Layer):
    # HGQ quantizer wrapper
    def __init__(self, k, i, f, overflow, round_mode, is_heterogeneous, is_data, hgq_gamma=0):
        super().__init__()
        self.k = k
        self.i = i
        self.f = f
        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.quantizer = create_quantizer(self.k, self.i, self.f, overflow, round_mode, is_heterogeneous, is_data)
        self.is_pretraining = False
        self.hgq_gamma = hgq_gamma

    def build(self, input_shape):
        super().build(input_shape)
        self.i = self.add_variable((), Constant(self.i), dtype="float32", trainable=False)
        self.f = self.add_variable((), Constant(self.f), dtype="float32", trainable=False)
        if self.use_hgq:
            self.quantizer.build(input_shape)

    def get_total_bits(self, shape):
        if self.use_hgq:
            return self.quantizer.bits_(shape)
        else:
            b = self.i + self.f + self.k
            return keras.ops.ones(shape) * b

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.quantizer.k, self.quantizer.quantizer.i, self.quantizer.quantizer.f
        else:
            return self.k, self.i, self.f

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.quantizer._i.assign(self.quantizer.quantizer._i * 0.0 + i)
            self.quantizer.quantizer._f.assign(self.quantizer.quantizer._f * 0.0 + f)
        self.i = i
        self.f = f

    def post_pretrain(self):
        self.is_pretraining = True

    def call(self, x, training=None):
        if not self.built:
            self.build(x.shape)
        if self.use_hgq:
            x = self.quantizer(x, training=training)
        else:
            x = self.quantizer(x, k=self.k, i=self.i, f=self.f, training=training)
        return x

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = (keras.ops.sum(self.quantizer.quantizer.i) + keras.ops.sum(self.quantizer.quantizer.f)) * self.hgq_gamma
        return loss
