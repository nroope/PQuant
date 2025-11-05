import torch
import torch.nn as nn

from pquant.core.quantizer_functions import create_quantizer


class Quantizer(nn.Module):
    def __init__(self, k, i, f, overflow, round_mode, is_heterogeneous, is_data, hgq_gamma=0):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(k), requires_grad=False)
        self.i = torch.nn.Parameter(torch.tensor(i), requires_grad=False)
        self.f = torch.nn.Parameter(torch.tensor(f), requires_grad=False)
        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.quantizer = create_quantizer(self.k, self.i, self.f, overflow, round_mode, is_heterogeneous, is_data)
        self.is_pretraining = False
        self.hgq_gamma = hgq_gamma

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.quantizer.k, self.quantizer.quantizer.i, self.quantizer.quantizer.f
        else:
            return self.k, self.i, self.f

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.quantizer._i.assign(self.quantizer.quantizer._i * 0.0 + i)
            self.quantizer.quantizer._f.assign(self.quantizer.quantizer._f * 0.0 + f)
        self.i.data = i
        self.f.data = f

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, x):
        if self.use_hgq:
            x = self.quantizer(x)
        else:
            x = self.quantizer(x, k=self.k, i=self.i, f=self.f)
        return x

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = (torch.sum(self.quantizer.quantizer.i) + torch.sum(self.quantizer.quantizer.f)) * self.hgq_gamma
        return loss
