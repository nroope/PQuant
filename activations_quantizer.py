import torch
import torch.nn as nn


def round_ste(x):
  return x + (-x + torch.round(x)).detach()
  
def quantized_tanh(x, bits=8.):
  non_sign_bits = bits - 1.0
  m = torch.pow(torch.tensor(2.0), non_sign_bits).to(x.device)
  p = torch.tanh(x)
  round_x = round_ste(p * m) / m
  x_clipped = torch.clip(round_x, -1.0, 1.0 - 1.0 / m)
  return x_clipped

class QuantizedTanh(nn.Module):
    def __init__(self, bits=8.):
        super(QuantizedTanh, self).__init__()
        self.bits = bits
  
    def forward(self, x):
        return quantized_tanh(x, self.bits)

    
def quantized_relu(x, bits, integer_bits):
    m = torch.tensor(2.) ** bits
    m_i = torch.tensor(2.) ** integer_bits
    m_f = torch.tensor(2.) ** (integer_bits - bits)
    x_u = torch.clip(x, 0, m_i - m_f)
    p = x * m / m_i
    xq = m_i * torch.clip(round_ste(p) / m, 0.0, 1.0 - 1.0 / m)
    return x_u + (-x_u + xq).detach()

class QuantizedReLU(nn.Module):
    def __init__(self, bits=8., integer_bits=0.):
        super(QuantizedReLU, self).__init__()
        self.bits = torch.tensor(bits) # Clip to 0, non sign bits == bits
        self.integer_bits = torch.tensor(integer_bits)
  
    def forward(self, x):
      return quantized_relu(x, self.bits, self.integer_bits)


    

def hard_sigmoid(x):
  """Computes hard_sigmoid function that saturates between 0 and 1."""
  return torch.clip(0.5 * x + 0.5, 0.0, 1.0)

def hard_tanh(x):
  """Computes hard_tanh function that saturates between -1 and 1."""
  return 2.0 * hard_sigmoid(x) - 1.0