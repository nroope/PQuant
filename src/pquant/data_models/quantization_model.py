from typing import List

from pydantic import BaseModel, Field


class BaseQuantizationModel(BaseModel):
    default_integer_bits: float = Field(default=0.0)
    default_fractional_bits: float = Field(default=7.0)
    enable_quantization: bool = Field(default=True)
    hgq_gamma: float = Field(default=0.0003)
    hgq_heterogeneous: bool = Field(default=True)
    layer_specific: List = Field(default_factory=list)
    use_high_granularity_quantization: bool = Field(default=False)
    use_real_tanh: bool = Field(default=False)
    use_symmetric_quantization: bool = Field(default=False)
    use_relu_multiplier: bool = Field(default=True)
