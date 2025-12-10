from typing import List

from pydantic import BaseModel, Field


class BaseQuantizationModel(BaseModel):
    default_weight_keep_negatives: float = Field(default=1.0)
    default_weight_integer_bits: float = Field(default=0.0)
    default_weight_fractional_bits: float = Field(default=7.0)
    default_data_keep_negatives: float = Field(default=0.0)
    default_data_integer_bits: float = Field(default=0.0)
    default_data_fractional_bits: float = Field(default=7.0)
    quantize_input: bool = Field(default=True)
    quantize_output: bool = Field(default=False)
    enable_quantization: bool = Field(default=True)
    hgq_gamma: float = Field(default=0.0003)
    hgq_beta: float = Field(default=1e-5)
    hgq_heterogeneous: bool = Field(default=True)
    layer_specific: List = Field(default_factory=list)
    use_high_granularity_quantization: bool = Field(default=False)
    use_real_tanh: bool = Field(default=False)
    overflow: str = Field(default="SAT")
    round_mode: str = Field(default="RND")
    use_relu_multiplier: bool = Field(default=True)
