from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseTrainingModel(BaseModel):
    model_config = ConfigDict(extra='allow')
    epochs: int = Field(default=200)
    fine_tuning_epochs: int = Field(default=0)
    pretraining_epochs: int = Field(default=50)
    pruning_first: bool = Field(default=False)
    rewind: str = Field(default="never")
    rounds: int = Field(default=1)
    save_weights_epoch: int = Field(default=-1)
    batch_size: int = Field(default=128)
    optimizer: str = Field(default="sgd")
    plot_frequency: int = Field(default=100)
    label_smoothing: float = Field(default=0.0)
    model: str = Field(default="resnet18")
    dataset: str = Field(default="cifar10")
    l2_decay: float = Field(default=0.001)
    momentum: float = Field(default=0.9)
    lr_schedule: Literal["cosine", "step", "none"] = Field(default="cosine")
    cosine_tmax: int = Field(default=200)
    lr: float = Field(default=0.001)
    prune_ratio: float = Field(default=10.0)
    default_integer_bits: int = Field(default=0)
