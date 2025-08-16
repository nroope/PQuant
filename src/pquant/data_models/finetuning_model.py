from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class HyperparameterSearch(BaseModel):
    numerical: Dict[str, List[Union[int, float]]] = Field(default_factory=dict)
    categorical: Optional[Dict[str, List[str]]] = Field(default_factory=dict)


class BaseFinetuningModel(BaseModel):
    experiment_name: str = Field(default="experiment_1")
    sampler: str = Field(default="TPESampler")
    num_trials: int = Field(default=0)
    hyperparameter_search: HyperparameterSearch = Field(default_factory=HyperparameterSearch)
