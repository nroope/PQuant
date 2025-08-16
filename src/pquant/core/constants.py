import mlflow
import optuna

from pquant.data_models.pruning_model import (
    ActivationPruningModel,
    AutoSparsePruningModel,
    CSPruningModel,
    DSTPruningModel,
    MDMMPruningModel,
    PDPPruningModel,
    WandaPruningModel,
)
from pquant.pruning_methods.constraint_functions import (
    EqualityConstraint,
    GreaterThanOrEqualConstraint,
    LessThanOrEqualConstraint,
)
from pquant.pruning_methods.metric_functions import (
    StructuredSparsityMetric,
    UnstructuredSparsityMetric,
)

PRUNING_MODEL_REGISTRY = {
    "cs": CSPruningModel,
    "dst": DSTPruningModel,
    "pdp": PDPPruningModel,
    "wanda": WandaPruningModel,
    "autosparse": AutoSparsePruningModel,
    "activation_pruning": ActivationPruningModel,
    "mdmm": MDMMPruningModel,
}

SAMPLER_REGISTRY = {
    "GridSampler": optuna.samplers.GridSampler,
    "RandomSampler": optuna.samplers.RandomSampler,
    "TPESampler": optuna.samplers.TPESampler,
    "CmaEsSampler": optuna.samplers.CmaEsSampler,
    "GPSampler": optuna.samplers.GPSampler,
    "PartialFixedSampler": optuna.samplers.PartialFixedSampler,
    "NSGAIISampler": optuna.samplers.NSGAIISampler,
    "NSGAIIISampler": optuna.samplers.NSGAIIISampler,
    "QMCSampler": optuna.samplers.QMCSampler,
    "BruteForceSampler": optuna.samplers.BruteForceSampler,
}

TRACKING_URI = "http://0.0.0.0:5000/"
DB_STORAGE = "sqlite:///optuna_study.db"

LOG_FUNCTIONS_REGISTRY = {
    "torch": mlflow.pytorch.log_model,
    "tensorflow": mlflow.tensorflow.log_model,
}

JAX_BACKEND = "jax"

FINETUNING_DIRECTION = {"maximize", "minimize"}
CONFIG_FILE = "config.yaml"

N_JOBS = 1
TORCH_BACKEND = "torch"
TF_BACKEND = 'tensorflow'


METRIC_REGISTRY = {
    "UnstructuredSparsity": UnstructuredSparsityMetric,
    "StructuredSparsity": StructuredSparsityMetric,
}

CONSTRAINT_REGISTRY = {
    "Equality": EqualityConstraint,
    "LessThanOrEqual": LessThanOrEqualConstraint,
    "GreaterThanOrEqual": GreaterThanOrEqualConstraint,
}
