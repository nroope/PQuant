import copy
import json
import logging
from typing import Annotated, Callable, Dict, Optional, Union

import keras
import os
import mlflow
import optuna
import torch
import yaml
from mlflow.models import infer_signature
from pydantic import BaseModel, Field, field_validator

from pquant.core import constants
from pquant.core.compressed_layers import add_compression_layers
from pquant.core.train import iterative_train
from pquant.data_models.finetuning_model import BaseFinetuningModel
from pquant.data_models.pruning_model import (
    ActivationPruningModel,
    AutoSparsePruningModel,
    BasePruningModel,
    CSPruningModel,
    DSTPruningModel,
    MDMMPruningModel,
    PDPPruningModel,
    WandaPruningModel,
)
from pquant.data_models.quantization_model import BaseQuantizationModel
from pquant.data_models.training_model import BaseTrainingModel


def get_sampler(sampler_type, **kwargs):
    try:
        return constants.SAMPLER_REGISTRY[sampler_type](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def log_model_by_backend(model, name, signature=None, registered_model_name=None):
    backend = keras.backend.backend()

    kwargs = {
        "artifact_path": name,
        "signature": signature,
        "registered_model_name": registered_model_name,
    }

    if backend == constants.JAX_BACKEND:
        raise NotImplementedError("JAX is not supported yet.")

    if backend not in constants.LOG_FUNCTIONS_REGISTRY:
        raise ValueError(f"Unsupported backend: {backend}")

    return constants.LOG_FUNCTIONS_REGISTRY[backend](model, **kwargs)


class MetricFunction(BaseModel):
    function_name: Callable
    direction: str

    @field_validator('direction')
    def validate_direction(cls, direction):
        if direction not in constants.FINETUNING_DIRECTION:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        return direction


class TuningConfig(BaseModel):
    finetuning_parameters: BaseFinetuningModel
    pruning_parameters: Annotated[
        Union[
            CSPruningModel,
            DSTPruningModel,
            PDPPruningModel,
            WandaPruningModel,
            AutoSparsePruningModel,
            ActivationPruningModel,
            MDMMPruningModel,
        ],
        Field(discriminator="pruning_method"),
    ]
    quantization_parameters: BaseQuantizationModel
    training_parameters: BaseTrainingModel

    @classmethod
    def load_from_file(cls, path_to_config_file):
        if path_to_config_file.endswith(('.yaml', '.yml')):
            with open(path_to_config_file) as f:
                config_data = yaml.safe_load(f)
        elif path_to_config_file.endswith('.json'):
            with open(path_to_config_file) as f:
                config_data = json.load(f)
        else:
            raise ValueError("Unsupported file type. Use .yaml, .yml, or .json")

        return cls.load_from_config(config_data)

    @classmethod
    def load_from_config(cls, config):
        pruning_section = config.get("pruning_parameters", {})
        pruning_method = pruning_section.get("pruning_method", "cs")
        pruning_model_cls = constants.PRUNING_MODEL_REGISTRY.get(pruning_method, BasePruningModel)

        return cls(
            finetuning_parameters=BaseFinetuningModel(**config.get("finetuning_parameters", {})),
            pruning_parameters=pruning_model_cls(**config.get("pruning_parameters", {})),
            quantization_parameters=BaseQuantizationModel(**config.get("quantization_parameters", {})),
            training_parameters=BaseTrainingModel(**config.get("training_parameters", {})),
        )

    def get_dict(self):
        return self.model_dump()


class TuningTask:
    def __init__(self, config: TuningConfig):
        self.config = config
        self.hyperparameters = {}
        self.objectives: Dict[str, MetricFunction] = {}
        self._training_function: Optional[Callable] = None
        self._validation_function: Optional[Callable] = None
        self._optimizer_function: Optional[Callable] = None
        self._scheduler_function: Optional[Callable] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_mlflow = False
        self.tracking_uri = None
        self.storage_db = None
    
    def set_tracking_uri(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    
    def set_user(self, user_email: str, access_token: str):
        os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user_email
        os.environ["MLFLOW_TRACKING_PASSWORD"] = access_token
        os.environ["NO_PROXY"] = "ngt.cern.ch"
    
    def set_storage_db(self, storage_db: str):
        self.storage_db = storage_db
    
    def set_enable_mlflow(self):
        self.enable_mlflow = True

    def get_dict(self):
        return self.config.model_dump()

    def set_objective_function(self, name: str, fn: Callable, direction: str):
        if not callable(fn):
            raise TypeError("Objective function must be callable.")
        self.objectives[name] = MetricFunction(function_name=fn, direction=direction)

    def set_training_function(self, fn: Callable):
        if not callable(fn):
            raise TypeError("Training function must be callable.")
        self._training_function = fn

    def set_validation_function(self, fn: Callable):
        if not callable(fn):
            raise TypeError("Validation function must be callable.")
        self._validation_function = fn

    def set_optimizer_function(self, fn: Callable):
        if not callable(fn):
            raise TypeError("Optimizer function must be callable.")
        self._optimizer_function = fn

    def set_scheduler_function(self, fn: Callable):
        if not callable(fn):
            raise TypeError("Scheduler function must be callable.")
        self._scheduler_function = fn

    def get_training_function(self) -> Callable:
        if not self._training_function:
            raise ValueError("Training function is not set.")
        return self._training_function

    def get_validation_function(self) -> Callable:
        if not self._validation_function:
            raise ValueError("Validation function is not set.")
        return self._validation_function

    def get_optimizer_function(self) -> Callable:
        if not self._optimizer_function:
            raise ValueError("Optimizer function is not set.")
        return self._optimizer_function

    def get_scheduler_function(self) -> Callable:
        if not self._scheduler_function:
            raise ValueError("Scheduler function is not set.")
        return self._scheduler_function

    def set_hyperparameters(self):
        hp_config = self.config.finetuning_parameters.hyperparameter_search
        numerical_params = hp_config.numerical
        categorical_params = hp_config.categorical

        if numerical_params:
            self.set_numerical_params(numerical_params)
        if categorical_params:
            self.set_categorical_params(categorical_params)

    def set_numerical_params(self, numerical_params):
        try:
            for param, value in numerical_params.items():
                if not isinstance(value, list) or len(value) < 2:
                    continue
                start_value, end_value = value[0], value[1]
                step = value[2] if len(value) == 3 else None
                use_float = any(isinstance(x, float) for x in (start_value, end_value, step) if x is not None)
                suggest_func = optuna.trial.Trial.suggest_float if use_float else optuna.trial.Trial.suggest_int

                if use_float:
                    start_value, end_value = float(start_value), float(end_value)
                    if step is not None:
                        step = float(step)
                else:
                    start_value, end_value = int(start_value), int(end_value)
                    if step is not None:
                        step = int(step)

                if step is None:
                    self.register_hyperparameter(param, suggest_func, param, start_value, end_value)
                else:
                    self.register_hyperparameter(param, suggest_func, param, start_value, end_value, step=step)

        except Exception as e:
            logging.error(f"Failed to register numerical hyperparameter '{param}': {e}")

    def set_categorical_params(self, categorical_params):
        for param, choices in categorical_params.items():
            if not isinstance(choices, list) or len(choices) == 0:
                continue
            try:
                self.register_hyperparameter(param, optuna.trial.Trial.suggest_categorical, param, choices)
            except Exception as e:
                logging.error(f"Failed to register categorical hyperparameter '{param}': {e}")

    def register_hyperparameter(self, name, optuna_func, *args, **kwargs):
        self.hyperparameters[name] = (optuna_func, args, kwargs)

    def objective(self, trial, model, train_func, valid_func, **kwargs):
        for param_name, (optuna_func, func_args, func_kwargs) in self.hyperparameters.items():
            new_value = optuna_func(trial, *func_args, **func_kwargs)
            logging.info(f"Suggested {param_name} = {new_value}")

            applied = False
            for sub_config in [self.config.training_parameters, self.config.finetuning_parameters]:
                if hasattr(sub_config, param_name):
                    setattr(sub_config, param_name, new_value)
                    applied = True
                    break
            if not applied:
                logging.error(f"'{param_name}' not found in config: value not applied.")

        trainloader = kwargs['trainloader']
        raw_input_batch = next(iter(trainloader))
        sample_input = raw_input_batch[0]
        sample_output = model(sample_input.to(next(model.parameters()).device))

        input_shape = sample_input.shape
        compressed_model = add_compression_layers(model, self.config, input_shape)
        optimizer_func = self.get_optimizer_function()
        optimizer = optimizer_func(self.config, compressed_model)
        scheduler_func = self.get_scheduler_function()
        scheduler = scheduler_func(optimizer, self.config)

        trained_model = iterative_train(
            compressed_model,
            self.config,
            train_func,
            valid_func,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            writer=None,
            **kwargs,
        )
        trained_model.eval()
        objectives = [
            metric_object.function_name(trained_model, device=self.device, **kwargs)
            for _, metric_object in self.objectives.items()
        ]

        if self.enable_mlflow:
            with mlflow.start_run(nested=True):
                mlflow.log_params({param_name: getattr(self.config, param_name) for param_name in self.config.model_fields})
                mlflow.log_metrics({key: val for key, val in zip(self.objectives.keys(), objectives)})
                signature = infer_signature(sample_input.cpu().numpy(), sample_output.detach().cpu().numpy())

                mlflow.log_text(yaml.safe_dump(self.get_dict()), "config.yaml")
                log_model_by_backend(
                    model=trained_model,
                    name=self.config.training_parameters.model,
                    signature=signature,
                    registered_model_name=self.config.training_parameters.model,
                )

        return objectives if len(objectives) > 1 else objectives[0]

    def run_optimization(self, model, **kwargs):
        if self.enable_mlflow:
            if not self.tracking_uri:
                raise ValueError("Tracking URI must be set when MLflow logging is enabled.")
            mlflow.set_tracking_uri(self.tracking_uri)
            finetuning_parameters = self.config.finetuning_parameters
            mlflow.set_experiment(finetuning_parameters.experiment_name)
          
        sampler = get_sampler(finetuning_parameters.sampler.type, **finetuning_parameters.sampler.params)
        study = optuna.create_study(
            study_name=finetuning_parameters.experiment_name,
            storage=self.storage_db,
            sampler=sampler,
            load_if_exists=True,
            directions=[metric_object.direction for _, metric_object in self.objectives.items()],
        )

        num_trials = finetuning_parameters.num_trials
        study.optimize(
            lambda trial: self.objective(
                trial, copy.deepcopy(model.cpu()).to(self.device), self.get_training_function(), self.get_validation_function(), **kwargs
            ),
            n_trials=num_trials,
            n_jobs=1,
        )

        return study.best_params
