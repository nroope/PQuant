# FAQs

## What models formats does PQuantML currently support?
PQuantML primarily supports PyTorch and TensorFlow/Keras models and supports both direct construction and automatic layer replacement using `add_compression_layers(...)`.

## What are requirements to use PQuantML?

The following conditions must be met:

## Can I use MLflow locally?
Yes. 
PQuantML integrates with MLflow for experiment tracking and model logging and local usage is fully supported:
To start local host: 
```python
mlflow ui --host 0.0.0.0 --port 5000
```
Also, PQuantML supports local or remote databases for results storage:
```python
from pquant.core.finetuning import TuningTask
tuner = TuningTask(config)
tuner.set_storage_db("sqlite:///optuna_study.db")
```
