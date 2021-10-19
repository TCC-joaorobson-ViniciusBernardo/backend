import logging
import random
from typing import Optional

from fastapi import FastAPI
from fastapi_pagination import Page, add_pagination, paginate


from sigeml.models.dataset import Dataset
from sigeml.models.model import XGBoostModel
from sigeml.models.predictions import predict_load_curve
from sigeml.models.repository import ModelsRepository, ExperimentsRepository
from sigeml.schemas import (
    DataProcessingConfig,
    Experiment,
    LoadCurveParams,
    TrainConfig,
    TrainingEvent,
)
from sigeml.services.training_queue import TrainingQueue

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("sigeml")

app = FastAPI()

models_repository = ModelsRepository()
experiments_repository = ExperimentsRepository()

training_queue = TrainingQueue()


@app.get("/load_curve")
async def get_load_curve(params: LoadCurveParams):
    load_curve = predict_load_curve(params.data)
    return load_curve


@app.post("/train")
def train_model(
    train_config: TrainConfig, data_processing_config: DataProcessingConfig
):
    event = TrainingEvent(
        train_config=train_config, data_processing_config=data_processing_config
    )
    training_queue.add_event(event)


@app.get("/experiments", response_model=Page[Experiment])
def get_experiments():
    return paginate(experiments_repository.get_runs_infos())


@app.get("/models")
def get_models():
    return models_repository.get_models_versions()


@app.get("/model_metrics/{run_id}")
def get_model_metrics(run_id: str):
    return models_repository.retrieve_model_metrics(run_id)


@app.delete("/delete_model/")
def delete_model(model_name: str, version: Optional[int] = None):
    if version:
        models_repository.delete_model_version(model_name, version)
    else:
        models_repository.delete_model(model_name)

add_pagination(app)
