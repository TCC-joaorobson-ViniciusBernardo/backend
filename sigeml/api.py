import logging
import random
from typing import Optional

from fastapi import FastAPI, Request
from fastapi_pagination import Page, add_pagination, paginate
from fastapi.middleware.cors import CORSMiddleware

from sigeml.models.dataset import Dataset
from sigeml.services.predictions import predict_load_curve
from sigeml.models.load_curves.repository import ModelsRepository, ExperimentsRepository
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
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_repository = ModelsRepository()
experiments_repository = ExperimentsRepository()

training_queue = TrainingQueue()


@app.get("/load_curve")
async def get_load_curve(params: LoadCurveParams):
    load_curve = predict_load_curve(params.data)
    return load_curve


@app.post("/train")
def train_model(train_config: TrainConfig, data_processing_config: DataProcessingConfig):
    event = TrainingEvent(train_config=train_config, data_processing_config=data_processing_config)
    training_queue.add_event(event)


@app.get("/experiments", response_model=Page[Experiment])
def get_experiments(request: Request):
    models_names = request.query_params.getlist("modelType[]")
    statuses = request.query_params.getlist("status[]")
    return paginate(
        experiments_repository.get_runs_infos(
            request.query_params.get("experimentName"), models_names, statuses
        )
    )


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


@app.delete("/delete_run/{run_id}")
def delete_run(run_id: str):
    experiments_repository.delete_run(run_id)


add_pagination(app)
