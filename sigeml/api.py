import logging
import random
from typing import Optional, Set, Union
from typing_extensions import Literal


from fastapi import FastAPI
from pydantic import BaseModel, StrictInt, StrictFloat, validator

from sigeml.models.model import XGBoostModel
from sigeml.models.predictions import predict_load_curve
from sigeml.models.repository import ModelsRepository, ExperimentsRepository
from sigeml.schemas import LoadCurveParams, TrainConfig

app = FastAPI()

models_repository = ModelsRepository()
experiments_repository = ExperimentsRepository()

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)


@app.get("/")
def root():
    return {"Hello": "World"}


@app.get("/sige")
def sige():
    return {"predio_1": random.randint(1000, 1200)}


@app.get("/load_curve")
async def get_load_curve(params: LoadCurveParams):
    load_curve = predict_load_curve(params.data)
    return load_curve


@app.post("/train")
async def train_model(config: TrainConfig):
    if config.model == "xgboost":
        xgb = XGBoostModel(config)
        xgb.train()


@app.get("/experiments/")
def get_experiments(experiment_id: Optional[str] = None):
    if experiment_id:
        return experiments_repository.get_runs_infos(experiment_id)
    return experiments_repository.get_experiments()


@app.get("/models/")
def get_models(model_name: Optional[str] = None):
    return models_repository.get_models_versions(model_name)


@app.get("/model_metrics/{run_id}")
def get_model_metrics(run_id: str):
    return models_repository.retrieve_model_metrics(run_id)


@app.delete("/delete_model/")
def delete_model(model_name: str, version: Optional[int] = None):
    if version:
        models_repository.delete_model_version(model_name, version)
    else:
        models_repository.delete_model(model_name)
