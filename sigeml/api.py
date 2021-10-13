import logging
import random
from typing import Optional

from fastapi import FastAPI

from sigeml.models.dataset import Dataset
from sigeml.models.model import XGBoostModel
from sigeml.models.predictions import predict_load_curve
from sigeml.models.repository import ModelsRepository, ExperimentsRepository
from sigeml.schemas import DataProcessingConfig, LoadCurveParams, TrainConfig

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("sigeml")

app = FastAPI()

models_repository = ModelsRepository()
experiments_repository = ExperimentsRepository()


@app.get("/load_curve")
async def get_load_curve(params: LoadCurveParams):
    load_curve = predict_load_curve(params.data)
    return load_curve


@app.post("/train")
def train_model(
    train_config: TrainConfig, data_processing_config: DataProcessingConfig
):
    dataset = Dataset(data_processing_config)
    dataset.load_data()
    if train_config.model == "xgboost":
        logger.info(f"Train config: {train_config}")
        xgb = XGBoostModel(train_config, dataset)
        xgb.train()


@app.get("/experiments/")
def get_experiments(experiment_id: Optional[str] = None):
    if experiment_id:
        return experiments_repository.get_runs_infos(experiment_id)
    return experiments_repository.get_experiments()


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
