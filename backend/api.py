import logging
import random
from typing import Literal, Optional, Set, Union


from fastapi import FastAPI
from pydantic import BaseModel, StrictInt, StrictFloat, validator

from models.predictions import predict_load_curve
from models.train_regressor import train_xgbregressor
from models.repository import ModelsRepository

app = FastAPI()

models_repository = ModelsRepository()


logger = logging.getLogger("api")
logger.setLevel(logging.INFO)



class LoadCurveParams(BaseModel):
    building: str
    data: list



class XGBoostParams(BaseModel):
    n_estimators: StrictInt = 100
    max_depth: Optional[StrictInt] = None
    learning_rate: Optional[StrictFloat] = None
    gamma: Optional[StrictFloat] = None
    random_state: Optional[StrictInt] = None


class TrainConfig(BaseModel):
    model: Literal["xgboost"]
    test_size: StrictFloat = 0.2
    remove_outliers = StrictBool = True
    hyperparams: Optional[Union[XGBoostParams]] = None
    metrics: Set[str] = {"rmse"}

    @validator("metrics")
    def check_if_metrics_are_valid(cls, metrics):
        invalid_metrics = metrics.difference({"rmse", "mse", "r2"})
        if invalid_metrics:
            raise ValueError(f"Not supported metrics used: {invalid_metrics}")


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
        train_xgbregressor() 


@app.get("/models")
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
