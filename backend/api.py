import logging
import random
from typing import Literal, Optional


from fastapi import FastAPI
from pydantic import BaseModel

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

class TrainParams(BaseModel):
    model: Literal["xgboost"]


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
async def train_model(params: TrainParams):
    if params.model == "xgboost":
        train_xgbregressor() 


@app.get("/models/")
def get_models(model_name: Optional[str] = None):
    if not model_name:
        return models_repository.get_registered_models()

    else:
        return models_repository.get_model_versions(model_name)


@app.get("/model_metrics/{run_id}")
def get_model_metrics(run_id: str):
    logger.info("oooooooooooi")
    return models_repository.retrieve_model_metrics(run_id)


@app.delete("/delete_model/")
def delete_model(model_name: str, version: Optional[int] = None):
    if version:
        models_repository.delete_model_version(model_name, version)
    else:
        models_repository.delete_model(model_name)
