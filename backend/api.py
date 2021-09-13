import random
from typing import Literal


from fastapi import FastAPI
from pydantic import BaseModel

from models.predictions import predict_load_curve
from models.train_regressor import train_xgbregressor

app = FastAPI()

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
