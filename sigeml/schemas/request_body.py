from time import time

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, validator


class LoadCurveParams(BaseModel):
    building: str
    data: list


class SIGEQueryParams(BaseModel):
    id: StrictInt
    start_date: StrictInt = int(time())
    end_date: StrictInt = int(time()) - 3600
    type: StrictStr


class DataProcessingConfig(BaseModel):
    remove_outliers: StrictBool = True
    query_params: SIGEQueryParams


class SVRParams(BaseModel):
    model: Literal["svr"]
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf"
    epsilon: float = 0.0
    gamma: Union[Literal["scale", "auto"], float] = "scale"
    tol: float = 1e-4
    C: float = 1.0
    max_iter: StrictInt = -1


class LinearRegressorParams(BaseModel):
    model: Literal["linearregressor"]
    positive: StrictBool = False
    fit_intercept: StrictBool = False


class XGBoostParams(BaseModel):
    model: Literal["xgboost"]
    n_estimators: StrictInt = 100
    max_depth: Optional[StrictInt] = 5
    learning_rate: Optional[float] = 0.3
    gamma: Optional[float] = 0
    random_state: Optional[StrictInt] = 0


class TrainConfig(BaseModel):
    model_params: Union[SVRParams, LinearRegressorParams, XGBoostParams]
    test_size: float = 0.2
    is_experiment: StrictBool = True
    experiment_name: StrictStr = "Default"
