from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictFloat, StrictStr, validator


class LoadCurveParams(BaseModel):
    building: str
    data: list


class SIGEQueryParams(BaseModel):
    id: StrictInt
    start_date: StrictInt
    end_date: StrictInt
    type: StrictStr


class DataProcessingConfig(BaseModel):
    remove_outliers: StrictBool = True
    query_params: SIGEQueryParams


class LinearSVRParams(BaseModel):
    model: Literal["linearsvr"]
    epsilon: StrictFloat = 0.0
    tol: StrictFloat = 1e-4
    C: StrictFloat = 1.0
    loss: Literal["epsilon_insensitive", "squared_epsilon_insensitive"] = "epsilon_insensitive"
    max_iter: StrictInt = 1000
    random_state: Optional[StrictInt] = None


class LinearRegressorParams(BaseModel):
    model: Literal["linearregressor"]
    normalize: StrictBool = False
    positive: StrictBool = False
    fit_intercept: StrictBool = False


class XGBoostParams(BaseModel):
    model: Literal["xgboost"]
    n_estimators: StrictInt = 100
    max_depth: Optional[StrictInt] = None
    learning_rate: Optional[StrictFloat] = None
    gamma: Optional[StrictFloat] = None
    random_state: Optional[StrictInt] = None


class TrainConfig(BaseModel):
    model_params: Union[LinearSVRParams, LinearRegressorParams, XGBoostParams] = Field(
        discriminator="model"
    )
    test_size: StrictFloat = 0.2
    is_experiment: StrictBool = True
    experiment_name: StrictStr = "Default"
