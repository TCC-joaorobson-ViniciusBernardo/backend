from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, StrictBool, StrictInt, StrictFloat, StrictStr, validator


class LoadCurveParams(BaseModel):
    building: str
    data: list


class SIGEQueryParams(BaseModel):
    id: StrictInt
    start_date: datetime
    end_date: datetime
    type: StrictStr


class DataProcessingConfig(BaseModel):
    remove_outliers: StrictBool = True
    query_params: SIGEQueryParams


class XGBoostParams(BaseModel):
    n_estimators: StrictInt = 100
    max_depth: Optional[StrictInt] = None
    learning_rate: Optional[StrictFloat] = None
    gamma: Optional[StrictFloat] = None
    random_state: Optional[StrictInt] = None


class TrainConfig(BaseModel):
    model: Literal["xgboost"]
    test_size: StrictFloat = 0.2
    model_params: Union[XGBoostParams] = XGBoostParams()
    is_experiment: StrictBool = True
    experiment_name: StrictStr = "Default"
