from typing import Literal, Optional, Set, Union

from pydantic import BaseModel, StrictBool, StrictInt, StrictFloat, StrictStr, validator


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
    remove_outliers: StrictBool = True
    model_params: Optional[Union[XGBoostParams]] = None
    is_experiment: StrictBool = True
    experiment_name: StrictStr = ""
