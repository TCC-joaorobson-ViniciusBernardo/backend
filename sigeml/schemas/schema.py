from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, StrictBool, StrictInt, StrictFloat, StrictStr, validator


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


class StatusEnum(Enum):
    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class ExperimentInfo(BaseModel):
    artifact_uri: StrictStr
    end_time: Optional[StrictInt]
    experiment_id: StrictStr
    lifecycle_stage: StrictStr
    run_id: StrictStr
    run_uuid: StrictStr
    start_time: Optional[StrictInt]
    status: StatusEnum
    user_id: StrictStr


class ExperimentTags(BaseModel):
    model_name: StrictStr


class ExperimentData(BaseModel):
    metrics: dict
    params: dict
    tags: ExperimentTags


class Experiment(BaseModel):
    info: ExperimentInfo
    data: ExperimentData
    experiment_name: StrictStr
    has_registered_model: StrictBool
