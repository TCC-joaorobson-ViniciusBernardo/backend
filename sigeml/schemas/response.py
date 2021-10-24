from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, StrictBool, StrictInt, StrictFloat, StrictStr


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


class ExperimentData(BaseModel):
    metrics: dict
    params: dict
    tags: dict


class Experiment(BaseModel):
    info: ExperimentInfo
    data: ExperimentData
    experiment_name: StrictStr
    has_registered_model: StrictBool
