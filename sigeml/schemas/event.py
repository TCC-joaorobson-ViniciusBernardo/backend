import time

from pydantic import BaseModel, StrictInt

from sigeml.schemas import DataProcessingConfig, TrainConfig


class TrainingEvent(BaseModel):
    train_config: TrainConfig
    data_processing_config: DataProcessingConfig
    event_time: StrictInt = int(time.time())
