import json
import logging
import time
from typing import Callable

from redis import Redis

from sigeml.config.config import get_redis_credentials
from sigeml.schemas import TrainingEvent


class TrainingQueue:
    def __init__(self, name: str = "training_queue") -> None:
        self.queue = Redis(**get_redis_credentials())
        self.name = name

    def __len__(self) -> int:
        return self.queue.llen(self.name)

    def add_event(self, event: TrainingEvent) -> None:
        self.queue.rpush(self.name, json.dumps(event.dict()))

    def get_event(self) -> str:
        return self.queue.lpop(self.name)
