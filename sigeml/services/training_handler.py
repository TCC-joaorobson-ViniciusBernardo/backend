import json
import logging
import time

from sigeml.services.training_queue import TrainingQueue
from sigeml.models.load_curves import XGBoostModel, LinearRegressorModel, SVRModel
from sigeml.models.dataset import Dataset
from sigeml.schemas import TrainingEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training_handler")


MODELS = {
    "xgboost": XGBoostModel,
    "linearregressor": LinearRegressorModel,
    "svr": SVRModel,
}


class TrainingHandler:
    def __init__(self, queue: TrainingQueue):
        self.queue = queue

    def loop(self) -> None:
        while True:
            if self.queue:
                self.__handle_new_training_event()
            else:
                time.sleep(3)

    def __handle_new_training_event(self) -> None:
        event = TrainingEvent(**json.loads(self.queue.get_event()))
        logger.info(f"Processing event: {event}")

        dataset = Dataset(
            event.data_processing_config,
            data_path="/app/sigeml/models/quarterly_measurements_CPD1.csv",
        )
        dataset.load_data()

        model = MODELS[event.train_config.model_params.model](event.train_config, dataset)
        model.train()
