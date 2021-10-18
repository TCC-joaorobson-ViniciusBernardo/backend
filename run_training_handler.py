import logging

from sigeml.services.training_queue import TrainingQueue
from sigeml.services.training_handler import TrainingHandler


if __name__ == "__main__":
    training_queue = TrainingQueue()
    training_handler = TrainingHandler(training_queue)
    training_handler.loop()
