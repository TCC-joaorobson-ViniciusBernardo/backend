from keras.models import load_model
import numpy as np


class TimeSeriesModel:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path)

    def predict(self):
        pass

    def retrain(self):
        pass


class LSTMModel(TimeSeriesModel):
    def retrain(self, X: np.array, y: np.array) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)
