from abc import abstractmethod
import logging

from hampel import hampel
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from sigeml.schemas import TrainConfig
from sigeml.config.config import get_postgres_uri
from sigeml.models.dataset import Dataset
from sigeml.services.sige import get_data_from_sige


class LoadCurveModel:
    def __init__(self, config: TrainConfig, dataset: Dataset) -> None:
        mlflow.set_tracking_uri(get_postgres_uri())

        if config.is_experiment:
            mlflow.set_experiment(config.experiment_name)

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.data = dataset.data
        self.X_train = self.X_test = self.y_train = self.y_test = np.array([])

    def get_X(self):
        X = (
            self.data.collection_date.dt.hour + self.data.collection_date.dt.minute / 60
        ).values.reshape((len(self.data), 1))
        return X

    def get_y(self):
        return self.data["consumption"].values

    def split_data(self):
        X = self.get_X()
        y = self.get_y()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=42, test_size=self.config.test_size
        )

    def train(self):
        self.split_data()
        self.run()

    @abstractmethod
    def run(self):
        pass

    def evaluate(self, actual, pred) -> None:
        pass


class XGBoostModel(LoadCurveModel):
    def __init__(self, config: TrainConfig, dataset: Dataset):
        super().__init__(config, dataset)
        self.params = self.config.model_params

    def run(self):
        with mlflow.start_run():
            xgb = XGBRegressor(
                n_estimators=self.params.n_estimators,
                max_depth=self.params.max_depth,
                gamma=self.params.gamma,
                learning_rate=self.params.learning_rate,
                random_state=self.params.random_state,
            ).fit(self.X_train, self.y_train)

            xgb.fit(self.X_train, self.y_train)

            y_pred = xgb.predict(self.X_test)

            rmse = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"XGB model (n_estimators={self.params.n_estimators}, "
                f"max_depth={self.params.max_depth}, learning_rate={self.params.learning_rate}, "
                f"gamma={self.params.gamma}, random_state={self.params.random_state})"
            )
            self.logger.info(f"RMSE: {rmse:.2f}")

            mlflow.log_param("n_estimators", self.params.n_estimators)
            mlflow.log_param("max_depth", self.params.max_depth)
            mlflow.log_param("learning_rate", self.params.learning_rate)
            mlflow.log_param("gamma", self.params.gamma)
            mlflow.log_param("random_state", self.params.random_state)
            mlflow.log_metric("rmse", rmse)

            if not self.config.is_experiment:
                mlflow.xgboost.log_model(
                    xgb, "model", registered_model_name="XGBRegressor"
                )

    def evaluate(self, actual, pred) -> None:
        rmse = np.sqrt(mean_squared_error(actual, pred))
        return rmse
