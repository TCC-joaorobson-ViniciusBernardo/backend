from abc import abstractmethod
import logging

from hampel import hampel
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from sigeml.schemas import TrainConfig
from sigeml.config.config import get_postgres_uri
from sigeml.models.dataset import Dataset
from sigeml.services.sige import get_data_from_sige


class LoadCurveModel:
    def __init__(self, config: TrainConfig, dataset: Dataset) -> None:
        mlflow.set_tracking_uri(get_postgres_uri())
        mlflow.set_experiment(config.experiment_name)

        self.config = config
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.data = dataset.data
        self.X_train = self.X_test = self.y_train = self.y_test = np.array([])
        self.model_params = self.config.model_params
        self.data_params = dataset.config

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

    def evaluate(self, actual, pred) -> float:
        rmse = np.sqrt(mean_squared_error(actual, pred))
        return rmse


class XGBoostModel(LoadCurveModel):
    def run(self):
        with mlflow.start_run():
            xgb = XGBRegressor(**self.model_params.dict()
            ).fit(self.X_train, self.y_train)

            xgb.fit(self.X_train, self.y_train)

            y_pred = xgb.predict(self.X_test)

            rmse = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"XGBRegressor {[str(p[0]) + '=' + str(p[1]) for p in self.model_params]}"
            )
            self.logger.info(f"RMSE: {rmse:.2f}")

            mlflow.log_params({**self.model_params.dict(), **self.data_params.dict()})
            mlflow.log_metric("rmse", rmse)

            if not self.config.is_experiment:
                mlflow.xgboost.log_model(
                    xgb, "model", registered_model_name="XGBRegressor"
                )


class SGDRegressorModel(LoadCurveModel):
    def run(self):
        with mlflow.start_run():
            sgd = SGDRegressor(
                loss=self.params.loss,
                penalty=self.params.penalty,
                alpha=self.params.alpha,
                max_iter=self.params.max_iter,
                tol=self.params.tol,
                eta0=self.params.eta0,
                learning_rate=self.params.learning_rate,
                random_state=self.params.random_state,
            ).fit(self.X_train, self.y_train)

            sgd.fit(self.X_train, self.y_train)

            y_pred = sgd.predict(self.X_test)

            rmse = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"SGDRegressor {[str(p[0]) + '=' + str(p[1]) for p in self.params]}"
            )
            self.logger.info(f"RMSE: {rmse:.2f}")

            mlflow.log_params({**self.model_params.dict(), **self.data_params.dict()})
            mlflow.log_metric("rmse", rmse)

            if not self.config.is_experiment:
                mlflow.sklearn.log_model(
                    sgd, "model", registered_model_name="SGDRegressor"
                )


class LinearSVRModel(LoadCurveModel):
    def run(self):
        with mlflow.start_run():
            svr = LinearSVR(
                epsilon=self.params.epsilon,
                tol=self.params.tol,
                C=self.params.C,
                loss=self.params.loss,
                max_iter=self.params.max_iter,
                random_state=self.params.random_state
            ).fit(self.X_train, self.y_train)

            svr.fit(self.X_train, self.y_train)

            y_pred = svr.predict(self.X_test)

            rmse = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"LinearSVR {[str(p[0]) + '=' + str(p[1]) for p in self.params]}"
            )
            self.logger.info(f"RMSE: {rmse:.2f}")

            mlflow.log_params({**self.model_params.dict(), **self.data_params.dict()})
            mlflow.log_metric("rmse", rmse)

            if not self.config.is_experiment:
                mlflow.sklearn.log_model(
                    svr, "model", registered_model_name="LinearSVR"
                )
