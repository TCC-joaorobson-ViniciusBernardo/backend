from abc import abstractmethod
import logging

from hampel import hampel
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sigeml.schemas import TrainConfig
from sigeml.config.config import get_postgres_uri
from sigeml.models.dataset import Dataset
from sigeml.services.sige import get_data_from_sige
from sigeml.services.predictions import store_experiment_predictions


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
        delattr(self.model_params, "model")

        self.data_params = dataset.config
        self.test_sample_size = 200

    def get_X(self):
        X = (
            self.data.collection_date.dt.hour + self.data.collection_date.dt.minute / 60
        ).values.reshape((len(self.data), 1))
        return X

    def get_y(self):
        return self.data["consumption"].values

    def get_sample_from_test_data(self):
        test_data = np.column_stack([self.X_test, self.y_test])
        np.random.seed(42)
        return test_data[np.random.choice(test_data.shape[0], self.test_sample_size, replace=False)]

    def get_test_points(self):
        test_data_sample = self.get_sample_from_test_data()
        return list(map(lambda d: {"x": float(d[0]), "y": float(d[1])}, test_data_sample))

    @staticmethod
    def get_load_curve(y_pred):
        preds = zip(np.arange(0, 24, 0.25), y_pred)
        return list(map(lambda d: {"x": float(d[0]), "y": float(d[1])}, preds))

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
        mae = mean_absolute_error(actual, pred)
        return mae


class XGBoostModel(LoadCurveModel):
    def run(self):
        with mlflow.start_run():
            xgb = XGBRegressor(**self.model_params.dict()).fit(self.X_train, self.y_train)

            xgb.fit(self.X_train, self.y_train)

            y_pred = xgb.predict(self.X_test)

            mae = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"XGBRegressor {[str(p[0]) + '=' + str(p[1]) for p in self.model_params]}"
            )
            self.logger.info(f"MAE: {mae:.2f}")

            params = {
                **self.model_params.dict(),
                **self.data_params.dict(),
                **{"model_name": "XGBRegressor"},
            }
            mlflow.log_params(params)
            mlflow.log_metric("mae", mae)

            if not self.config.is_experiment:
                mlflow.xgboost.log_model(xgb, "model", registered_model_name="XGBRegressor")

            run = mlflow.active_run()
            load_curve = self.get_load_curve(xgb.predict(np.arange(0, 23.25, 0.25).reshape(-1, 1)))
            store_experiment_predictions(run.info.run_id, self.get_test_points(), load_curve)


class LinearRegressorModel(LoadCurveModel):
    def run(self):
        with mlflow.start_run():
            reg = LinearRegression(**self.model_params.dict()).fit(self.X_train, self.y_train)

            reg.fit(self.X_train, self.y_train)

            y_pred = reg.predict(self.X_test)

            load_curve = self.get_load_curve(reg.predict(np.arange(0, 23.25, 0.25).reshape(-1, 1)))

            mae = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"LinearRegressor {[str(p[0]) + '=' + str(p[1]) for p in self.model_params]}"
            )
            self.logger.info(f"MAE: {mae:.2f}")

            params = {
                **self.model_params.dict(),
                **self.data_params.dict(),
                **{"model_name": "LinearRegressor"},
            }
            mlflow.log_params(params)

            mlflow.log_metric("mae", mae)

            if not self.config.is_experiment:
                mlflow.sklearn.log_model(reg, "model", registered_model_name="LinearRegressor")

            run = mlflow.active_run()
            load_curve = self.get_load_curve(reg.predict(np.arange(0, 23.25, 0.25).reshape(-1, 1)))
            store_experiment_predictions(run.info.run_id, self.get_test_points(), load_curve)


class SVRModel(LoadCurveModel):
    def run(self):
        with mlflow.start_run():
            svr = SVR(**self.model_params.dict()).fit(self.X_train, self.y_train)

            svr.fit(self.X_train, self.y_train)

            y_pred = svr.predict(self.X_test)

            mae = self.evaluate(self.y_test, y_pred)

            self.logger.info(f"SVR {[str(p[0]) + '=' + str(p[1]) for p in self.model_params]}")
            self.logger.info(f"MAE: {mae:.2f}")

            params = {
                **self.model_params.dict(),
                **self.data_params.dict(),
                **{"model_name": "SVR"},
            }
            mlflow.log_params(params)

            mlflow.log_metric("mae", mae)

            if not self.config.is_experiment:
                mlflow.sklearn.log_model(svr, "model", registered_model_name="SVR")

            run = mlflow.active_run()
            load_curve = self.get_load_curve(svr.predict(np.arange(0, 23.25, 0.25).reshape(-1, 1)))
            store_experiment_predictions(run.info.run_id, self.get_test_points(), load_curve)
