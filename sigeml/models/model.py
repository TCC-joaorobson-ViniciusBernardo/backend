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


class LoadCurveModel:
    def __init__(self, config: TrainConfig) -> None:
        mlflow.set_tracking_uri(get_postgres_uri())

        if config.is_experiment:
            mlflow.set_experiment(config.experiment_name)

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def load_data(self) -> None:
        self.data = pd.read_csv("/app/backend/models/quarterly_measurements_CPD1.csv")
        self.data["collection_date"] = pd.to_datetime(
            self.data["collection_date"], format="%Y-%m-%d %H:%M:%S"
        )
        self.data = self.data.sort_values("collection_date")

    def set_energy_consumption(self) -> None:
        self.data["consumption"] = (
            self.data["generated_energy_peak_time"]
            + self.data["generated_energy_off_peak_time"]
            + self.data["consumption_peak_time"]
            + self.data["consumption_off_peak_time"]
        )

    def remove_outliers(self):
        if self.config.remove_outliers:
            self.data["consumption"] = hampel(
                self.data["consumption"], window_size=16, n=2, imputation=True
            )

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
        self.load_data()
        self.set_energy_consumption()
        self.remove_outliers()
        self.split_data()
        self.run()

    @abstractmethod
    def run(self):
        pass

    def evaluate(self, actual, pred) -> None:
        pass


class XGBoostModel(LoadCurveModel):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

    def run(self):
        with mlflow.start_run():
            params = self.config.model_params
            xgb = XGBRegressor(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                gamma=params.gamma,
                learning_rate=params.learning_rate,
                random_state=params.random_state,
            ).fit(self.X_train, self.y_train)

            xgb.fit(self.X_train, self.y_train)

            y_pred = xgb.predict(self.X_test)

            rmse = self.evaluate(self.y_test, y_pred)

            self.logger.info(
                f"XGB model (n_estimators={params.n_estimators}, max_depth={params.max_depth}, "
                f"learning_rate={params.learning_rate}, gamma={params.gamma}, "
                f"random_state={params.random_state})"
            )
            self.logger.info(f"RMSE: {rmse:.2f}")

            mlflow.log_param("n_estimators", params.n_estimators)
            mlflow.log_param("max_depth", params.max_depth)
            mlflow.log_param("learning_rate", params.learning_rate)
            mlflow.log_param("gamma", params.gamma)
            mlflow.log_param("random_state", params.random_state)
            mlflow.log_metric("rmse", rmse)

            if not self.config.is_experiment:
                mlflow.xgboost.log_model(xgb, "model")

    def evaluate(self, actual, pred) -> None:
        rmse = np.sqrt(mean_squared_error(actual, pred))
        return rmse
