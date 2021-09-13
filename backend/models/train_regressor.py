import os
import warnings
import sys

from hampel import hampel
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_xgbregressor(n_estimators=10, max_depth=3):
    mlflow.set_tracking_uri("postgresql://postgres:password@postgres:5432/mlflow")
    warnings.filterwarnings("ignore")

    data = pd.read_csv('/app/backend/models/quarterly_measurements_CPD1.csv')
    data['collection_date'] = pd.to_datetime(data['collection_date'], format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values('collection_date')
    data['consumption'] = data['generated_energy_peak_time'] + \
                          data['generated_energy_off_peak_time'] + \
                          data['consumption_peak_time'] + \
                          data['consumption_off_peak_time']

    data['no_outliers_consumption'] = hampel(data['consumption'], 
                                             window_size=16, 
                                             n=2, 
                                             imputation=True)
    X = (data.collection_date.dt.hour + data.collection_date.dt.minute / 60).values.reshape((len(data), 1))
    y = data['no_outliers_consumption'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    with mlflow.start_run():
        xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth).fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("XGB model (n_estimators=%d, max_depth=%d):" % (n_estimators, max_depth))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.xgboost.log_model(xgb, "model", registered_model_name="XGBRegressor")
        else:
            mlflow.xgboost.log_model(xgb, "model")
#
#
#if __name__ == "__main__":
#    train_xbgregressor()
