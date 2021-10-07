import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from xgboost import DMatrix

from sigeml.config.config import get_postgres_uri


def predict_load_curve(data: list) -> dict:
    mlflow.set_tracking_uri(get_postgres_uri())
    client = MlflowClient()
    registered_models = client.list_registered_models()

    if not registered_models:
        return {}

    else:
        model = mlflow.xgboost.load_model(
            f"models:/{registered_models[0].name}/{registered_models[0].latest_versions[0].version}"
        )

        prediction = model.predict(DMatrix(np.array([[d] for d in data])))

        return {"prediction": prediction.tolist()}
