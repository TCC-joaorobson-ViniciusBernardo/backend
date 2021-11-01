import json
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import psycopg2
import psycopg2.extras
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


def store_experiment_predictions(run_id: str, test_data_points: dict, load_curve: dict) -> None:
    with psycopg2.connect(get_postgres_uri()) as conn:
        with conn.cursor() as curs:
            curs.execute(
                """INSERT INTO load_curves (run_id, test_data_points, load_curve)
                          VALUES (%s, %s, %s);""",
                (run_id, json.dumps(test_data_points), json.dumps(load_curve)),
            )


def get_experiment_predictions(run_id: str) -> dict:
    with psycopg2.connect(get_postgres_uri()) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as curs:
            curs.execute("""SELECT * FROM load_curves where run_id = %s;""", (run_id,))
            predictions = curs.fetchone()
    return predictions
