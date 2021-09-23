import mlflow
from mlflow.tracking import MlflowClient


def get_registered_models():
    mlflow.set_tracking_uri("postgresql://postgres:password@postgres:5432/mlflow")
    cl = MlflowClient()
    registered_models = cl.list_registered_models()

    return [{"model_name": r.name, "creation_time": r.creation_timestamp} for r in registered_models]


def get_model_versions(model_name: str = ""):
    mlflow.set_tracking_uri("postgresql://postgres:password@postgres:5432/mlflow")
    cl = MlflowClient()
    versions = cl.search_model_versions(f"name='{model_name}'")

    return [dict(model_version) for model_version in versions]   
