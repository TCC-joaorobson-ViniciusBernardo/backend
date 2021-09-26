import logging
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


class ModelsRepository:
    
    def __init__(self) -> None:
        mlflow.set_tracking_uri("postgresql://postgres:password@postgres:5432/mlflow")
        self.client = MlflowClient()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)


    def get_registered_models(self):
        registered_models = self.client.list_registered_models()

        return [{"model_name": r.name, "creation_time": r.creation_timestamp} for r in registered_models]


    def get_model_versions(self, model_name: str) -> list[dict]:
        versions = self.client.search_model_versions(f"name='{model_name}'")

        return [dict(model_version) for model_version in versions]   


    def retrieve_model_metrics(self, run_id: str) -> Optional[dict]:
        self.logger.info(f"Retrieving metric for run_id: {run_id}")
        try:
            run = self.client.get_run(run_id)
        except Exception as e:
            return {} 
        else:
            return run.data.metrics
        

    def delete_model_version(self, model_name: str, version: Optional[int]) -> None:
        self.logger.info(f"Deleting model {model_name} with version {version}")
        try:
            self.client.delete_model_version(name=model_name, version=str(version))
        except Exception as e:
            self.logger.error(f"Unable to delete the model: {e}")
        
    def delete_model(self, model_name: str) -> None:
        self.logger.info(f"Deleting model: {model_name}")
        try:
            self.client.delete_registered_model(name=model_name)
        except Exception as e:
            self.logger.error(f"Unable to delete the model: {e}")
