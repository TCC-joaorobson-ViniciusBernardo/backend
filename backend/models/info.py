from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


class ModelsInformation:
    
    def __init__(self) -> None:
        mlflow.set_tracking_uri("postgresql://postgres:password@postgres:5432/mlflow")
        self.client = MlflowClient()

    def get_registered_models(self):
        registered_models = self.client.list_registered_models()

        return [{"model_name": r.name, "creation_time": r.creation_timestamp} for r in registered_models]


    def get_model_versions(self, model_name: str) -> list[dict]:
        versions = self.client.search_model_versions(f"name='{model_name}'")

        return [dict(model_version) for model_version in versions]   


    def retrieve_model_metrics(self, run_id: str) -> Optional[dict]:
        try:
            run = self.client.get_run(run_id)
        except mlflow.exceptions:
            print('ooooooooi')
            return {} 
        else:
            return run.data.metrics
        
