import logging
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from sigeml.config.config import get_postgres_uri


class Repository:
    def __init__(self) -> None:
        mlflow.set_tracking_uri(get_postgres_uri())
        self.client = MlflowClient()
        self.logger = logging.getLogger(__name__)


class ModelsRepository(Repository):
    def __init__(self) -> None:
        super().__init__()

    def get_registered_models(self):
        self.logger.info("Retrieving registered models")
        registered_models = self.client.list_registered_models()

        return [
            {"model_name": r.name, "creation_time": r.creation_timestamp}
            for r in registered_models
        ]

    def get_models_versions(self, model_name: str) -> list[dict]:
        self.logger.info("Retrieving stored models versions")
        versions = self.client.search_model_versions("")

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


class ExperimentsRepository(Repository):
    def __init__(self) -> None:
        super().__init__()

    def get_experiments(self) -> list:
        self.logger.info("Retrieving list of experiments")
        experiments = self.client.list_experiments()

        return [dict(e) for e in experiments]

    def get_runs_infos(self, experiment_id: str) -> list:
        self.logger.info(f"Retrieving list of runs for experiment: {experiment_id}")
        runs = self.client.list_run_infos(experiment_id)

        return [dict(self.client.get_run(r.run_id)) for r in runs]
