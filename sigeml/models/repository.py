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
    def get_models_versions(self) -> list[dict]:
        self.logger.info("Retrieving stored models versions")
        versions = self.client.search_model_versions("")

        return [dict(model_version) for model_version in versions]

    def retrieve_model_metrics(self, run_id: str) -> Optional[dict]:
        self.logger.info(f"Retrieving metric for run_id: {run_id}")
        try:
            run = self.client.get_run(run_id)
        except mlflow.exceptions.MlflowException as e:
            self.logger.error(f"Error retrieving metric for run: {e}")
            return {}
        else:
            return run.data.metrics

    def delete_model_version(self, model_name: str, version: Optional[int]) -> None:
        self.logger.info(f"Deleting model {model_name} with version {version}")
        try:
            self.client.delete_model_version(name=model_name, version=str(version))
        except mlflow.exceptions.MlflowException as e:
            self.logger.error(f"Unable to delete the model: {e}")

    def delete_model(self, model_name: str) -> None:
        self.logger.info(f"Deleting model: {model_name}")
        try:
            self.client.delete_registered_model(name=model_name)
        except mlflow.exceptions.MlflowException as e:
            self.logger.error(f"Unable to delete the model: {e}")


class ExperimentsRepository(Repository):
    def get_run_ids_from_registered_models_versions(self) -> list:
        models = self.client.search_model_versions("")
        run_ids = [model.run_id for model in models]

        return run_ids

    def get_runs_infos(self) -> list:
        self.logger.info("Retrieving list of experiments")
        models_run_ids = self.get_run_ids_from_registered_models_versions()
        runs_infos: list = []
        experiments = self.client.list_experiments()
        experiments_ids = {exp.experiment_id: exp.name for exp in experiments}

        runs = self.client.search_runs(list(experiments_ids.keys()))

        for run in runs:
            run_info = run.to_dictionary()
            run_info.update(
                {"experiment_name": experiments_ids[run_info["info"]["experiment_id"]]}
            )
            run_info.update(
                {
                    "has_registered_model": True
                    if run_info["info"]["run_id"] in models_run_ids
                    else False
                }
            )
            run_info["data"]["tags"] = {"model_name": run_info["data"]["tags"]["model_name"]}
            runs_infos.append(run_info)

        return runs_infos
