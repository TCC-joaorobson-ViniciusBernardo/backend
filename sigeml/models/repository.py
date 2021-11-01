import logging
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from sigeml.config.config import get_postgres_uri
from sigeml.services.predictions import get_experiment_predictions


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

    def filter_experiments_by_name(self, experiments, experiment_name: str):
        experiments_ids: dict = {}
        if experiment_name:
            experiments = filter(lambda exp: exp.name == experiment_name, experiments)
            experiments_ids = {exp.experiment_id: exp.name for exp in experiments}
        else:
            experiments_ids = {exp.experiment_id: exp.name for exp in experiments}

        return experiments_ids

    @staticmethod
    def add_predictions_in_run_info(run_info) -> None:
        predictions = get_experiment_predictions(run_info["info"]["run_id"])
        run_info["predictions"] = {}
        if predictions:
            run_info["predictions"]["load_curve"] = predictions.get("load_curve", [])
            run_info["predictions"]["test_data_points"] = predictions.get("test_data_points", [])
        else:
            run_info["predictions"]["load_curve"] = []
            run_info["predictions"]["test_data_points"] = []

    def filter_run_by_model_name(self, run, models):
        if models:
            return True if run.data.tags["model_name"] in models else False
        return True

    def filter_run_by_status(self, run, statuses):
        if statuses:
            return True if run.info.status in statuses else False
        return True

    def get_runs_infos(self, experiment_name: str, models: list, statuses: list) -> list:
        self.logger.info("Retrieving list of experiments")
        models_run_ids = self.get_run_ids_from_registered_models_versions()
        runs_infos: list = []

        experiments = self.client.list_experiments()
        experiments_ids = self.filter_experiments_by_name(experiments, experiment_name)

        runs = self.client.search_runs(experiment_ids=list(experiments_ids.keys()))

        for run in runs:
            if self.filter_run_by_model_name(run, models) and self.filter_run_by_status(
                run, statuses
            ):
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
                self.add_predictions_in_run_info(run_info)
                runs_infos.append(run_info)

        return runs_infos

    def delete_run(self, run_id: str) -> None:
        self.logger.info(f"Deleting run: {run_id}")
        try:
            self.client.delete_run(run_id)
        except mlflow.exceptions.MlflowException as e:
            self.logger.error(f"Unable to delete the runn: {e}")
