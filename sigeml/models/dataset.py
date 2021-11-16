import pandas as pd

from sigeml.schemas import DataProcessingConfig
from sigeml.services.sige import get_data_from_sige

class Dataset:
    def __init__(self, data_processing_config: DataProcessingConfig, data_path: str, group_consumption: bool = True) -> None:
        self.config = data_processing_config
        self.data: pd.DataFrame = pd.DataFrame()
        self.data_path = data_path
        self.group_consumption = group_consumption

    def load_data(self) -> None:
        self.data = pd.read_csv(self.data_path)

        if self.group_consumption:
            self.set_energy_consumption()

        #self.data = get_data_from_sige(self.config.query_params)
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

    def remove_outliers(self) -> None:
        if self.config.remove_outliers:
            self.data["consumption"] = hampel(
                self.data["consumption"], window_size=16, n=2, imputation=True
            )
