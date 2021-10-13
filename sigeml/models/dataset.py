import pandas as pd

from sigeml.schemas import DataProcessingConfig
from sigeml.services.sige import get_data_from_sige

class Dataset:
    def __init__(self, data_processing_config: DataProcessingConfig) -> None:
        self.config = data_processing_config
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> None:
        #self.data = pd.read_csv("/app/sigeml/models/quarterly_measurements_CPD1.csv")
        self.data = get_data_from_sige(self.config.query_params)
        self.data["collection_date"] = pd.to_datetime(
            self.data["collection_date"], format="%Y-%m-%d %H:%M:%S"
        )
        self.data = self.data.sort_values("collection_date")

    def remove_outliers(self) -> None:
        if self.config.remove_outliers:
            self.data["consumption"] = hampel(
                self.data["consumption"], window_size=16, n=2, imputation=True
            )
