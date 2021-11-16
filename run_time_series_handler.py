from sigeml.services.time_series_handler import *
from sigeml.services.publisher import Publisher
from sigeml.schemas import DataProcessingConfig
from sigeml.models.dataset import Dataset
from sigeml.models.time_series import LSTMModel


if __name__ == "__main__":
    dataset = Dataset(
        DataProcessingConfig(remove_outliers=False, query_params={"id": 1, "type": "realtime"}),
        data_path="/app/sigeml/models/time_series_initial_data.csv",
        group_consumption=False,
    )
    dataset.load_data()

    model = LSTMModel("sigeml/models/time_series/model.json")
    publisher = Publisher()

    time_series_handler = TimeSeriesHandler(dataset, model, publisher)
    time_series_handler.loop()
