import json
import pickle

import pandas as pd

from sigeml.models.dataset import Dataset
from sigeml.models.time_series import TimeSeriesModel
from sigeml.services.sige import get_data_from_sige, request_realtime_consumption
from sigeml.services.publisher import Publisher
from sigeml.schemas import SIGEQueryParams


SUPPORTED_IDS = {1: "CPD 1"}


class TimeSeriesHandler:
    def __init__(
        self,
        initial_data: Dataset,
        model: TimeSeriesModel,
        publisher: Publisher,
        window_size: int = 10,
    ) -> None:
        self.data_scaler = pickle.load(open("sigeml/models/time_series/scaler.pk", "rb"))
        self.dataset = initial_data
        self.model = model
        self.publisher = publisher
        self.window_size = window_size
        self.predictions = self.window_size * [None]

    def get_predictions(self):
        pass

    def get_last_consumption_values(self, id_: int):
        params = SIGEQueryParams(id=id_, type="realtime")
        data = get_data_from_sige(params)
        return data

    def get_latest_n_timestamps(self, timestamp: int) -> list:
        return list(range(timestamp - (self.window_size - 1) * 60, timestamp + 60, 60))

    def loop(self):
        while True:
            #  TODO: for more than one building, train/predict in parallel
            for _id in SUPPORTED_IDS:
                new_data = self.get_last_consumption_values(_id)
                new_data["collection_date"] = pd.to_datetime(
                    new_data["collection_date"], format="%Y-%m-%d %H:%M:%S"
                )

                timestamps = self.get_latest_n_timestamps(
                    int(new_data.iloc[0].collection_date.timestamp())
                )

                self.dataset.data = pd.concat([self.dataset.data[1:], new_data]).reset_index(
                    drop=True
                )
                self.dataset.remove_outliers()
                data = self.data_scaler.transform(self.dataset.data[["consumption"]])
                self.model.retrain(data[:-1].reshape((1, 96, 1)), data[-1].reshape((1, 1, 1)))

                prediction = self.data_scaler.inverse_transform(
                    self.model.predict(data[1:].reshape((1, 96, 1)))
                )
                self.predictions = self.predictions[1:]
                self.predictions.append(float(prediction[0][0]))
                print(f"Prediction: {prediction}")

                self.publisher.send_data(
                    "CPD_1",
                    json.dumps(
                        {
                            "data": self.dataset.data.consumption.values[-9:].tolist(),
                            "predictions": self.predictions,
                            "timestamps": timestamps,
                        }
                    ),
                )
