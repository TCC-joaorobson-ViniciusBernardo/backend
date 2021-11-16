from datetime import datetime
import time

import pandas as pd
import requests

from sigeml.config.config import get_sige_api_url
from sigeml.schemas import SIGEQueryParams


def convert_sige_api_data_to_dataframe(data: dict) -> pd.DataFrame:
    consumption = data.get("consumption")
    df = pd.DataFrame()

    if consumption:
        df = pd.DataFrame(consumption, columns=["collection_date", "consumption"])

    return df


def request_quarterly_consumption(query_params: SIGEQueryParams) -> dict:

    payload: dict = query_params.dict()
    data = requests.get(get_sige_api_url() + "/graph/quarterly-total-consumption/", params=payload)
    return data


def request_realtime_consumption(query_params: SIGEQueryParams, period: int = 3) -> dict:

    payload: dict = {"id": query_params.id}
    active_power: list = []
    for p in range(period):
        data = requests.get(get_sige_api_url() + "/realtime-measurements/", params=payload)
        if data.json():
            data = data.json()[0]
            active_power.append(data.get("total_active_power"))
            collection_date = data.get("collection_date")
        if p < period - 1:
            time.sleep(60)

    data = {
        "consumption": [
            [
                datetime.fromisoformat(collection_date).strftime("%Y-%m-%d %H:%M:%S"),
                (1 / 60) * sum(active_power),
            ]
        ]
    }
    return data


def get_data_from_sige(query_params: SIGEQueryParams) -> pd.DataFrame:

    if query_params.type == "hourly":
        data = request_quaterly_consumption(query_params)
        df = convert_sige_api_data_to_dataframe(data)
    elif query_params.type == "realtime":
        data = request_realtime_consumption(query_params)
        df = convert_sige_api_data_to_dataframe(data)

    return df
