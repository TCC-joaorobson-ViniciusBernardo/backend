from datetime import datetime

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


def request_quaterly_consumption(
    query_params: SIGEQueryParams
) -> dict:

    payload: dict = query_params.dict()
    data = requests.get(get_sige_api_url(), params=payload)
    data = data.json()
    return data


def get_data_from_sige(
    query_params: SIGEQueryParams
) -> pd.DataFrame:
    
    data = request_quaterly_consumption(query_params)
    df = convert_sige_api_data_to_dataframe(data)

    return df
