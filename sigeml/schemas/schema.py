from typing import Literal, Optional, Union

from pydantic import BaseModel, StrictBool, StrictInt, StrictFloat, StrictStr, validator


class LoadCurveParams(BaseModel):
    building: str
    data: list


class XGBoostParams(BaseModel):
    n_estimators: StrictInt = 100
    max_depth: Optional[StrictInt] = None
    learning_rate: Optional[StrictFloat] = None
    gamma: Optional[StrictFloat] = None
    random_state: Optional[StrictInt] = None


class TrainConfig(BaseModel):
    model: Literal["xgboost"]
    test_size: StrictFloat = 0.2
    remove_outliers: StrictBool = True
    model_params: Optional[Union[XGBoostParams]] = XGBoostParams()
    is_experiment: StrictBool = True
    experiment_name: StrictStr = ""

    @validator("is_experiment", "experiment_name", always=True)
    def validate_experiment_name(  # pylint: disable=no-self-argument,no-self-use
        cls, value, values
    ) -> str:
        if "is_experiment" in values and values["is_experiment"] and not value:
            raise ValueError("Experiment must have a name")
        return value
