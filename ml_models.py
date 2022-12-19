import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from fastapi import HTTPException
from api_models import ModelType
import pandas as pd
from io import BytesIO
from work_with_database import DBModel

ML_MODELS = {
    ModelType.R: Ridge,
    ModelType.L: Lasso,
    ModelType.DTC: DecisionTreeClassifier,
    ModelType.DTR: DecisionTreeRegressor
}


def convert_byte_data(data: bytearray, extension: str) -> pd.DataFrame:
    """
    Convert data from byte format to pd.DataFrame
    :param data: file in byte format
    :param extension: extension of file
    :return: converted data  pd.DataFrame
    """
    if extension == "xlsx" or extension == "xls":
        data = pd.read_excel(data)
    elif extension == "csv":
        data = pd.read_csv(BytesIO(data))
    else:
        raise HTTPException(status_code=415, detail="Invalid file type")
    return data


class MLModel(object):
    """
    Wrapper class with machine learning models

    Attributes:
        self.type_model - Name of ML model;
        self.params - determined parameters of ML model;
        self.model_name - unique name for ML model;
        self.model - ML model;
        self.is_trained - flag that model is trained;
        self.train_columns - columns in train dataset;
        self.target_column - target column

    Methods:
        _create_model - Create model with type self.type_model with parameters self.params;
        path_to_model - Generate path to current model;
        _read_model - Read model from disk;
        dump_model - Dump model to the disk;
        fit - Fit model;
        predict - Predict data using current model;
        get_info - Get basic info about model;
        delete_model - Delete current model from disk;
    """

    def __init__(self, type_model: ModelType = None, params: dict[str, any] = None, model_name: str = None):
        """
        Create MLModel
        :param type_model: type of created model
        :param params: parameters for created model
        :param model_name: name of the model to load from disk
        :return: None
        """
        self.type_model = type_model
        self.params = params
        self.model_name = model_name

        self.model = None
        self.is_trained = False
        self.train_columns = None
        self.target_column = None

        if type_model is not None:
            self._create_model()
        elif model_name is not None:
            self._read_model()

    def _create_model(self) -> None:
        """
        Create model with type self.type_model with parameters self.params
        :return: None
        """
        self.model = ML_MODELS[self.type_model](**self.params)
        models_list = DBModel.get_grouped_models_by_type(self.type_model)
        ids = list(map(lambda x: int(x.split("_")[1]), models_list))
        num_model = sorted(list(set(np.arange(max(ids + [0]) + 2)) - set(ids)))[0]
        self.model_name = f"{self.type_model}_{num_model}"

    def check_exist_model(self) -> bool:
        """
        Check model exist in DataBase
        :return: True - exist,
                False - not exist
        """
        return self.model_name in DBModel.get_models_list()

    def _read_model(self) -> None:
        """
        Read model from DataBase
        :return: None
        :raise:
            HTTPException - "The model with the given name is not exist"
        """
        if not self.check_exist_model():
            raise HTTPException(status_code=404, detail="The model with the given name is not exist")

        data_model = DBModel.get_model(self.model_name)

        self.model = data_model["model"]
        self.is_trained = data_model["is_trained"]
        self.train_columns = data_model["train_columns"]
        self.params = data_model["params"]
        self.target_column = data_model["target_column"]
        self.type_model = self.model_name.split("_")[0]

    def dump_model(self) -> None:
        """
        Dump model to the DataBase
        :return: None
        """
        model_data = {"model": self.model,
                      "is_trained": self.is_trained,
                      "train_columns": self.train_columns,
                      "params": self.params,
                      "target_column": self.target_column,
                      }
        DBModel.dump_model(model_name=self.model_name, model_data=model_data)

    def fit(self, df: pd.DataFrame, target_column: str = "target") -> None:
        """
        Fit model
        :param df: train dataset
        :param target_column: target_column
        :return: None
        :raise:
            HTTPException - "No 'target' column in data"
            HTTPException - "No train columns in data"
        """
        self.target_column = target_column
        if self.target_column not in df.columns:
            raise HTTPException(status_code=400, detail="No target column in data")
        if len(df.columns) == 1:
            raise HTTPException(status_code=400, detail="No train columns in data")
        self.train_columns = list(set(df.columns) - {self.target_column})
        self.model.fit(df[self.train_columns], df[self.target_column])
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> list:
        """
        Predict data using current model
        :param df: dataset for predicting data
        :return: list of predicted data
        :raise:
            HTTPException - "This model is not trained"
            HTTPException - "Mismatch of column names with train data"
        """
        if not self.is_trained:
            raise HTTPException(status_code=418, detail="This model is not trained")
        if len(set(self.train_columns) - set(df.columns)) != 0:
            raise HTTPException(status_code=400, detail="Mismatch of column names with train data")
        return list(self.model.predict(df[self.train_columns]))

    def get_info(self) -> dict[str, bool | dict | None | str | ModelType]:
        """
        Get basic info about model
        :return: basic info about model
        """
        return {
            "model_name": self.model_name,
            "type_model": self.type_model,
            "is_trained": self.is_trained,
            "train_columns": self.train_columns,
            "target_column": self.target_column,
        }

    def delete_model(self) -> None:
        """
        Delete current model from disk
        :return: None
        """
        DBModel.delete_model(model_name=self.model_name)
