from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from fastapi import HTTPException
from api_models import ModelType
import pickle as pkl
import pandas as pd
import os

MODELS_PATH = "models_dir"

ML_MODELS = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor
}


def get_models_list() -> (list[str], dict[str, list[str]]):
    """
    Get name of models from disk
    :return:
        models_list - list of existing models
        models_dct - {type_model: [model_names]}
    """
    models = os.listdir(MODELS_PATH)
    models_list = []
    models_dct = {key: [] for key in ML_MODELS}
    for model in models:
        model_name = model.split(".")[0]
        if len(model_name.split("_")) > 1:
            model_type = model_name.split("_")[0]
            model_id = model_name.split("_")[1]
            if model_type in models_dct and model_id.isdigit():
                models_dct[model_type].append(int(model_id))
                models_list.append(model_name)
    return models_list, models_dct


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

    def _create_model(self):
        """
        Create model with type self.type_model with parameters self.params
        :return: None
        """
        self.model = ML_MODELS[self.type_model](**self.params)
        _, models_dct = get_models_list()
        num_model = max(models_dct[self.type_model]) + 1 if len(models_dct[self.type_model]) > 0 else 0
        self.model_name = f"{self.type_model}_{num_model}"

    def path_to_model(self) -> str:
        """
        Generate path to current model
        :return: path to current model
        """
        return os.path.join(MODELS_PATH, f'{self.model_name}.pkl')

    def _read_model(self):
        """
        Read model from disk
        :return: None
        :raise:
            HTTPException - "The model with the given name is not exist"
        """
        if not os.path.exists(self.path_to_model()):
            raise HTTPException(status_code=404, detail="The model with the given name is not exist")
        with open(self.path_to_model(), 'rb') as file:
            data_model = pkl.load(file)
        self.model = data_model["model"]
        self.is_trained = data_model["is_trained"]
        self.train_columns = data_model["train_columns"]
        self.params = data_model["params"]
        self.target_column = data_model["target_column"]
        self.type_model = self.model_name.split("_")[0]

    def dump_model(self) -> str:
        """
        Dump model to the disk
        :return: path to dumped model
        """
        with open(self.path_to_model(), 'wb') as file:
            pkl.dump({"model": self.model,
                      "is_trained": self.is_trained,
                      "train_columns": self.train_columns,
                      "params": self.params,
                      "target_column": self.target_column,
                      }, file)
        return os.path.join(MODELS_PATH, f'{self.model_name}.pkl')

    def fit(self, df: pd.DataFrame, target_column: str = "target"):
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
            raise HTTPException(status_code=400, detail="No 'target' column in data")
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
            "target_column" : self.target_column,
            "model_params": self.params,
        }

    def delete_model(self):
        """
        Delete current model from disk
        :return: None
        """
        os.remove(self.path_to_model())
