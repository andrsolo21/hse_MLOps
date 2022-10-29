from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle as pkl
import pandas as pd
import os

MODELS_PATH = "models_dir"

TARGET_COLUMN = "target"

ML_MODELS = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor
}


def get_models_list():
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


def convert_uploaded_file(uploaded_file):
    data = None
    if uploaded_file.filename.split(".")[-1] == "xlsx" or uploaded_file.filename.split(".")[-1] == "xls":
        data = pd.read_csv(uploaded_file.file)
    elif uploaded_file.filename.split(".")[-1] == "csv":
        data = pd.read_csv(uploaded_file.file)
    # else:
    #     return TODO: return error code wrong format
    return data


class MLModel(object):
    def __init__(self, type_model=None, params=None, model_name=None):
        self.type_model = type_model
        self.params = params
        self.model_name = model_name

        self.model = None
        self.num_model = None
        self.is_fitted = False
        self.train_columns = None

        if type_model is not None:
            self._create_model()
        elif model_name is not None:
            self._read_model()

    def _create_model(self):
        self.model = ML_MODELS[self.type_model](**self.params)
        _, models_dct = get_models_list()
        self.num_model = max(models_dct[self.type_model]) + 1 if len(models_dct[self.type_model]) > 0 else 0
        self.model_name = f"{self.type_model}_{self.num_model}"

    def path_to_model(self):
        return os.path.join(MODELS_PATH, f'{self.model_name}.pkl')

    def _read_model(self):
        with open(self.path_to_model(), 'rb') as file:
            # TODO check that file exist
            data_model = pkl.load(file)
        self.model = data_model["model"]
        self.is_fitted = data_model["is_fitted"]
        self.train_columns = data_model["train_columns"]
        self.params = data_model["params"]
        self.type_model = self.model_name.split("_")[0]
        self.num_model = int(self.model_name.split("_")[1])

    def dump_model(self):
        with open(self.path_to_model(), 'wb') as file:
            pkl.dump({"model": self.model,
                      "is_fitted": self.is_fitted,
                      "train_columns": self.train_columns,
                      "params": self.params
                      }, file)
        return os.path.join(MODELS_PATH, f'{self.model_name}.pkl')

    def fit(self, df):
        if TARGET_COLUMN not in df.columns:
            pass
            # TODO error with no target column
        if len(df.columns) == 1:
            pass
            # TODO return error no train data
        self.train_columns = list(set(df.columns) - {TARGET_COLUMN})
        self.model.fit(df[self.train_columns], df[TARGET_COLUMN])
        self.is_fitted = True
        return self

    def predict(self, df):
        if not self.is_fitted:
            pass
            # TODO return model is not trained
        if len(set(self.train_columns) - set(df.columns)) != 0:
            pass
            # TODO return columns mismatch
        return self.model.predict(df[self.train_columns])

    def get_info(self):
        return {
            "model_name": self.model_name,
            "type_model": self.type_model,
            "is_fitted": self.is_fitted,
            "train_columns": self.train_columns,
            "model_params": self.params,
        }

    def delete_model(self):
        return os.remove(self.path_to_model())
