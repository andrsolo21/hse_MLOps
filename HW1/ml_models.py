from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle as pkl

import os

MODELS_PATH = "models_dir"

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


class MLModel(object):
    def __init__(self, type_model=None, params=None, model_name=None):
        self.type_model = type_model
        self.params = params
        self.model_name = model_name

        self.model = None
        self.num_model = None

        if type_model is not None:
            self._create_model()
        elif model_name is not None:
            self._read_model()

    def _create_model(self):
        self.model = ML_MODELS[self.type_model](**self.params)
        _, models_dct = get_models_list()
        self.num_model = max(models_dct[self.type_model]) + 1 if len(models_dct[self.type_model]) > 0 else 0
        self.model_name = f"{self.type_model}_{self.num_model}"

    def fit_model(self, x, y):
        self.model.fit(x, y)
        return self

    def _read_model(self):
        with open(os.path.join(MODELS_PATH, f'{self.model_name}.pkl'), 'rb') as file:
            self.model = pkl.load(file)
        self.type_model = int(self.model_name.split("_")[0])
        self.num_model = int(self.model_name.split("_")[1])

    def dump_model(self):
        with open(os.path.join(MODELS_PATH, f'{self.model_name}.pkl'), 'wb') as file:
            pkl.dump(self.model, file)
        return os.path.join(MODELS_PATH, f'{self.model_name}.pkl')