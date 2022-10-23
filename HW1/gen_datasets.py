from sklearn.datasets import make_regression, make_classification
import pandas as pd
import os

target_folder_name = "data_examples"


def store_regression(**kwargs):
    x, y = make_regression(**kwargs)
    data = pd.DataFrame(x)
    data.columns = [f"feature_{i}" for i in data.columns]
    data["target"] = y
    data.to_csv(os.path.join(target_folder_name, "regression_data.csv"), index=False)


def store_classification(**kwargs):
    x, y = make_classification(**kwargs)
    data = pd.DataFrame(x)
    data.columns = [f"feature_{i}" for i in data.columns]
    data["target"] = y
    data.to_csv(os.path.join(target_folder_name, "classification_data.csv"), index=False)


store_regression(n_samples=100, n_features=10)
store_classification(n_samples=100, n_features=10)
