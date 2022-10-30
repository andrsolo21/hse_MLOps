from sklearn.datasets import make_regression, make_classification
from enum_models import DatasetType
import pandas as pd
import os

EXAMPLE_FOLDER = "data_examples"


def store_test_data(name: str, dataset_type: DatasetType = "regression", **kwargs) -> str:
    """
    Generate dataset for testing models
    :param name: name of created data
    :param dataset_type: type of data, dataset_type = {'regression', 'classification'}
    :param kwargs: parameters for sklearn.datasets.make_classification function
    :return: path to created data
    """
    if dataset_type == DatasetType.regression:
        x, y = make_regression(**kwargs)
    elif dataset_type == DatasetType.classification:
        x, y = make_classification(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset type {dataset_type}")

    data = pd.DataFrame(x)
    data.columns = [f"feature_{i}" for i in data.columns]
    data["target"] = y
    data.to_csv(os.path.join(EXAMPLE_FOLDER, f"{name}.csv"), index=False)
    return os.path.join(EXAMPLE_FOLDER, f"{name}.csv")


if __name__ == "__main__":
    print(store_test_data(name="regression_100s_10f",
                          dataset_type=DatasetType.regression,
                          n_samples=100,
                          n_features=10)
          )
    print(store_test_data(name="regression_100s_12f",
                          dataset_type=DatasetType.regression,
                          n_samples=100,
                          n_features=12)
          )
    print(store_test_data(name="classification_100s_10f",
                          dataset_type=DatasetType.classification,
                          n_samples=100,
                          n_features=10)
          )
    print(store_test_data(name="classification_100s_12f",
                          dataset_type=DatasetType.classification,
                          n_samples=100,
                          n_features=12)
          )
