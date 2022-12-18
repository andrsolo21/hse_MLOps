import os
import unittest
import pandas as pd
from ml_models import MLModel
from enum_models import ModelType
from work_with_database import DBModel
from sklearn.metrics import mean_absolute_error as mae


class TestModels(unittest.TestCase):

    def setUp(self):
        DBModel.CONN_PARAMS = None
        self.model = MLModel(type_model=ModelType.L, params={})
        self.fitted_model = MLModel(type_model=ModelType.L, params={})
        self.df = pd.read_csv(os.path.join("data_examples", "regression_100s_10f.csv"))
        self.fitted_model.fit(self.df)

    def test_creating_models(self):
        self.assertEqual(MLModel(type_model=ModelType.L, params={}).type_model, "Lasso")
        self.assertEqual(MLModel(type_model=ModelType.R, params={}).type_model, "Ridge")
        self.assertEqual(MLModel(type_model=ModelType.DTC, params={}).type_model, "DecisionTreeClassifier")
        self.assertEqual(MLModel(type_model=ModelType.DTR, params={}).type_model, "DecisionTreeRegressor")

    def test_name_model(self):
        self.assertEqual(self.model.model_name, "Lasso_0")

    def test_for_fitted(self):
        self.assertEqual(self.model.is_trained, False)
        self.assertEqual(self.fitted_model.is_trained, True)

    def test_for_predict(self):
        self.df["predict"] = self.fitted_model.predict(self.df)
        self.assertEqual(mae(self.df["target"], self.df["predict"]) < 5, True)


if __name__ == "__main__":
    unittest.main()
