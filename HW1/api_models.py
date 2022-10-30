from pydantic import BaseModel
from enum_models import *


class RLParams(BaseModel):
    """
    Parameters for Ridge and Lasso models
    """
    alpha: float | None = 1
    fit_intercept: bool | None = None
    normalize: bool | None = None
    max_iter: int | None = 100
    random_state: int | None = 42


class DTCParams(BaseModel):
    """
    Parameters for DecisionTreeClassifier model
    """
    criterion: ClassificationCriterionType | None = ClassificationCriterionType.gini
    splitter: SplitterType | None = SplitterType.best
    max_depth: int | None = None
    min_samples_split: int | float | None = 2
    min_samples_leaf: int | float | None = 1
    min_weight_fraction_leaf: float | None = 0
    random_state: int | None = 42


class DTRParams(BaseModel):
    """
    Parameters for DecisionTreeRegressor model
    """
    criterion: RegressionCriterionType | None = RegressionCriterionType.squared_error
    splitter: SplitterType | None = SplitterType.best
    max_depth: int | None = None
    min_samples_split: int | float | None = 2
    min_samples_leaf: int | float | None = 1
    min_weight_fraction_leaf: float | None = 0
    random_state: int | None = 59


class RespondError404(BaseModel):
    """
    Parameters for DecisionTreeRegressor model
    """
    detail: str
