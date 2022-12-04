from pydantic import BaseModel
from enum_models import *


class ONEClassParams(BaseModel):
    """
    One class with parameters for all models
    """
    alpha: float | None = 1
    fit_intercept: bool | None = None
    max_iter: int | None = 100
    random_state: int | None = 42
    splitter: SplitterType | None = SplitterType.best
    max_depth: int | None = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0
    criterion: OneCriterionType | None = OneCriterionType.gini


class RLParams(BaseModel):
    """
    Parameters for Ridge and Lasso models
    """
    alpha: float | None = 1
    fit_intercept: bool | None = None
    max_iter: int | None = 100
    random_state: int | None = 42


class DTCParams(BaseModel):
    """
    Parameters for DecisionTreeClassifier model
    """
    criterion: ClassificationCriterionType | None = ClassificationCriterionType.gini
    splitter: SplitterType | None = SplitterType.best
    max_depth: int | None = 8
    min_samples_split: int | None = 2
    min_samples_leaf: int | None = 1
    min_weight_fraction_leaf: float = 0
    random_state: int | None = 42


class DTRParams(BaseModel):
    """
    Parameters for DecisionTreeRegressor model
    """
    criterion: RegressionCriterionType | None = RegressionCriterionType.squared_error
    splitter: SplitterType | None = SplitterType.best
    max_depth: int | None = 8
    min_samples_split: int | None = 2
    min_samples_leaf: int | None = 1
    min_weight_fraction_leaf: float | None = 0
    random_state: int | None = 59


mapper_models_params = {
    ModelType.R: RLParams,
    ModelType.L: RLParams,
    ModelType.DTC: DTCParams,
    ModelType.DTR: DTRParams,
}


def reformat_model_params(model_params: ONEClassParams, model_type: ModelType):
    """
    Reformat model_params with corresponding model_type
    :param model_params: model parameters from request
    :param model_type: type of model from request
    :return: model_params with corresponding model_type
    """
    result_params = mapper_models_params[model_type]().dict()
    dict_model_params = model_params.dict()
    for key in model_params.__fields__:
        if key in result_params:
            if key == "criterion":
                if model_type == ModelType.DTC and dict_model_params[key] in RegressionCriterionType.__members__:
                    result_params[key] = ClassificationCriterionType.gini
                    continue
                if model_type == ModelType.DTR and dict_model_params[key] in ClassificationCriterionType.__members__:
                    result_params[key] = RegressionCriterionType.squared_error
                    continue
            result_params[key] = dict_model_params[key]
    return mapper_models_params[model_type](**result_params)


class AvailableModelTypeRespond(BaseModel):
    """
    Respond for get_available_model_types
    """
    available_model_types: list[str]


class ModelsListRespond(BaseModel):
    """
    Respond for get_models_list
    """
    models_list: list[str]


class RespondError(BaseModel):
    """
    Parameters for DecisionTreeRegressor model
    """
    detail: str


class CreateModelRespond(BaseModel):
    """
    Respond for create_model request
    """
    path: str = ""
    model_type: ModelType | None = None
    # model_params: ONEClassParams


class FitModelRespond(BaseModel):
    """
    Respond for fit_model request
    """
    is_trained: bool


class PredictModelRespond(BaseModel):
    """
    Respond for get_predictions request
    """
    predict: list[float]


class GetInfoRespond(BaseModel):
    """
    Respond for get_model_info
    """
    model_name: str
    type_model: ModelType
    is_trained: bool
    train_columns: list[str]
    target_column: str
    # model_params: RLParams | DTCParams | DTRParams
