from enum import Enum


class ModelType(str, Enum):
    """
    Names of existing models
    """
    R = "Lasso"
    L = "Ridge"
    DTC = "DecisionTreeClassifier"
    DTR = "DecisionTreeRegressor"


class OneCriterionType(str, Enum):
    """
    Classification:

    criterion{“gini”, “entropy”, “log_loss”};
    The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss”
    and “entropy” both for the Shannon information gain

    Regression:

    criterion{“squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, default="squared_error"
    The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error,
    which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of
    each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential
    splits, “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal
    node, and “poisson” which uses reduction in Poisson deviance to find splits.
    """
    gini = "gini"
    entropy = "entropy"
    log_loss = "log_loss"
    squared_error = "squared_error"
    friedman_mse = "friedman_mse"
    absolute_error = "absolute_error"
    poisson = "poisson"


class ClassificationCriterionType(str, Enum):
    """
    criterion{“gini”, “entropy”, “log_loss”}
    The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss”
    and “entropy” both for the Shannon information gain
    """
    gini = "gini"
    entropy = "entropy"
    log_loss = "log_loss"


class SplitterType(str, Enum):
    """
    splitter{“best”, “random”}
    The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and
    “random” to choose the best random split.
    """
    best = "best"
    random = "random"


class RegressionCriterionType(str, Enum):
    """
    criterion{“squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, default="squared_error"
    The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error,
    which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of
    each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential
    splits, “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal
    node, and “poisson” which uses reduction in Poisson deviance to find splits.
    """
    squared_error = "squared_error"
    friedman_mse = "friedman_mse"
    absolute_error = "absolute_error"
    poisson = "poisson"


class DatasetType(str, Enum):
    """
    Types for creating test datasets
    """
    regression = "regression"
    classification = "classification"
