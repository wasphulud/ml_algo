"""This module implements the RandomForest class, which is an ensemble model that
uses decision trees to predict for regression and classification problems. It
trains multiple decision trees on different subsets of data and combines their
predictions for a more robust result. It extends GenericBagging, inheriting its
methods like fit and predict.
Requires pandas, ensemble.bagging, and trees.decision_trees submodules.
"""
import pandas as pd

from ensemble.bagging import GenericBagging
from trees.decision_trees import DecisionTree, DecisionTreeParams


class RandomForest(GenericBagging):
    """RandomForest model for Classification and Regression:

    This class is an implementation of a random forest ensemble model that combines
    multiple decision trees to make predictions on classification and regression
    problems. The RandomForest class extends the GenericBagging class and inherits
    its methods like fit and predict to train and predict the random forest model.

    Args:
        decision_tree_params (DecisionTreeParams, optional): an object containing
            hyperparameters for the decision tree. Defaults to DecisionTreeParams().
        n_estimators (int, optional): number of decision trees in the ensemble.
            Defaults to 10.
        max_samples_frac (float, optional): fraction of samples to use when training
            each decision tree. Defaults to 0.5.
        max_features_frac (float, optional): fraction of features to use when
            training each decision tree. Defaults to 1.
        num_processes (int, optional): number of processes to use when training each
            decision tree. Defaults to 1.

    Methods:
        fit(dataframe, target): fit the random forest model to the training data.
        predict(dataframe): predict the target values for the given test data.

    Raises:
        AssertionError: if max_features_frac is not specified in the constructor.
    """

    def __init__(
        self,
        decision_tree_params: DecisionTreeParams = DecisionTreeParams(),
        n_estimators: int = 10,
        max_samples_frac: float = 0.5,
        max_features_frac: float = 1,
        num_processes: int = 1,
    ) -> None:
        """Constructor of the class

        Args:
            model (class): model to be used as base estimator
            n_estimators (int, optional): number of base estimators. Defaults to 10.
            max_samples_frac (float, optional): number of samples to draw from X to train
            each base estimator. Defaults to 1.0.
            max_features_frac (float, optional): number of features to draw from X to train
            each base estimator. Defaults to 1.0.
        """
        model = DecisionTree(decision_tree_params=decision_tree_params)
        super().__init__(
            model=model,
            n_estimators=n_estimators,
            max_samples_frac=max_samples_frac,
            max_features_frac=max_features_frac,
            num_processes=num_processes,
        )
        assert max_features_frac is not None, "max_features_frac must be specified"

    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "RandomForest":
        """Fit the model

        Args:
            dataframe (np.array): features
            target (np.array): target
        """
        super()._fit(dataframe, target)
        return self
