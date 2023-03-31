import pandas as pd

from ensemble.bagging import GenericBagging
from trees.decision_trees import DecisionTree, DecisionTreeParams


class RandomForest(GenericBagging):
    """Random Forest model for Classification and Regression

    Args:
        GenericBagging (class): generic bagging model
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
