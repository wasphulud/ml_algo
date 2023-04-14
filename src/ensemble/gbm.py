from typing import List
import pandas as pd
import numpy as np


from abc_models.models import SupervisedTabularDataModel

from trees.decision_trees import DecisionTree


class GBMClassifier(SupervisedTabularDataModel):
    """Gradient Boosting Machine

    Args:
        SupervisedTabularDataModel (class): abstract class for supervised
        tabular data models
    """

    # TODO: add learning rate
    # TODO: add early stopping ?
    # TODO: add subsampling
    # TODO: add feature subsampling??
    # TODO: add regularization
    # TODO: add loss function

    num_classes: int = 0

    epsilon: float = 1e-5

    def __init__(
        self, n_estimators, decision_tree_params, learning_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators: list[list] = [[]]
        self.decision_tree_params = decision_tree_params
        self.decision_tree_params.mode = "regression"

    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "GBMClassifier":
        """Fit the model

        Args:
            dataframe (np.array): features
            target (np.array): target
        """
        # TODO: check if target is object
        self.num_classes = len(target.unique())
        assert self.num_classes > 1, "num classes int should be higher than 2"
        self.estimators = [[] for k in range(self.num_classes)]
        dataframe = dataframe.reset_index(drop=True)
        target = target.reset_index(drop=True)
        target_dummies = pd.get_dummies(target)
        target_dummies_columns = target_dummies.columns
        assert self.num_classes == len(
            target_dummies_columns
        ), "num classes and num columns must match"
        estimator = pd.DataFrame(
            0, index=dataframe.index, columns=list(target_dummies_columns)
        )

        for _ in range(1, self.n_estimators + 1):
            normalization = pd.concat(
                [np.sum(np.exp(estimator), axis=1)] * self.num_classes, axis=1
            )
            normalization.columns = target_dummies_columns
            probabilities = np.exp(estimator) / (normalization + self.epsilon)

            negative_gradient = target_dummies - probabilities
            for k in range(self.num_classes):
                kth_negative_gradient = negative_gradient[target_dummies_columns[k]]
                current_estimator = DecisionTree(
                    decision_tree_params=self.decision_tree_params
                )
                current_estimator.fit(dataframe, kth_negative_gradient)
                gamma_m_k = {}
                for j, leaf in enumerate(current_estimator.leaves):
                    region_indexes = leaf["target_indexes"]
                    kth_j_negative_gradient = kth_negative_gradient.iloc[region_indexes]
                    gamma_m_k_j = (
                        (self.num_classes - 1)
                        / self.num_classes
                        * np.sum(kth_j_negative_gradient)
                        / (
                            np.sum(
                                np.multiply(
                                    np.abs(kth_j_negative_gradient),
                                    1 - np.abs(kth_j_negative_gradient),
                                )
                            )
                        )
                    )
                    gamma_m_k[j] = gamma_m_k_j
                    estimator.iloc[region_indexes, k] += (
                        self.learning_rate * gamma_m_k_j
                    )
                self.estimators[k].append(current_estimator)

        return self

    def _predict(self, dataframe: pd.DataFrame) -> pd.Series:
        predictions = pd.DataFrame(
            0, index=dataframe.index, columns=list(range(self.num_classes))
        )
        for k, estimators in enumerate(self.estimators):
            for estimator in estimators:
                predictions[k] += estimator.predict(dataframe)

        predictions = np.exp(predictions) / pd.concat(
            [np.sum(np.exp(predictions), axis=1)] * self.num_classes, axis=1
        )

        predictions = predictions.idxmax(axis=1)
        return predictions
