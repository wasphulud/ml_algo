""" This module implements the adaboost algorithm
"""

from typing import List
import copy
import pandas as pd
import numpy as np

from abc_models.models import SupervisedTabularDataModel, STMT


class AdaBoost(SupervisedTabularDataModel):
    """AdaBoost model for Classification and Regression"""

    def __init__(
        self,
        model: STMT,
        n_estimators: int = 10,
        random_state: int = 42,
    ) -> None:
        """Constructor of the class

        Args:
            model (class): model to be used as base estimator
            n_estimators (int, optional): number of base estimators. Defaults to 10.
        """
        super().__init__()
        self.model = model
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.estimators: List[STMT] = []
        self.alphas: List[float] = []

        self.epsilon = 1e-5

    def _init_weights(self, n_samples: int) -> np.ndarray:
        """Initialize the sample weights uniformly.

        Args:
            n_samples (int): number of training samples.

        Returns:
            np.ndarray: sample weights.
        """
        return np.ones(n_samples) / n_samples

    def _compute_error(
        self, weights: np.ndarray, predictions: pd.Series, target: pd.Series
    ) -> float:
        """Compute the weighted error rate of an estimator.

        Args:
            weights (np.ndarray): sample weights.
            predictions (pd.Series): predictions of the current model
            target (pd.Series): target data.

        Returns:
            float: weighted error rate.
        """
        return np.sum(weights * (predictions != target)) / np.sum(weights)

    def _compute_alpha(self, err: float) -> float:
        """Compute the weight of an estimator.

        Args:
            err (float): weighted error rate.

        Returns:
            float: weight of the estimator.
        """
        return np.log((1 - err + self.epsilon) / (err + self.epsilon))

    @staticmethod
    def _update_weights(
        weights: np.ndarray, alpha: float, predictions: pd.Series, target: pd.Series
    ) -> np.ndarray:
        """Update the sample weights based on the performance of an estimator.

        Args:
            weights (np.ndarray): sample weights.
            alpha (float): weight of the estimator.
            predicsion (pd.Series): predictions from the current model
            target (pd.Series): target data.

        Returns:
            np.ndarray: updated sample weights.
        """

        weights = weights * np.exp(alpha * (predictions != target))
        return weights / np.sum(weights)

    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "AdaBoost":
        n_samples = dataframe.shape[0]
        weights = self._init_weights(n_samples)

        for _ in range(self.n_estimators):
            estimator = copy.deepcopy(self.model)
            indices = np.random.choice(
                dataframe.index, size=n_samples, replace=True, p=weights
            )

            training_dataframe = dataframe.loc[indices]
            training_target = target[indices]

            estimator.fit(training_dataframe, training_target)
            predictions = estimator.predict(dataframe)

            err = self._compute_error(weights, predictions, target)
            alpha = self._compute_alpha(err)

            self.estimators.append(estimator)
            self.alphas.append(alpha)

            weights = self._update_weights(weights, alpha, predictions, target)

        return self

    def _predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(dataframe.shape[0])
        for alpha, estimator in zip(self.alphas, self.estimators):
            current_pred = estimator.predict(dataframe)
            predictions += alpha * current_pred

        return np.sign(predictions)


'''
from typing import List
import copy
import pandas as pd
import numpy as np


from abc_models.models import SupervisedTabularDataModel, STMT


class AdaBoost(SupervisedTabularDataModel):
    """AdaBoost model for Classification and Regression

    Args:
        SupervisedTabularDataModel (class): abstract class for supervised
        tabular data models
    """

    def __init__(
        self,
        model: STMT,
        n_estimators: int = 10,
        # learning_rate: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """Constructor of the class

        Args:
            model (class): model to be used as base estimator
            n_estimators (int, optional): number of base estimators. Defaults to 10.
            #learning_rate (float, optional): learning rate. Defaults to 1.0.
        """
        super().__init__()
        self.model = model
        self.n_estimators = n_estimators
        # self.learning_rate = learning_rate
        self.estimators: List[STMT] = []
        self.alphas: List[float] = []
        self.epsilon = 1e-5
        self.random_state = random_state

    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "AdaBoost":
        n_samples = dataframe.shape[0]
        weights = (
            pd.Series(np.ones(n_samples), index=dataframe.index) / n_samples
        )  # 1- init the weights
        for _ in range(self.n_estimators):  # 2 -for each base estimator
            estimator = copy.deepcopy(self.model)
            # print(dataframe.index.shape, weights.shape)
            indices = np.random.choice(
                dataframe.index, size=n_samples, replace=True, p=weights
            )
            training_dataframe = dataframe.loc[indices]
            training_target = target[training_dataframe.index]
            self.estimators.append(estimator)
            estimator.fit(
                training_dataframe, training_target
            )  # a fit the estimator using the weights
            err = np.sum(weights * (estimator.predict(dataframe) != target)) / np.sum(
                weights
            )  # b - compute the error

            # print("err", err)
            alpha = np.log(
                (1 - err + self.epsilon) / (err + self.epsilon)
            )  # c - compute the alpha

            self.alphas.append(alpha)
            # print("alpha", alpha)
            weights = weights * np.exp(
                alpha * ((estimator.predict(dataframe) != target))
            )  # d - update the weights
            weights = weights / np.sum(weights)  # e - normalize the weights
        return self

    def _predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(dataframe.shape[0])
        for alpha, estimator in zip(self.alphas, self.estimators):
            current_pred = estimator.predict(dataframe)
            predictions += alpha * current_pred
        print(self.alphas)
        return np.sign(predictions)
'''
