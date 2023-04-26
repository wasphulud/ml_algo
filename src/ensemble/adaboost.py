""" This module implements the adaboost algorithm
AdaBoost (short for Adaptive Boosting) is a machine learning algorithm that combines multiple "weak" classifiers into a "strong" classifier. It was proposed by Yoav Freund and Robert Schapire in 1995. In each iteration of the algorithm, it trains a weak classifier on the data, and then adjusts the weights of the misclassified samples to give them more weight. The algorithm then trains the next weak classifier on the modified data, and continues the process for a pre-determined number of iterations or until the classifier reaches a desired level of accuracy. The final classifier is a weighted combination of the weak classifiers, where each weak classifier's weight is proportional to its accuracy. AdaBoost is commonly used in classification problems and has been shown to be effective in a wide range of applications.
"""

from typing import List, Optional
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
        random_state: Optional[int] = None,
    ) -> None:
        """Constructor of the class

        Args:
            model (class): model to be used as base estimator
            n_estimators (int, optional): number of base estimators. Defaults to 10.
        """
        super().__init__()
        self.model = model
        self.n_estimators = n_estimators

        self.estimators: List[STMT] = []
        self.alphas: List[float] = []
        self.epsilon = 1e-5
        np.random.seed(random_state)

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
        weights = self._init_weights(n_samples)  # 1- init the weights

        for m in range(self.n_estimators):  # 2 - for each base estimator
            if m % 10 == 0:
                print(f"Initiating estimator {m} out of {self.n_estimators} ... ")
            estimator = copy.deepcopy(self.model)
            indices = np.random.choice(
                dataframe.index, size=n_samples, replace=True, p=weights
            )

            training_dataframe = dataframe.loc[indices]
            training_target = target[indices]

            estimator.fit(
                training_dataframe, training_target
            )  # a - fit the estimator using the weights
            predictions = estimator.predict(dataframe)

            err = self._compute_error(
                weights, predictions, target
            )  # b - compute the error
            alpha = self._compute_alpha(err)  # c - compute the alpha

            self.estimators.append(estimator)
            self.alphas.append(alpha)

            weights = self._update_weights(
                weights, alpha, predictions, target
            )  # d - update the weights

        return self

    def _predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(dataframe.shape[0])
        for alpha, estimator in zip(self.alphas, self.estimators):
            current_pred = estimator.predict(dataframe)
            predictions += alpha * current_pred

        return np.sign(predictions)
