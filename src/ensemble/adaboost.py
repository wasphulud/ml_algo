""" This module implements the adaboost algorithm
"""
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
            training_dataframe = copy.deepcopy(dataframe.loc[indices])
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
