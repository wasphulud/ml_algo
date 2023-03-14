""" This module implements the bagging algorithm for Classification
and Regression.
"""
from typing import List, Type
import pandas as pd
import numpy as np

from abc_models.models import SupervisedTabularDataModel, STMT


class GenericBagging(SupervisedTabularDataModel):
    """Generic Bagging model for Classification and Regression

    Args:
        SupervisedTabularDataModel (class): abstract class for supervised
        tabular data models
    """

    def __init__(
        self,
        model: STMT,
        n_estimators: int = 10,
        max_samples: float = 0.5,
        max_features: float = 1.0,
    ) -> None:
        """Constructor of the class

        Args:
            model (class): model to be used as base estimator
            n_estimators (int, optional): number of base estimators. Defaults to 10.
            max_samples (float, optional): number of samples to draw from X to train
            each base estimator. Defaults to 1.0.
            max_features (float, optional): number of features to draw from X to train
            each base estimator. Defaults to 1.0.
        """
        super().__init__()
        self.model = model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.estimators: List[STMT] = []

    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "GenericBagging":
        """Fit the model

        Args:
            dataframe (np.array): features
            target (np.array): target
        """

        for _ in range(self.n_estimators):
            # bootstrap samples

            dataframe_sample = dataframe.sample(frac=self.max_samples)
            target_sample = target[dataframe_sample.index]

            # fit model
            estimator = self.model
            estimator = estimator.fit(dataframe=dataframe_sample, target=target_sample)
            self.estimators.append(estimator)
        return self

    def _predict(self, dataframe: pd.DataFrame) -> np.Series:
        """Predict using the model

        Args:
            dataframe (np.ndarray): features

        Returns:
            np.ndarray: predictions
        """
        predictions = np.zeros((dataframe.shape[0], len(self.estimators)))
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(dataframe)
        predictions = pd.DataFrame(predictions)
        mode = predictions.mode(axis=1)
        return mode

    def score(self, dataframe: pd.DataFrame, target: pd.Series) -> float:
        """Score the model

        Args:
            X (np.array): features
            y (np.array): target

        Returns:
            float: score
        """
        return np.mean(self.predict(dataframe) == target)
