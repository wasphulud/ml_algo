""" This module implements the bagging algorithm for Classification
and Regression.
"""
from multiprocessing import Pool
from typing import List
import pandas as pd
import numpy as np


from abc_models.models import SupervisedTabularDataModel, STMT
from trees.decorators import timer


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
        num_workers: int = 1,
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
        self.num_workers = num_workers
        self.estimators: List[STMT] = []

    @timer
    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "GenericBagging":
        """Fit the model

        Args:
            dataframe (np.array): features
            target (np.array): target
        """
        work = []
        for _ in range(self.n_estimators):
            # bootstrap samples

            dataframe_sample = dataframe.sample(frac=self.max_samples)
            target_sample = target[dataframe_sample.index]
            work.append([dataframe_sample, target_sample])

        # Synchronous Pool Context that will close automatically after use
        with Pool(self.num_workers) as pool:
            self.estimators = pool.starmap(self.model.fit, work)

        return self

    def _predict(self, dataframe: pd.DataFrame) -> pd.Series:
        """Predict using the model

        Args:
            dataframe (np.ndarray): features

        Returns:
            np.ndarray: predictions
        """
        predictions = pd.DataFrame(
            [], columns=[f"estimator {k}" for k in range(1, self.n_estimators + 1)]
        )
        for k, estimator in enumerate(self.estimators):
            prediction = pd.DataFrame(estimator.predict(dataframe))
            predictions[f"estimator {k+1}"] = prediction
        mode = predictions.mode(axis=1)
        mode = mode[[0]]  # keep first column
        mode.columns = ["Predictions"]
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
