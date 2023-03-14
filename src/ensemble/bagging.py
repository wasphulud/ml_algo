""" This module implements the bagging algorithm for Classification
and Regression.
"""

from abc_models.models import SupervisedTabularDataModel
import numpy as np

from typing import List


class GenericBagging(SupervisedTabularDataModel):
    """Generic Bagging model for Classification and Regression

    Args:
        SupervisedTabularDataModel (class): abstract class for supervised
        tabular data models
    """

    def __init__(
        self, model, n_estimators=10, max_samples=0.5, max_features=1.0
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
        self.estimators: List[type[model]] = []

    def _fit(self, X, y):
        """Fit the model

        Args:
            X (np.array): features
            y (np.array): target
        """

        for _ in range(self.n_estimators):
            # bootstrap samples

            X_sample = X.sample(frac=self.max_samples)
            y_sample = y[X_sample.index]

            # fit model
            estimator = self.model
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)
        return self

    def _predict(self, X):
        """Predict using the model

        Args:
            X (np.array): features

        Returns:
            np.array: predictions
        """
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(X)
        predictions = pd.DataFrame(predictions)
        mode = predictions.mode(axis=1)
        return mode

    def score(self, X, y):
        """Score the model

        Args:
            X (np.array): features
            y (np.array): target

        Returns:
            float: score
        """
        return np.mean(self.predict(X) == y)
