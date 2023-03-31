""" This module implements the bagging algorithm for Classification
and Regression.
"""
from multiprocessing import Pool
from typing import List
import copy
import pandas as pd
import numpy as np


from abc_models.models import SupervisedTabularDataModel, STMT
from common.decorators import timer


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
        max_samples_frac: float = 0.5,
        max_features_frac: float = 1.0,
        num_processes: int = 1,
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
        self.max_samples_frac = max_samples_frac
        self.max_features_frac = max_features_frac
        self.num_processes = num_processes
        self.estimators: List[STMT] = []

    @timer
    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "GenericBagging":
        """Fit the model

        Args:
            dataframe (np.array): features
            target (np.array): target
        """
        work = []
        n_samples = dataframe.shape[0]
        for _ in range(self.n_estimators):
            # bootstrap samples
            dataframe_sample = dataframe.sample(
                frac=self.max_samples_frac, replace=False
            )
            dataframe_sample = dataframe_sample.sample(n=n_samples, replace=True)

            # bootstrap features
            if self.max_features_frac < 1:
                dataframe_sample = dataframe_sample.sample(
                    frac=self.max_features_frac, axis=1
                )
            target_sample = target[dataframe_sample.index]
            work.append([dataframe_sample, target_sample])

        # Synchronous Pool Context that will close automatically after use
        with Pool(self.num_processes) as pool:
            self.estimators = pool.starmap(copy.deepcopy(self.model.fit), work)

        return self

    def _single_estimator_prediction(
        self, estimator: STMT, dataframe: pd.DataFrame
    ) -> pd.Series:
        """Predict using the model

        Args:
            dataframe (np.ndarray): features

        Returns:
            np.ndarray: predictions
        """
        return estimator.predict(dataframe)

    def _intermediate_predictions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """predictions from all the estiamtors

        Args:
            dataframe (np.ndarray): features

        Returns:
            np.ndarray: predictions
        """
        predictions = pd.DataFrame(
            [], columns=[f"estimator {k}" for k in range(1, self.n_estimators + 1)]
        )
        work = map(lambda model: [model, dataframe], self.estimators)
        with Pool(self.num_processes) as pool:
            predictions_list = pool.starmap(
                self._single_estimator_prediction, list(work)
            )

        for k, prediction in enumerate(predictions_list):
            predictions[f"estimator {k+1}"] = prediction
        return predictions

    def _predict(self, dataframe: pd.DataFrame) -> pd.Series:
        """Predict using the model

        Args:
            dataframe (np.ndarray): features

        Returns:
            np.ndarray: predictions
        """
        predictions = self._intermediate_predictions(dataframe)
        mode = predictions.mode(axis=1)
        mode = mode[[0]]  # keep first column
        mode.columns = ["Predictions"]
        return mode.squeeze()

    @timer
    def cumaccuracy(self, dataframe: pd.DataFrame, target: pd.Series) -> List[float]:
        """Compute the cumulative accuracy of the model
        using the predictions of the k-first estimators
        """
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fit before calling predict"
            )
        predictions = self._intermediate_predictions(dataframe)
        # for each estimator, compute the prediction using all the previous estimators
        modes = [
            predictions.iloc[:, :k].mode(axis=1)[0]
            for k in range(1, len(predictions.columns) + 1)
        ]
        accuracies = [
            float(f"{sum((target == mode) / len(target)) * 100:.2f}") for mode in modes
        ]

        return accuracies
