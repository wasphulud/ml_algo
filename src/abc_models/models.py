""" This module contains an implementation of
an abstract class for supervised algorithms.
"""

import logging
from abc import ABC, abstractmethod
from typing import TypeVar
import pandas as pd
from sklearn.metrics import classification_report

STMT = TypeVar("STMT", bound="SupervisedTabularDataModel")


class SupervisedTabularDataModel(ABC):
    """Generic definition of a supervised tabular algorithm model"""

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self: STMT, dataframe: pd.DataFrame, target: pd.Series) -> STMT:
        """Fit the model to the data"""
        fitted_model = self._fit(dataframe, target)
        self._is_fitted = True
        return fitted_model

    def predict(self, dataframe: pd.DataFrame) -> pd.Series:
        """Predict the target values for the given data"""
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fit before calling predict"
            )
        return self._predict(dataframe)

    def accuracy(self, dataframe: pd.DataFrame, target: pd.Series) -> float:
        """Compute the accuracy of the model"""
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fit before calling predict"
            )
        return self._accuracy(dataframe, target)

    def _accuracy(self, dataframe: pd.DataFrame, target: pd.Series) -> float:
        """Compute the accuracy of the model"""
        predicted_values = self.predict(dataframe)
        accuracy = sum((target == predicted_values) / len(target)) * 100
        return accuracy

    def report(self, dataframe: pd.DataFrame, target: pd.Series) -> str:
        """Compute the classification report of the model"""
        if not self._is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} must be fit before calling predict"
            )
        return self._report(dataframe, target)

    def _report(self, dataframe: pd.DataFrame, target: pd.Series) -> str:
        """Compute the classification report of the model"""
        logging.warning("Report works only for classification problems")
        predicted_values = self.predict(dataframe)
        report = classification_report(target, predicted_values)
        return report

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted"""
        return self._is_fitted

    @abstractmethod
    def _fit(self: STMT, dataframe: pd.DataFrame, target: pd.Series) -> STMT:
        pass

    @abstractmethod
    def _predict(self, dataframe: pd.DataFrame) -> pd.Series:
        pass
