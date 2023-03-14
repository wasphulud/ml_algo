""" This module contains an implementation of
an abstract class for supervised algorithms.
"""

from abc import ABC, abstractmethod

from typing import TypeVar, Protocol

import pandas as pd

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


class SupervisedTabularDataInterface(Protocol):
    """Generic definition of a supervised tabular algorithm model"""

    def fit(self, dataframe: pd.DataFrame, target: pd.Series) -> None:
        """Fit the model to the data"""

    def predict(self, dataframe: pd.DataFrame) -> None:
        """Predict the target values for the given data"""
