""" This module contains an implementation of
an abstract class for supervised algorithms.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

STMT = TypeVar("STMT", bound="SupervisedTabularDataModel")


class SupervisedTabularDataModel(ABC):
    """Generic definition of a tree algorithm model"""

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self: STMT, dataframe: np.ndarray, target: np.ndarray) -> STMT:
        """Fit the model to the data"""
        fitted_model = self._fit(dataframe, target)
        self._is_fitted = True
        return fitted_model

    def predict(self, dataframe: np.ndarray) -> np.ndarray:
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
    def _fit(self: STMT, dataframe: np.ndarray, target: np.ndarray) -> STMT:
        pass

    @abstractmethod
    def _predict(self, dataframe: np.ndarray) -> np.ndarray:
        pass
