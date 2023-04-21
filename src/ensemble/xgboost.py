"""
Extreme Boosting Algprithm
"""

from typing import List
import pandas as pd
import numpy as np


from abc_models.models import SupervisedTabularDataModel

from trees.decision_trees import DecisionTree


class XGBClassifier(SupervisedTabularDataModel):
    def __init__(
        self,
        n_estimators,
        decision_tree_params,
        learning_rate: float = 0.1,
        subsample_frac: float = 1,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators: list[list[DecisionTree]] = [[]]
        self.decision_tree_params = decision_tree_params
        self.decision_tree_params.mode = "regression"
        self.subsample_frac = subsample_frac
