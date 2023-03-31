""" This module contains the implementation of cross valdiation methods.
We will user sklearn for this
TODO: Add better docstring
"""

import copy
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from common.multiprocessing import NestablePool as Pool


def k_fold_cross_validation(
    model,
    dataframe: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    shuffle=True,
    random_state=None,
    num_processes=1,
) -> float:
    """Perform k-fold cross validation

    Args:
        dataframe (np.array): features
        target (np.array): target
        model (class): model to be used, which have a method .fit and a method .accuracy
        n_splits (int, optional): number of splits. Defaults to 5.

    Returns:
        float: mean accuracy
    """
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if isinstance(dataframe, pd.DataFrame):
        dataframe = dataframe.reset_index(drop=True)
        target = target.reset_index(drop=True)
    accuracies = []
    work = []
    for train_index, test_index in kfold.split(dataframe):
        if isinstance(dataframe, pd.DataFrame):
            x_train, x_test = dataframe.iloc[train_index], dataframe.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        elif isinstance(dataframe, np.ndarray):
            x_train, x_test = dataframe[train_index], dataframe[test_index]
            y_train, y_test = target[train_index], target[test_index]
        work.append([copy.deepcopy(model), x_train, y_train, x_test, y_test])
        # model.fit(x_train, y_train)
        # accuracies.append(model.accuracy(x_test, y_test))
    # return np.mean(accuracies)
    # Synchronous Pool Context that will close automatically after use
    with Pool(num_processes) as pool:
        accuracies = pool.starmap(train_and_get_accuracy, work)
    return np.mean(accuracies)


def train_and_get_accuracy(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model.accuracy(x_test, y_test)
