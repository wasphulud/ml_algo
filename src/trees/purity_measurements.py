""" This module implements the decision tree algorithm for Classification
and Regression.

Usage:
------
    >>> trees.purity_measurements import compute_information_gain
    >>> info_gain_split = compute_information_gain(target, mask, target.dtype == "O")


Functions:
----------
    * compute_entropy_column: Computes the entropy of a pandas series
    * compute_entropy: Computes the entropy of each column in a dataframe
    * compute_variance: Computes the variance of each column in a dataframe
    * compute_information_gain: Computes the information gain of a pandas serie
            based on a binary mask (defined by the original feature column)
"""

import logging

import pandas as pd
import numpy as np

LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_entropy_column(column: pd.Series) -> float:
    """This function returns the entropy of a pandas series"""
    prob = column.value_counts(normalize=True)  # get the probability of each value
    return -sum(prob * np.log2(prob))


def compute_entropy(dataframe: pd.DataFrame) -> pd.Series:
    """
    The entropy function return the entropy of each column in a dataframe"""
    return dataframe.apply(compute_entropy_column, axis=1)


def compute_variance(dataframe: pd.DataFrame) -> pd.Series:
    """
    The variance function return the variance of each column in a dataframe"""

    if len(dataframe) == 1:
        return 0
    return dataframe.var()


# Help create the information gain function


def compute_information_gain(
    column: pd.Series, mask: pd.Series, verbose: bool = False) -> float:
    """ This function returns the information gain of a pandas serie.
    
    The function computes the information gain of spliting a parent node to two
    children leaves based on a binary mask (defined by the original feature column)
    """

    if mask.shape[0] == 0:
        return 0
    positives = sum(mask) / mask.shape[0]
    negatives = sum(~mask) / mask.shape[0]
    # check if the column's type os categorical
    bool_category = column.dtype == "O"
    if bool_category:
        if verbose:
            logging.debug("The target column is categorical")
        info_gain = compute_entropy_column(column) - (
            positives * compute_entropy_column(column[mask])
            + negatives * compute_entropy_column(column[~mask])
        )
    else:  # if the column is numerical
        if verbose:
            logging.debug("The target column is numerical")
        info_gain = compute_variance(column) - (
            positives * compute_variance(column[mask]) + negatives * compute_variance(column[~mask])
        )
    return info_gain
