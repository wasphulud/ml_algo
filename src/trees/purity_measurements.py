""" This module implements the decision tree algorithm for Classification
and Regression.

Usage:
------
    >>> purity_measurements import compute_information_gain
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


def compute_entropy_column(column: pd.Series) -> float:
    """This function returns the entropy of a pandas series

    Args:
        column: The target column.
    Returns:
        float: The entropy of the column.
    """

    # get the probability of each value
    prob = column.value_counts(normalize=True)
    return -sum(prob * np.log2(prob))


def compute_entropy(dataframe: pd.DataFrame) -> pd.Series:
    """This function computes the entropy of each column in a dataframe

    Args:
        dataframe: The dataframe to compute the entropy.
    Returns:
        pd.Series: The entropy of each column in the dataframe.
    """

    return dataframe.apply(compute_entropy_column, axis=1)


def compute_variance(dataframe: pd.DataFrame) -> pd.Series:
    """This function returns the variance of each column in a dataframe

    Args:
        dataframe: The dataframe to compute the variance.
    Returns:
        pd.Series: The variance of each column in the dataframe.
    """

    if len(dataframe) == 1:
        return 0
    return dataframe.var()


# Help create the information gain function


def compute_information_gain(
    column: pd.Series, mask: pd.Series, verbose: bool = False
) -> float:
    """This function returns the information gain of a pandas serie.

    The function computes the information gain of spliting a parent node to two
    children leaves based on a binary mask (defined by the original feature column)
    The function uses the entropy for categorical (objects) columns and the variance for
    numerical columns.

    Args:
        column: The target column, values of the parent node.
        mask: A binary mask defined by the original feature column.
    Returns:
        float: The information gain of the split.

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
            positives * compute_variance(column[mask])
            + negatives * compute_variance(column[~mask])
        )
    return info_gain
