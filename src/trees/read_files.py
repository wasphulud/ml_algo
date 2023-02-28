"""This module contains helpers to read files and preprocess the custom datasets.


Functions:
---------
    * read_csv: Reads a csv file and returns a pandas dataframe.
    * preprocess_bmi_dataset: Preprocess the bmi dataset.
    * preprocess_titanic_dataset: Preprocess the titanic dataset.
"""

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """This function reads a csv file and returns a pandas dataframe.

    Args:
        path (str): path to the csv file.

    Returns:
        pd.DataFrame: pandas dataframe.
    """

    return pd.read_csv(path)


def preprocess_bmi_dataset(dataset: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """This function preprocess the bmi dataset.

    Args:
        dataset (pd.DataFrame,): The bmi dataset in a pandas dataframe.
        target_label (str): The target label.

    Returns:
        pd.DataFrame: Preprocessed dataset
    """

    dataset[target_label] = dataset[target_label] >= 4  # .astype("object")
    return dataset


def preprocess_titanic_dataset(
    dataset: pd.DataFrame, target_label: str
) -> pd.DataFrame:
    """This function preprocess the titanic dataset.

    It keeps only 4 features including the labels ("Survived")
    and removes the rows with missing values.

    Args:
        dataset (pd.DataFrame): The full titanic dataset.
        target_label (str): The target label.

    Returns:
        pd.DataFrame: Preprocessed dataset
    """

    dataset = dataset.loc[:, ["Embarked", "Age", "Fare", "Survived"]]
    dataset = dataset.loc[dataset.isna().sum(axis=1) == 0, :]
    dataset[target_label] = dataset[target_label]  # .astype("object")
    return dataset
