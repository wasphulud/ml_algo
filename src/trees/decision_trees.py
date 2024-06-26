# pylint: disable=fixme
""" This module implements the decision tree algorithm for Classification
and Regression.

Usage:
------
    >>> import pandas as pd
    >>> from trees.decision_trees import DecisionTree, DecisionTreeParams
    >>> data = pd.read_csv("../../data/bmi.csv")
    >>> data["Index"] = data["Index"] >= 4
    >>> training_set = data.sample(frac=0.8, random_state=42)
    >>> test_set = data.drop(training_set.index)
    >>> decision_tree_params = DecisionTreeParams()
    >>> decision_tree = DecisionTree(decision_tree_params=decision_tree_params, verbose=True)
    >>> tree = decision_tree.fit(data, "Index")
    >>> predicted_values = decision_tree.predict(test_set)

Classes:
--------
    DecisionTree: A class which its instances trains a decision tree and infer
        new values for a given sample.

Functions:
----------
    * get_best_split(pd.DataFrame, str) -> tuple[str, float, float]
        returns the best split informations for a given dataframe and a target label
    * get_best_split_feature(pd.Series, pd.Series) -> tuple[str, float, bool, bool]
        returns the best split informations for a given column and the labels
    * get_categorical_combinations(pd.Series) -> list[tuple]
        returns all the possible combinations of a categorical column
    * split_data_node(pd.DataFrame, str, str/float) -> tuple[pd.DataFrame, pd.DataFrame]
        splits the dataframe using the split informations.

Imported Classes:
-------------------
    abc_models.models.SupervisedTabularDataModel
    trees.exceptions.UnsupportedModeError
Imported Functions:
-------------------
    compute_information_gain(pd.Series, pd.Series, bool, bool) -> float

TODO:
    * move to monorepo ?
    * Compute accuracy
    * Model the tree as class on its own, or dataclass.
    * implement grid search ?
    * implement cross validation ?
    * Add the Gini Index -> Add the CART algorithm ?
    * Add the pruning algorithm ??
    * Add the random forest algorithm
    * Add the gradient boosting algorithm ??
    * Add the XGBoost algorithm
    * Add the LightGBM algorithm ??
    * Add the CatBoost algorithm ??
    * Add the AdaBoost algorithm ??
    * Add the Stacking algorithm ??
    * Add the Bagging algorithm ??
    * Add the Voting algorithm ??
"""

import logging
import itertools
import math

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from collections import defaultdict

from abc_models.models import SupervisedTabularDataModel
from trees.purity_measurements import compute_information_gain
from trees.exceptions import UnsupportedModeError

from common.decorators import timer


@dataclass
class DecisionTreeParams:
    """This class contains the inputs for the decision tree algorithm.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples
            required to split an internal node.
        min_information_gain (float): The minimum information gain
            required to split an internal node.
        mode (str): The mode of the tree. Either classification or regression.
    """

    max_depth: int = 5
    min_samples_split: int = 10
    min_information_gain: float = 1e-10
    mode: str = "classification"
    turn_off_frac: float = 0.0


class DecisionTree(SupervisedTabularDataModel):
    """This class implements the decision tree algorithm for Classification
    and Regression.

    This class allows to initiale and train a tree based on the decision tree
    algorithm. The training is based on maximizing the information gain at each
    level by reducing the impurity of each node using on the entropy.
    [TODO] Gini Index is NOT used as of now.

    Attributes:
        decision_tree_params (DecisionTreeParams): The inputs dataclass for the decision
        verbose (bool): If True, the tree will print out the information
            about the training.


    Methods: TODO: these public methods are now depricated.
        train(dataframe: pd.DataFrame, target: str) -> dict
            This function trains the tree.
        infer(dataframe: pd.DataFrame, decision_tree: dict) -> pd.Series
            This function returns the prediction for a given pandas dataframe
            sample.
    """

    tree: dict[str, dict] = {}
    _target_label: str = ""
    _splits: dict[str, list] = defaultdict(list)

    # initialize the class
    def __init__(
        self,
        decision_tree_params: DecisionTreeParams = DecisionTreeParams(),
        verbose: Optional[bool] = False,
    ):
        super().__init__()
        self.max_depth = decision_tree_params.max_depth
        self.min_samples_split = decision_tree_params.min_samples_split
        self.min_information_gain = decision_tree_params.min_information_gain
        self.mode = decision_tree_params.mode
        self.turn_off_frac = decision_tree_params.turn_off_frac
        self._check_mode()  # check if the mode is supported
        self.verbose = verbose
        self.leaves: list[dict] = []

    def _check_mode(self) -> None:
        """This private method checks if the mode is supported."""
        if self.mode not in ["classification", "regression"]:
            raise UnsupportedModeError(
                "The mode is not supported. Please choose between classification and"
                " regression"
            )

    def _init_target_label(self, target: str) -> None:
        """This private method initializes the target feature

        Args:
            target (str): target's feature name
        """

        self._target_label = target

    def _cast_target_label(self, target: pd.Series) -> None:
        """This private method casts the target feature to the relevant type.

        If the mode is classification, the target is casted to object.
        If the mode is regression, the target is casted to float.


        Args:
            dataframe (pd.DataFrame): training dataset
        """
        target_type = target.dtype
        if self.mode == "classification":
            if target_type != "object":
                logging.warning(
                    "target column is not of type object and is %s ---> casting to"
                    " object",
                    target_type,
                )
            target = target.astype("object")
            return
        if target_type == "object":
            logging.warning(
                "target column is type object and is where it shoult                   "
                " be float/float32/float64 ---> casting to float32"
            )
        target = target.astype("float32")

    @timer
    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "DecisionTree":
        """This method trains the decision tree using the input dataframe.

        Args:
            dataframe (pd.DataFrame): training dataset
            target (str): target's feature name

        Returns:
            dict: The decision tree
        """
        target_label = target.name
        self._init_target_label(target_label)
        self._cast_target_label(target)
        # self._splits = self._splits.fromkeys(list(dataframe.columns), [])
        self._splits = defaultdict(list)
        self.tree = self._build_tree(
            dataframe, target, self.max_depth, self.turn_off_frac
        )
        return self

    def _validate_dataframe(
        self, dataframe: pd.DataFrame, target: pd.Series, max_depth: int
    ) -> tuple[bool, pd.DataFrame, str]:
        """This function validates the dataframe

        Args:
            dataframe (pd.DataFrame): The dataframe to validate.
            max_depth (int): The maximum depth allowed for the current subtree.

        Returns:
            tuple(bool, pd.DataFrame, str): A tuple containing:
                bool: is a valid dataframe?
                pd.DataFrame: the dataframe.
                str: a string message explaining why the dataframe is not Valid.
        """
        if dataframe.shape[0] == 0:
            return False, None, " [OUT] --> Empty dataframe"

        # check if the dataframe is pure
        if target.nunique() == 1:
            return (
                False,
                dataframe,
                " [OUT] --> Pure dataframe",
            )

        if dataframe.shape[0] < self.min_samples_split:
            return (
                False,
                dataframe,
                " [OUT] --> Min Samples Split reached",
            )
        # check if the max depth is reached
        if max_depth == 0:
            return (
                False,
                dataframe,
                " [OUT] --> Max depth reached",
            )
        return True, dataframe, " [IN] --> Valid dataframe"

    def _build_tree(
        self,
        dataframe: pd.DataFrame,
        target: pd.Series,
        max_depth: int,
        turn_off_frac: float = 0,
    ) -> dict:
        """This function builds the decision tree

        This function builds the decision tree by recursively calling itself
        until a termination condition is met. The recursiveness is done on
        the children nodes, which are created by using the best split feature
        that maximized the information gain wrt the parent node.

        Args:
            dataframe (pd.DataFrame): dataframe used to train the tree
            max_depth (int): maximum depth of the tree

        Returns:
            dict: decision tree
        """

        # TODO: fix too many local variables error
        if self.verbose:
            logging.debug("Current depth: %s", max_depth)
            logging.debug("Current dataframe shape: %s", dataframe.shape)

        # validate the dataframe
        validation, validation_dataframe, validation_message = self._validate_dataframe(
            dataframe, target, max_depth
        )

        # if the dataframe is not valid then it's a leaf and compute its value.
        if not validation:
            if self.verbose:
                logging.debug(validation_message)
            if validation_dataframe is None:
                return {}
            return self._compute_leaf_value(target)

        # get the best split
        (
            split_variable,
            split_value,
            split_info_gain,
            split_is_categorical,
        ) = get_best_split(dataframe, target, turn_off_frac, self.verbose)

        # Is the information gain termination gain is met ?
        if split_info_gain is None or split_info_gain < self.min_information_gain:
            if self.verbose:
                logging.debug(" [OUT] --> Min information gain reached")
            return self._compute_leaf_value(target)

        if self.verbose:
            logging.debug(" --> Best split: %s", split_variable)
            logging.debug(" --> Best split value: %s", split_value)
            logging.debug(" --> Best split info gain: %s", split_info_gain)
            logging.debug(" --> Best split is categorical: %s", split_is_categorical)

        # register the best split
        self._register_split(split_variable, split_info_gain)

        # split the dataframe using the variable and its value to two children
        left_data, left_target, right_data, right_target = split_data_node(
            dataframe, target, split_variable, split_value
        )

        # compute the two children subtrees recursively
        left_response = self._build_tree(
            left_data, left_target, max_depth - 1, turn_off_frac
        )
        right_response = self._build_tree(
            right_data, right_target, max_depth - 1, turn_off_frac
        )

        # if both children have the same outcome then the current node is a
        # leaf
        if left_response == right_response:
            if self.verbose:
                logging.debug(" [OUT] --> Left and right responses are the same")
            return left_response

        # The final decision tree
        decision_tree = {
            "split_variable": split_variable,
            "split_value": split_value,
            "split_info_gain": split_info_gain,
            "split_is_categorical": split_is_categorical,
            "left": left_response,
            "right": right_response,
        }
        return decision_tree

    def _compute_leaf_value(self, serie: pd.Series) -> dict:
        """This function returns the prediction for a given serie living in a single leaf

        Args:
            serie (pd.Series): The values from which to compute the node's value.

        Raises:
            ValueError: If the mode is not either classification or regression.

        Returns:
            str: The prediction for the given serie:
                    * the mathematical mode if classification
                    * the mean of the serie if regression
        """

        if self.mode == "classification":
            leaf = {
                "is_leaf": serie.value_counts().idxmax(),
                "target_indexes": serie.index.tolist(),
            }
            self.leaves.append(leaf)
            return leaf
        if self.mode == "regression":
            leaf = {"is_leaf": serie.mean(), "target_indexes": serie.index.tolist()}
            self.leaves.append(leaf)
            return leaf
        raise ValueError("The mode must be either classification or regression")

    def _infer_one_entry(self, sample: pd.Series, decision_tree: dict) -> str:
        """This function returns the prediction for a given sample

        Args:
            sample (pd.Series): The sample for which the user wants to predict the value.
            decision_tree (dict): the decision tree.

        Returns:
            str: the prediction for the given entry
        """

        # check if the node is a leaf
        if "is_leaf" in decision_tree:
            return decision_tree["is_leaf"]
        if not isinstance(decision_tree, dict):
            return decision_tree
        # get the split variable
        split_variable = decision_tree["split_variable"]
        # get the split value
        split_value = decision_tree["split_value"]
        # get the split type
        split_is_categorical = decision_tree["split_is_categorical"]
        # check the split condition
        if split_is_categorical:
            if sample[split_variable] in split_value:
                return self._infer_one_entry(sample, decision_tree["left"])
            return self._infer_one_entry(sample, decision_tree["right"])
        # remove the brackets and access the value
        split_value = split_value[0]
        if sample[split_variable] < split_value:
            return self._infer_one_entry(sample, decision_tree["left"])
        return self._infer_one_entry(sample, decision_tree["right"])

    def _predict(self, dataframe: pd.DataFrame) -> pd.Series:
        """This function returns the predictions for a given dataframe

        Args:
            dataframe (pd.DataFrame): the dataframe for which the user wants
            to predict the values.

        Returns:
            pd.Series: the prediction for the given dataframe
        """

        return dataframe.apply(self._infer_one_entry, args=(self.tree,), axis=1)

    def _register_split(self, split_variable: str, split_info_gain: float):
        """This function registers the split variable and its information gain

        Args:
            split_variable (str): the split variable
            split_info_gain (float): the split information gain
        """
        self._splits[split_variable].append(split_info_gain)

    @property
    def get_features_relevance(self) -> dict:
        """This function returns the feature relevance for the decision tree.

        Returns:
            dict: a dictionary containing the feature relevance
        """
        return self._compute_feature_relevance()

    def _compute_feature_relevance(self) -> dict:
        """This function computes the feature relevance for the decision tree.

        Returns:
            dict: a dictionary containing the feature relevance
        """
        feature_relevance = {}
        for feature, information_gain in self._splits.items():
            feature_relevance[feature] = np.sum(information_gain)
        return feature_relevance


def split_data_node(
    dataframe: pd.DataFrame,
    target: pd.Series,
    split_variable: str,
    split_value: str,  # the split value is not cesea=ssarily a string FIXME
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """This function returns the split dataframe based on a variable

    Args:
        dataframe (pd.DataFrame): dataset we want to split
        split_variable (str): the feature name to use to split the dataset
        split_value (_type_): the feature value to use to split the dataset

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): A tuple containing two pandas
        dataframes
    """

    split_is_categorical = dataframe[split_variable].dtype == "O"
    if split_is_categorical:
        mask = dataframe[split_variable].isin(split_value)
    else:
        split_value = split_value[0]
        mask = dataframe[split_variable] < split_value
    return dataframe[mask], target[mask], dataframe[~mask], target[~mask]


def get_best_split(
    dataframe: pd.DataFrame,
    target: pd.Series,
    turn_off_frac: float = 0,
    verbose: Optional[bool] = False,
) -> tuple[str, str, float, bool]:
    """This function returns the best split for a given dataframe.

    Given a dataset and the target feature name, we compute all the possible
    splits and measure the information gain for each split. The split with the
    highest information gain value is kept and returned.

    Args:
        dataframe (pd.DataFrame): the dataset to compute the best split for.
        target_name (str): Target feature's name
        verbose (bool, optional): Verbose flag. Defaults to False.

    Returns:
        tuple(str, str, float, bool): A tuple containing:
            * the best split variable name
            * the best split value: it can be either a float or a string
            * the best split information gain value
            * the best split variable type is categorical
    """
    if turn_off_frac:
        dataframe = dataframe.sample(frac=1 - turn_off_frac, axis=1)

    info_gain_recap = dataframe.apply(
        get_best_split_feature, target=target, verbose=verbose
    ).reset_index(drop=True)
    if info_gain_recap.iloc[-1].sum() == 0:
        return ("", "", -math.inf, False)

    info_gain_recap = info_gain_recap.loc[:, info_gain_recap.iloc[-1, :]]
    split_variable = info_gain_recap.iloc[1].astype(np.float64).idxmax()
    split_value = info_gain_recap[split_variable][0]
    split_info_gain = info_gain_recap[split_variable][1]
    split_is_categorical = info_gain_recap[split_variable][2]

    return split_variable, split_value, split_info_gain, split_is_categorical


def get_best_split_feature(
    feature: pd.Series, target: pd.Series, verbose: bool = False
) -> tuple[list, float, bool, bool]:
    """This function returns the best split for a given feature

    Args:
        feature (pd.Series): the feature to compute the best split for.
        target (pd.Series): the target values.
        verbose (bool, optional): verbosity flag. Defaults to False.

    Returns:
        tuple(list, float, bool, bool): A tuple containing:
            * the best split value: it can be either a float or a string
            * the best split information gain
            * if the best split variable type is catagorical
            * a boolean indicating if the split is valid or not
    """

    assert isinstance(feature, pd.Series), f"Is not a pandas series, {type(feature)}"
    # check if the column's type os categorical
    is_cat = feature.dtypes == "O"
    if verbose:
        logging.debug("Feature name ---> %s", feature.name)
        logging.debug("Is categorical? ---> %s", feature.dtype == "O")

    info_gain = []
    split_value = []

    if is_cat:
        splits = get_categorical_combinations(feature)
    else:
        splits = feature.sort_values().unique()[1:]

    for split in splits:
        mask = feature.isin(split) if is_cat else feature < split
        info_gain_split = compute_information_gain(target, mask)
        split_value.append(split)
        info_gain.append(info_gain_split)

    if len(info_gain) == 0:
        if verbose:
            logging.debug(" --> No information gain")
        return [""], -math.inf, is_cat, False

    best_split_info_gain = max(info_gain)
    best_split_value = split_value[info_gain.index(best_split_info_gain)]

    if verbose:
        logging.debug(" --> Best info gain: %s", best_split_info_gain)
        logging.debug(" ----> Best split: %s", best_split_value)
    if not is_cat:
        best_split_value = [best_split_value]
    return best_split_value, best_split_info_gain, is_cat, True


def get_categorical_combinations(feature: pd.Series) -> list[list]:
    """This function returns the possible combinations of a categorical variable

    The function will return all the possible combinations except the empty
    and the full subsets.

    Args:
        feature (pd.Series): the column dataset for which to compute the combinations

    Returns:
        list[tuple]: the list of all the possible combinations.
    """

    feature = feature.unique()

    options = []
    for length in range(0, len(feature) + 1):
        for subset in itertools.combinations(feature, length):
            options.append(list(subset))
    return options[1:-1]  # remove the empty and the full lists
