import logging
import itertools

import pandas as pd
import numpy as np

from purity_measurements import compute_information_gain

# implement the decision tree class


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int or None = None,
        min_information_gain: float = 1e-10,
        mode: str = "classification",
        verbose: str = False,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.mode = mode
        self.verbose = verbose
        self.tree = dict
        self.target = None

    def _init_target(self, target: str) -> None:
        self.target = target

    def train(self, dataframe: pd.DataFrame, target: str):
        self._init_target(target)
        self.tree = self._build_tree(dataframe, self.max_depth)
        return self.tree

    def _validate_dataframe(self, dataframe: pd.DataFrame, max_depth: int) -> tuple:
        """
        This function validates the dataframe
        """
        if dataframe.shape[0] == 0:
            return False, None, " [OUT] --> Empty dataframe"

        # check if the dataframe is pure
        if dataframe[self.target].nunique() == 1:
            return (
                False,
                dataframe,
                " [OUT] --> Pure dataframe",
            )

        if (
            self.min_samples_split is not None
            and dataframe.shape[0] < self.min_samples_split
        ):
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

    def _build_tree(self, dataframe: pd.DataFrame, max_depth: int) -> dict:

        if self.verbose:
            logging.debug("Current depth: %s", max_depth)
            logging.debug("Current dataframe shape: %s", dataframe.shape)

        # validate the dataframe
        validation, validation_dataframe, validation_message = self._validate_dataframe(
            dataframe, max_depth
        )

        if not validation:
            if self.verbose:
                logging.debug(validation_message)
            if validation_dataframe is None:
                return None
            return self.compute_leaf_value(dataframe[self.target])

        # get the best split
        (
            split_variable,
            split_value,
            split_info_gain,
            split_is_categorical,
        ) = get_best_split(dataframe, self.target, self.verbose)

        if split_info_gain < self.min_information_gain:
            if self.verbose:
                logging.debug(" [OUT] --> Min information gain reached")
            return self.compute_leaf_value(dataframe[self.target])

        if self.verbose:
            logging.debug(" --> Best split: %s", split_variable)
            logging.debug(" --> Best split value: %s", split_value)
            logging.debug(" --> Best split info gain: %s", split_info_gain)
            logging.debug(" --> Best split is categorical: %s", split_is_categorical)
        # split the dataframe
        left_data, right_data = split_data_node(dataframe, split_variable, split_value)

        left_response = self._build_tree(left_data, max_depth - 1)
        right_response = self._build_tree(right_data, max_depth - 1)

        if left_response == right_response:
            if self.verbose:
                logging.debug(" [OUT] --> Left and right responses are the same")
            return left_response

        # create the decision tree
        decision_tree = {
            "split_variable": split_variable,
            "split_value": split_value,
            "split_info_gain": split_info_gain,
            "split_is_categorical": split_is_categorical,
            "left": left_response,
            "right": right_response,
        }
        return decision_tree

    def compute_leaf_value(self, serie: pd.Series) -> str:
        """
        This function returns the prediction for a given serie
        """
        if self.mode == "classification":
            return serie.value_counts().idxmax()
        if self.mode == "regression":
            return serie.mean()
        raise ValueError("The mode must be either classification or regression")

    def infer_one_entry(self, sample: pd.Series, decision_tree: dict) -> str:
        """
        This function returns the prediction for a given sample
        """

        # check if the node is a leaf

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
                return self.infer_one_entry(sample, decision_tree["left"])
            return self.infer_one_entry(sample, decision_tree["right"])
        if sample[split_variable] < split_value:
            return self.infer_one_entry(sample, decision_tree["left"])
        return self.infer_one_entry(sample, decision_tree["right"])

    def infer_sample(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        This function returns the prediction for a given dataframe
        """
        return dataframe.apply(self.infer_one_entry, args=(self.tree,), axis=1)


def split_data_node(dataframe: pd.DataFrame, split_variable: str, split_value) -> tuple:
    """
    This function returns the split dataframe based on a variable
    """
    split_is_categorical = dataframe[split_variable].dtype == "O"
    if split_is_categorical:
        mask = dataframe[split_variable].isin(split_value)
    else:
        mask = dataframe[split_variable] < split_value
    return dataframe[mask], dataframe[~mask]


def get_best_split(
    dataframe: pd.DataFrame, target: str, verbose: bool = False
) -> tuple:
    """
    This function returns the best split for a given dataframe
    """

    info_gain_recap = (
        dataframe.drop(target, axis=1)
        .apply(get_best_split_feature, target=dataframe[target], verbose=verbose)
        .reset_index(drop=True)
    )

    if info_gain_recap.iloc[-1].sum() == 0:
        return [None] * 4

    info_gain_recap = info_gain_recap.loc[:, info_gain_recap.iloc[-1, :]]
    split_variable = info_gain_recap.iloc[1].astype(np.float64).idxmax()
    split_value = info_gain_recap[split_variable][0]
    split_info_gain = info_gain_recap[split_variable][1]
    split_is_categorical = info_gain_recap[split_variable][2]

    return split_variable, split_value, split_info_gain, split_is_categorical


def get_best_split_feature(
    feature: pd.Series, target: pd.Series, verbose: bool = False
) -> tuple:
    """
    This function returns the best split for a given feature
    """
    # check if the column's type os categorical
    is_cat = feature.dtype == "O"
    if verbose:
        logging.debug("Feature name ---> %s", feature.name)
        logging.debug("Is categorical? ---> %s", is_cat)

    info_gain = []
    split_value = []

    if is_cat:
        splits = get_cat_combinations(feature)
    else:
        splits = feature.sort_values().unique()[1:]

    for split in splits:
        mask = feature.isin(split) if is_cat else feature < split
        info_gain_split = compute_information_gain(target, mask, is_cat)
        split_value.append(split)
        info_gain.append(info_gain_split)

    if len(info_gain) == 0:
        if verbose:
            logging.debug(" --> No information gain")
        return None, None, is_cat, False

    best_info_gain = max(info_gain)
    best_split = split_value[info_gain.index(max(info_gain))]

    if verbose:
        logging.debug(" --> Best info gain: %s", best_info_gain)
        logging.debug(" ----> Best split: %s", best_split)

    return best_split, best_info_gain, is_cat, True


def get_cat_combinations(feature: pd.Series) -> list:
    """
    This function returns all the possible combinations of a categorical variable
    """
    feature = feature.unique()

    options = []
    for length in range(0, len(feature) + 1):
        for subset in itertools.combinations(feature, length):
            options.append(list(subset))
    return options[1:-1]  # remove the empty list and the full list


def dirty_excecution():
    from pprint import pprint

    data = pd.read_csv("../../../data/data.csv")
    data["obese"] = (data.Index >= 4).astype("int")
    data.drop("Index", axis=1, inplace=True)
    # data['bmi'] = data['Weight'] / (data['Height'] ) ** 2
    # print(data.shape)
    d_tree = DecisionTree(max_depth=1000, min_samples_split=2, verbose=False)
    d_tree.train(data, "obese")
    pprint(d_tree.tree)
    print(pd.concat([d_tree.infer_sample(data), data["obese"]], axis=1))
    print(sum(d_tree.infer_sample(data) == data["obese"]) / data.shape[0])

    print("########### TITANIC NOW")
    titanic = pd.read_csv("../../../data/titanic.csv")

    titanic_lite = titanic.loc[:, ["Embarked", "Age", "Fare", "Survived"]]
    titanic_lite = titanic_lite.loc[titanic_lite.isna().sum(axis=1) == 0, :]
    d_tree_titanic = DecisionTree(max_depth=10, min_samples_split=1, verbose=True)
    d_tree_titanic.train(titanic_lite.iloc[:300], "Survived")
    pprint(d_tree_titanic.tree)
    # print(d_tree_titanic.infer_sample(titanic_lite.iloc[500:]))
    print(
        sum(
            d_tree_titanic.infer_sample(titanic_lite.iloc[400:])
            == titanic_lite.iloc[400:]["Survived"]
        )
        / titanic_lite.iloc[400:].shape[0]
    )


dirty_excecution()
