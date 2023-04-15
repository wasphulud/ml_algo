"""The main function of the decision trees package.

It reads the cli arguments and creates a DecisionTree object to train and test the model.

Usage:
    python main.py
        --csv <path_to_csv_file>
        --target_label <target_label>
        --mode <classification/regression>
        --max_depth <max_depth>
        --min_samples_split <min_samples_split>
        --min_information_gain <min_information_gain>
        --preprocess <preprocess>
        --verbose <verbose>
"""
import sys
import logging
from typing import Any

from trees.cli import parse_args
from trees.decision_trees import DecisionTree, DecisionTreeParams
from trees.read_files import (
    read_csv,
    preprocess_bmi_dataset,
    preprocess_titanic_dataset,
)
from ensemble.random_forest import RandomForest


LOGGING_LEVEL = logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args: list[Any]) -> None:
    """main function of the module

    Args:
        args: arguments to be parsed see cli.py

    Raises:
        ValueError: if preprocess argument is not recognized
    """
    args_namespace = parse_args(args)
    arguments = vars(args_namespace)
    logging.info("cli arguments: %s", arguments)
    decision_tree_params = DecisionTreeParams(
        max_depth=arguments["max_depth"],
        min_samples_split=arguments["min_samples_split"],
        min_information_gain=arguments["min_information_gain"],
        mode=arguments["mode"],
    )
    decision_tree = DecisionTree(
        decision_tree_params=decision_tree_params,
        verbose=arguments["verbose"],
    )

    random_forest = RandomForest(
        decision_tree_params=DecisionTreeParams(
            max_depth=10, mode=arguments["mode"], turn_off_frac=0.3
        ),
        n_estimators=100,
        max_samples_frac=0.8,
        max_features_frac=1,
        num_processes=10,
    )

    if arguments["csv"]:
        data = read_csv(arguments["csv"])
        # check for the preprocessing if it exists and apply it
        if arguments["preprocess"] == "bmi":
            data = preprocess_bmi_dataset(data, arguments["target_label"])
            data["Index"] = data["Index"] * 1
            data["Index"] = data["Index"].astype("bool")
        elif arguments["preprocess"] == "titanic":
            data = preprocess_titanic_dataset(data, arguments["target_label"])
        elif arguments["preprocess"] not in ["", "bmi", "titanic"]:
            logging.warning("preprocess argument not recognized")
            raise ValueError(
                f"preprocess argument not recognized {arguments['preprocess']}"
            )
        training_set = data.sample(
            frac=0.8, random_state=None
        )  # pylint: disable=maybe-no-member
        test_set = data.drop(training_set.index)  # pylint: disable=maybe-no-member
        decision_tree.fit(
            training_set.drop(arguments["target_label"], axis=1),
            training_set[arguments["target_label"]],
        )
        logging.info(
            "The decision tree model accuracy is: %.2f%%",
            decision_tree.accuracy(test_set, test_set[arguments["target_label"]]),
        )
        logging.info(
            "The features importance is is: %s",
            decision_tree.get_features_relevance,
        )
        if arguments["mode"] == "classification":
            logging.info(
                "Binary Classification Report for decision tree \n%s",
                decision_tree.report(test_set, test_set[arguments["target_label"]]),
            )
        random_forest.fit(
            training_set.drop(arguments["target_label"], axis=1),
            training_set[arguments["target_label"]].astype("object"),
        )
        logging.info(
            "The random forest model accuracy is: %.2f%%",
            random_forest.accuracy(test_set, test_set[arguments["target_label"]]),
        )
        if arguments["mode"] == "classification":
            logging.info(
                "Binary Classification Report for random forest \n%s",
                random_forest.report(test_set, test_set[arguments["target_label"]]),
            )
        print(random_forest.cumaccuracy(test_set, test_set[arguments["target_label"]]))

    else:
        logging.info("No csv file provided")
    logging.info("Goodbye!")


if __name__ == "__main__":
    main(sys.argv[1:])
