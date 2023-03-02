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

from trees.cli import parse_args
from trees.decision_trees import DecisionTree, DecisionTreeParams
from trees.read_files import (
    read_csv,
    preprocess_bmi_dataset,
    preprocess_titanic_dataset,
)


LOGGING_LEVEL = logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=(
        "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    """main function of the module

    Args:
        args: arguments to be parsed see cli.py

    Raises:
        ValueError: if preprocess argument is not recognized
    """
    args = parse_args(args)
    arguments = vars(args)
    logging.info("cli arguments: %s", arguments)
    decision_tree_params = DecisionTreeParams(
        max_depth=arguments["max_depth"],
        min_samples_split=arguments["min_samples_split"],
        min_information_gain=arguments["min_information_gain"],
        mode=arguments["mode"],
        verbose=arguments["verbose"],
    )
    decision_tree = DecisionTree(
        decision_tree_params=decision_tree_params,
        verbose=arguments["verbose"],
    )

    if arguments["csv"]:
        data = read_csv(args.csv)
        # check for the preprocessing if it exists and apply it
        if arguments["preprocess"] == "bmi":
            data = preprocess_bmi_dataset(data, arguments["target_label"])
        elif arguments["preprocess"] == "titanic":
            data = preprocess_titanic_dataset(data, arguments["target_label"])
        elif arguments["preprocess"] not in ["", "bmi", "titanic"]:
            logging.warning("preprocess argument not recognized")
            raise ValueError(
                f"preprocess argument not recognized {arguments['preprocess']}"
            )
        training_set = data.sample(
            frac=0.8, random_state=42
        )  # pylint: disable=maybe-no-member
        test_set = data.drop(training_set.index)  # pylint: disable=maybe-no-member
        decision_tree.train(training_set, arguments["target_label"])
        logging.info(
            "The model accuracy is: %.2f%%",
            sum(
                (
                    test_set[arguments["target_label"]]
                    == decision_tree.infer_sample(test_set)
                )
                / len(test_set)
            )
            * 100,
        )
    else:
        logging.info("No csv file provided")
    logging.info("Goodbye!")


if __name__ == "__main__":
    main(sys.argv[1:])
