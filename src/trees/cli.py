"""This is the cli module of the project.
"""

from argparse import ArgumentParser, Namespace


def parse_args(args) -> Namespace:
    """ argument parser for the module

    Args:
        args: arguments to be parsed

    Returns:
        parsed arguments
    """

    parser = ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--preprocess", type=str, default="")
    parser.add_argument("--target_label", type=str)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_information_gain", type=float, default=1e-10)
    parser.add_argument("--mode", type=str, default="classification")
    parser.add_argument("--verbose", type=bool, default=False)

    return parser.parse_args(args)
