import unittest

from decision_trees import get_best_split_feature
import pandas as pd
import numpy as np


class Test_TestBestSplit(unittest.TestCase):
    def test_get_best_split_feature_valid_input(self):
        # Test valid input
        dataframe = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "target": [1, 1, 1, 1, 1, 0, 0, 0, 0],
            }
        )
        feature = dataframe["feature1"]
        target = dataframe["target"].astype("object")

        # Test valid input
        # TODO: Review values in assert statement
        assert get_best_split_feature(feature, target) == (
            [6],
            0.9910760598382222,
            False,
            True,
        )

    def test_get_best_split_feature_2(self):
        """Test function to check the best split feature."""

        # Initialize features and target values
        features = pd.Series(["a", "b", "c", "d", "e", np.nan])
        target = pd.Series([0, 1, 0, 0, 1, 1])

        # Compute the best split feature
        (
            best_split_value,
            best_split_info_gain,
            is_cat,
            is_valid,
        ) = get_best_split_feature(features, target)
        # Assert results
        assert best_split_value == [
            "a",
            "c",
            "d",
        ], "The best split value should be ['a', 'c', 'd']"
        assert (
            best_split_info_gain == 0.3
        ), "The best split information gain should be 0.3"
        assert is_cat == True, "The best split variable type should be categorical"
        assert is_valid == True, "The split should be valid"


if __name__ == "__main__":
    unittest.main()
