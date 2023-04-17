"""
Gradient Boosting Machine implementation following "Introduction to
Statistical Learning" (ISLR) book.

This module provides the GBMClassifier class, which is a supervised
tabular data model that uses decision trees as base estimators to perform
classification. This implementation uses a gradient boosting approach to
boost the performance of the individual decision trees by iteratively
fitting new trees to the negative gradients of the loss function.

Attributes:
    num_classes (int): The number of classes in the target variable.
    epsilon (float): A small value added to the normalization denominator
        to avoid division by zero errors.

Methods:
    __init__(self, n_estimators, decision_tree_params, learning_rate=0.1):
        Initializes a new GBMClassifier instance.
    _fit(self, dataframe, target):
        Fits the model to the training data.
    _predict(self, dataframe):
        Predicts class labels for new data.

"""

from typing import List
import pandas as pd
import numpy as np


from abc_models.models import SupervisedTabularDataModel

from trees.decision_trees import DecisionTree


class GBMClassifier(SupervisedTabularDataModel):
    """
    A supervised machine learning model that uses gradient boosting and decision
    trees to perform classification.

    The `GBMClassifier` class is an implementation of the gradient boosting
    machine learning algorithm for classification tasks. This model works by
    fitting an ensemble of decision trees to the negative gradients of a loss
    function, and iteratively improving the predictions of the model.

    Attributes:
        num_classes (int): The number of classes in the target variable.
        epsilon (float): A small value added to the normalization denominator
            to avoid division by zero errors.
        n_estimators (int): The number of decision trees to fit to the data.
        learning_rate (float): The learning rate of the model, which controls
            the contribution of each new tree to the final ensemble.
        estimators (list[list[DecisionTree]]): The list of decision trees that
            form the ensemble for each class.

    Methods:
        __init__(self, n_estimators, decision_tree_params, learning_rate=0.1):
            Initializes a new GBMClassifier instance.
        _fit(self, dataframe, target):
            Fits the model to the training data.
        _predict(self, dataframe):
            Predicts class labels for new data.

    This class inherits from the `SupervisedTabularDataModel` abstract class
    and implements the `_fit` and `_predict` methods to provide a complete
    supervised learning interface. The decision trees used in the model are
    instances of the `DecisionTree` class, which can be customized through
    the `decision_tree_params` parameter passed to the constructor.

    Note that this implementation assumes that the target variable is categorical,
    and that its values are represented as integers from 0 to `num_classes - 1`.
    If the target variable has a different representation, it should be encoded
    appropriately before fitting the model.
    """

    # TODO: add learning rate
    # TODO: add early stopping ?
    # TODO: add subsampling
    # TODO: add feature subsampling??
    # TODO: add regularization
    # TODO: add loss function

    num_classes: int = 0

    epsilon: float = 1e-5

    def __init__(
        self,
        n_estimators,
        decision_tree_params,
        learning_rate: float = 0.1,
        subsample_frac: float = 1,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators: list[list[DecisionTree]] = [[]]
        self.decision_tree_params = decision_tree_params
        self.decision_tree_params.mode = "regression"
        self.subsample_frac = subsample_frac

    def _fit(self, dataframe: pd.DataFrame, target: pd.Series) -> "GBMClassifier":
        """Fit the GBMClassifier to the training data.

        Args:
            dataframe (pd.DataFrame): The feature matrix of shape (n_samples, n_features).
            target (pd.Series): The target vector of shape (n_samples,).

        Returns:
            GBMClassifier: The fitted GBMClassifier.

        Raises:
            AssertionError: If the number of classes is not greater than 1, or if the number of
                classes does not match the number of target_dummies_columns.
        """
        # TODO: check if target is object
        self.num_classes = len(target.unique())
        assert self.num_classes > 1, "num classes int should be higher than 2"
        self.estimators = [[] for k in range(self.num_classes)]
        dataframe = dataframe.reset_index(drop=True)
        target = target.reset_index(drop=True)
        target_dummies = pd.get_dummies(target)
        target_dummies_columns = target_dummies.columns
        assert self.num_classes == len(
            target_dummies_columns
        ), "num classes and num columns must match"
        estimator = pd.DataFrame(
            0, index=dataframe.index, columns=list(target_dummies_columns)
        )

        for m in range(1, self.n_estimators + 1):
            if m % 100 == 0:
                print(f"Initiating estimator {m} out of {self.n_estimators} ... ")
            normalization = pd.concat(
                [np.sum(np.exp(estimator), axis=1)] * self.num_classes, axis=1
            )
            normalization.columns = target_dummies_columns

            probabilities = np.exp(estimator) / (normalization + self.epsilon)

            negative_gradient = target_dummies - probabilities

            indices = np.random.choice(
                dataframe.index,
                size=int(dataframe.shape[0] * self.subsample_frac),
                replace=False,
            )

            training_dataframe = dataframe.loc[indices]
            training_negative_gradient = negative_gradient.loc[indices]
            for k in range(self.num_classes):
                kth_negative_gradient = training_negative_gradient[
                    target_dummies_columns[k]
                ]
                current_estimator = DecisionTree(
                    decision_tree_params=self.decision_tree_params
                )
                current_estimator.fit(training_dataframe, kth_negative_gradient)
                gamma_m_k = {}
                for j, leaf in enumerate(current_estimator.leaves):
                    region_indexes = leaf["target_indexes"]
                    kth_j_negative_gradient = kth_negative_gradient.loc[region_indexes]
                    gamma_m_k_j = (
                        (self.num_classes - 1)
                        / self.num_classes
                        * np.sum(kth_j_negative_gradient)
                        / (
                            np.sum(
                                np.multiply(
                                    np.abs(kth_j_negative_gradient),
                                    1 - np.abs(kth_j_negative_gradient),
                                )
                            )
                        )
                    )
                    gamma_m_k[j] = gamma_m_k_j
                    estimator.iloc[region_indexes, k] += (
                        self.learning_rate * gamma_m_k_j
                    )
                self.estimators[k].append(current_estimator)

        return self

    def _predict(self, dataframe: pd.DataFrame) -> pd.Series:
        """Predict class labels for new data.

        Args:
            dataframe (pd.DataFrame): The feature matrix of shape (n_samples, n_features).

        Returns:
            pd.Series: The predicted class labels of shape (n_samples,).

        Notes:
            The predicted class labels are determined using a weighted sum of the class probabilities
            estimated by the decision trees in each class-specific ensemble, where the weights are the
            learning rate and the fitted gamma values. The predicted class is the one with the highest
            probability.

        Raises:
            ValueError: If the number of classes is not greater than 1.
        """

        predictions = pd.DataFrame(
            0, index=dataframe.index, columns=list(range(self.num_classes))
        )
        for k, estimators in enumerate(self.estimators):
            for estimator in estimators:
                predictions[k] += estimator.predict(dataframe)

        predictions = np.exp(predictions) / pd.concat(
            [np.sum(np.exp(predictions), axis=1)] * self.num_classes, axis=1
        )

        predictions = predictions.idxmax(axis=1)
        return predictions
