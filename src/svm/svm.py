""" This module contains the implementation of the SVM algorithm. """

from abc import abstractmethod

from sklearn.utils import shuffle
from scipy.optimize import minimize
import numpy as np

from abc_models.models import SupervisedTabularDataModel


class NaiveSVC(SupervisedTabularDataModel):
    """Implementation of the SVC algorithm

    TODO: implement the kernel trick
    TODO: implement the dual form
    TODO: Re-implement the optimizer
    TODO: Cythonization ?"""

    weights: np.ndarray = np.array([])
    bias: float = 0

    def __init__(
        self, learning_rate: float, lambda_rate: float, iterations: int = 1000
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_rate = lambda_rate

    def _set_init_weights(self, data: np.ndarray) -> None:
        """Set the initial weights and bias"""
        self.weights = np.zeros(data.shape[1])
        self.bias = 0

    def _fit(self, dataframe: np.ndarray, target: np.ndarray) -> "NaiveSVC":
        """_summary_

        Args:
            data (np.ndarray): the training data
            target (np.ndarray): the training labels

        Returns:
            SVC: the class model
        """
        self._set_init_weights(dataframe)
        for _ in range(self.iterations):
            x_data, y_target = shuffle(dataframe, target)  # type: ignore
            for idx, x_train in enumerate(x_data):
                y_train: float = y_target[idx]
                condition = y_train * (np.dot(self.weights, x_train) + self.bias) < 1
                gradient_w, gradient_b = self._compute_gradient(
                    condition, x_train, y_train
                )

                self._update_weights(gradient_w, gradient_b)
        return self

    def _predict(self, dataframe: np.ndarray) -> np.ndarray:
        # Predict the target values
        prediction = np.dot(dataframe, self.weights) + self.bias
        return np.sign(prediction)

    def _update_weights(self, gradient_w: np.ndarray, gradient_b: float) -> None:
        """This method updates the weights and bias

        Args:
            gradient_w (np.ndarray): the gradient of the weights
            gradient_b (float): the gradient of the bias
        """

        self.weights -= self.learning_rate * gradient_w
        self.bias -= self.learning_rate * gradient_b

    def _compute_gradient(
        self, condition: bool, x_train: np.ndarray, y_train: float
    ) -> tuple[np.ndarray, float]:
        """This method computes  the gradient descent

        Args:
            condition (bool): support vectors condition
            x_train (np.ndarray): training input
            y_train (np.ndarray): training input label

        Returns:
            _type_: _description_
        """

        if condition:
            gradient_w = self.lambda_rate * self.weights - x_train * y_train
            gradient_b = -y_train
        else:
            gradient_w = self.lambda_rate * self.weights
            gradient_b = 0
        return gradient_w, gradient_b


class GenericSVM(SupervisedTabularDataModel):
    """Generic implementation of SVM algorithm family"""

    alpha: np.ndarray = np.array([])
    weights: np.ndarray = np.array([])
    support_vectors: np.ndarray = np.array([])
    support_labels: np.ndarray = np.array([])
    intercept: float = 0
    epsilon_clip: float = 1e-6
    budget: int = 1
    dataframe: np.ndarray = np.array([])
    target: np.ndarray = np.array([])

    def _fit(self, dataframe: np.ndarray, target: np.ndarray) -> "GenericSVM":
        """training the MMC model

        Args:
            data (np.ndarray): the training data
            target (np.ndarray): the training labels

        Returns:
            MaxMarginClassifier: the class model
        """

        # our goal is to minimize
        #       L_d(alpha)=\sum\alpha_i -1/2\sum\sum<\alpha_iy_ix_i,\alpha_ky_kx_k>
        # thus maximize -L_d(alpha)
        # we will create the gram matrix of Xy (<yixi,ykxk>)
        # we will then impelment the L_d(alpha) as a function of alpha
        # and then use the scipy.optimize.minimize to find the optimal alpha
        # then we will use the alpha to compute the weights and the bias
        # and finally we will use the weights and the bias to predict the
        # target values

        # create the gram matrix
        # Gram_XY = (X.Y) * transpose(X.Y)
        self.dataframe = dataframe
        self.target = target

        gram_datay = np.dot(
            dataframe * target[:, np.newaxis], (dataframe * target[:, np.newaxis]).T
        )

        # constrains on alpha
        cons = self.constraints(target)

        # maximize by minimizing the opposite
        optimization_result = minimize(
            fun=lambda a: -self.objective(gram_datay, a),
            x0=np.zeros(len(target)),
            jac=lambda a: -self.objective_derivative(gram_datay, a),
            constraints=cons,
            method="SLSQP",
        )

        # get the optimal alpha
        self.alpha = optimization_result.x
        self.weights = np.dot(self.alpha * target, dataframe)
        #  alpha is sparse and strong duality means is alpha > 0 then yi(<weights, xi> +b) = 1
        # if yi(<weights, xi> +b) > 1 the distance of xi to the hyperplan is
        # larger than the margin.
        self.compute_support_vectors()
        self.compute_intercept()

        return self

    def _predict(self, dataframe: np.ndarray) -> np.ndarray:
        """Predict y value in {-1, 1}"""
        return 2 * (np.matmul(dataframe, self.weights) > 0) - 1

    def compute_support_vectors(self) -> None:
        """This method computes the support vectors and their labels"""
        self.support_vectors = self.dataframe[self.alpha > self.epsilon_clip]
        self.support_labels = self.target[self.alpha > self.epsilon_clip]

    @staticmethod
    def objective(gram: np.ndarray, alpha: np.ndarray) -> float:
        """define the objective function"""
        return alpha.sum() - 0.5 * alpha.dot(alpha.dot(gram))

    @staticmethod
    def objective_derivative(gram: np.ndarray, alpha: np.ndarray) -> float:
        """define the partial derivative of the objective function on alpha"""
        return np.ones_like(alpha) - np.dot(alpha, gram)

    def compute_intercept(self) -> None:
        """This function computes the hyperplan intercept"""

        # using a support vector x with target y: b = target - <weights, x>
        # we typically use an average of all the solutions for numericalstability
        # for max margin classifier, the distrubution are separable thus we can
        # choose any support vector or average on all of them

        self.intercept = (
            self.support_labels - np.matmul(self.support_vectors, self.weights)
        ).mean()

    @abstractmethod
    def constraints(self, target: np.ndarray) -> tuple:
        """placeholder to define the constraints

        Args:
            target (np.ndarray): _description_

        Returns:
            tuple: _description_
        """


class MaxMarginClassifier(GenericSVM):
    """Implements the Max Margin Classifier algorithm"""

    def constraints(self, target: np.ndarray) -> tuple:
        """constraints"""
        # f(alpha) >= 0
        # <alpha, x> = 0
        function_alpha = -np.eye(len(target))
        constants = np.zeros(len(target))

        cons = (
            {
                "type": "eq",
                "fun": lambda a: np.dot(a, target),
                "jac": lambda a: target,
            },
            {
                "type": "ineq",
                "fun": lambda a: constants - np.dot(function_alpha, a),
                "jac": lambda a: -function_alpha,
            },
        )
        return cons


class SVC(GenericSVM):
    """Implements the Support Vector Classifier algorithm"""

    def __init__(self, budget: int = 1) -> None:
        super().__init__()
        self.budget = budget

    def constraints(self, target: np.ndarray) -> tuple:
        """constraints"""
        # f(alpha) >= 0
        # <alpha, x> = 0
        ndim = len(target)
        function_alpha = np.vstack((-np.eye(ndim), np.eye(ndim)))
        constants = np.hstack((np.zeros(ndim), self.budget * np.ones(ndim)))

        cons = (
            {
                "type": "eq",
                "fun": lambda a: np.dot(a, target),
                "jac": lambda a: target,
            },
            {
                "type": "ineq",
                "fun": lambda a: constants - np.dot(function_alpha, a),
                "jac": lambda a: -1 * function_alpha,
            },
        )
        return cons

    def compute_intercept(self):
        """Implementation of the intercept for SVC"""
        # using a support vector x with target y: b = target - <weights, x>
        # we typically use an average of all the solutions for numerical stability
        # we use point that are leaning on the margin thus, their alpha are
        # equal to the budget

        vectors = self.dataframe[
            (self.alpha > self.epsilon_clip) & (self.alpha < self.budget)
        ]
        labels = self.target[
            (self.alpha > self.epsilon_clip) & (self.alpha < self.budget)
        ]
        # using a support vector x with target y: b = target - <weights, x>
        # we typically use an average of all the solutions for numerical
        # stability

        self.intercept = (labels - np.matmul(vectors, self.weights)).mean()
