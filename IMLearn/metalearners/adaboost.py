import numpy as np
from IMLearn.base.base_estimator import BaseEstimator  # TODO: change back to ...base
from typing import Callable, NoReturn
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner
    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator
    self.iterations_: int
        Number of boosting iterations to perform
    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator
        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator
        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_sample = X.shape[0]

        # initialize D to be the uniform distribution
        self.D_ = np.array([1/n_sample for i in range(n_sample)])
        self.models_, self.weights_ = [], []  # arrays of let self.iterations_

        for i in range(self.iterations_):
            # 1) find the best weak lerner for the current data
            weak_lerner = self.wl_().fit(X, y * self.D_)

            # 2) calculate the weight of the weak lerner
            y_pre = weak_lerner.predict(X)
            err = np.sum(self.D_[y != y_pre])
            weight = 0.5 * np.log((1 / err) - 1)

            # 3) update D
            self.D_ *= np.exp(-y * weight * y_pre)
            self.D_ /= np.sum(self.D_)

            # 4) save the parameters of the i'th iteration
            self.models_.append(weak_lerner)
            self.weights_.append(weight)

        self.models_ = np.array(self.models_)
        self.weights_ = np.array(self.weights_)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pre = np.zeros(X.shape[0])

        for i in range(T):
            y_pre += self.weights_[i] * self.models_[i].predict(X)

        return np.sign(y_pre)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pre = self.partial_predict(X, T)
        err = misclassification_error(y, y_pre)
        return err

