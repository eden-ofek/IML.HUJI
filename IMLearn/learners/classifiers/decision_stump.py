from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from IMLearn.metrics import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm
    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split
    self.j_ : int
        The index of the feature by which to split the data
    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_feature = X.shape[1]
        min_err = np.inf

        for i in range(n_feature):
            for sign in [1, -1]:
                threshold, err = self._find_threshold(X[:,i], y, sign)

                if err < min_err:
                    min_err = err
                    self.j_ = i
                    self.threshold_ = threshold
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        response = np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)
        return response

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature
        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for
        labels: ndarray of shape (n_samples,)
            The labels to compare against
        sign: int
            Predicted label assigned to values equal to or above threshold
        Returns
        -------
        thr: float
            Threshold by which to perform split
        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold
        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        n_sample = values.shape[0]
        sorted_values_indices = np.argsort(values)
        sorted_values = np.take(values, sorted_values_indices)

        sorted_order_labels = np.take(labels, sorted_values_indices)
        signed_sorted_order_labels = np.sign(sorted_order_labels)

        # initialize return values
        min_thr = sorted_values[0]  # initialize the threshold as the smallest value
        min_err = 1  # the biggest loss error

        # initialize threshold as values[0]. all labels equal to "sign" because all values >= values[0] (after sorting)
        thr_order_labels = np.ones(n_sample) * sign

        for i in range(n_sample):
            err_vec = np.where(signed_sorted_order_labels != thr_order_labels, np.abs(sorted_order_labels), 0)
            err = np.sum(err_vec) / n_sample

            if err < min_err:
                min_err = err
                min_thr = sorted_values[i]

            # change the i'th label to "-sign" because we moved one value forward by i++
            thr_order_labels[i] = -sign

        return min_thr, min_err

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
        y_pre = self.predict(X)
        err = misclassification_error(y, y_pre)
        return err
