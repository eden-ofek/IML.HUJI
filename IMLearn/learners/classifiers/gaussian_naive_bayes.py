from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier
        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`
        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`
        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`
        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # create the classes array from the given response vector
        # indices: holds for every cell in y its idx in classes
        self.classes_, counts = np.unique(y, return_counts=True)
        classes_number = self.classes_.size
        samples_number = X.shape[0]
        features_number = X.shape[1]

        # pi_i = count how many labels are tags as i'th class and then divide it by samples number
        self.pi_ = counts / samples_number

        # mu_i = 1/number of i'th class' labels * number of samples that tags as this label
        self.mu_ = np.zeros((classes_number, features_number))
        self.vars_ = np.zeros((classes_number, features_number))

        for idx, label in enumerate(self.classes_):
            x_tag_as_label = X[y == label]
            self.mu_[idx] = np.mean(x_tag_as_label, axis=0)
            self.vars_[idx] = np.var(x_tag_as_label, axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        argmax_label = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[argmax_label]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        samples_number = X.shape[0]
        classes_number = self.classes_.size
        likelihoods = np.zeros((samples_number, classes_number))

        from ..gaussian_estimators import MultivariateGaussian
        multi_gaussian = MultivariateGaussian()
        multi_gaussian.fitted_ = True

        for idx, label in enumerate(self.classes_):
            multi_gaussian.cov_ = np.diag(self.vars_[idx])
            multi_gaussian.mu_ = self.mu_[idx]
            likelihoods[:, idx] = multi_gaussian.pdf(X) * self.pi_[idx]

        return likelihoods

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
        from ...metrics import misclassification_error
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)