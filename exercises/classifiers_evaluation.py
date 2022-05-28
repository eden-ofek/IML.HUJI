from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi
pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    dataset = np.load(filename)
    X = dataset[:, :2]
    y = dataset[:, 2].astype(int)
    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("C:/Users/edeno/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        model = Perceptron(callback=lambda perceptron, xi, yi: losses.append(perceptron._loss(X, y))).fit(X,y)

        # Plot figure of the losses as a function of iteration number
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(losses)+1)), y=losses))
        fig.update_layout(title=r"$\text{Loss on "+n+" as a function of fitiing iteration over Percpetron Algorithm}$",
                          xaxis_title=r"$\text{Fitting Iterations}$", yaxis_title=r"$\text{Loss}$")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("C:/Users/edeno/IML.HUJI/datasets/" + f)
        classes = np.unique(y)

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        y_pred_lda = lda.predict(X)
        y_pred_gnb = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        gnb_accuracy = accuracy(y, y_pred_gnb)
        lda_accuracy = accuracy(y, y_pred_lda)

        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=[r"$\text{Gaussian Naive Bayes, accuracy: " + str(round(gnb_accuracy, 3)) + "}$",
                                             r"$\text{LDA, accuracy: " + str(round(lda_accuracy, 3)) + "}$"],
                             horizontal_spacing=0.05, vertical_spacing=0.05)

        fig.update_layout(title={'text': r"$\text{Predict over " + f[:-4] + "}$",
                                 'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
                          font=dict(size=16))

        # Add traces for data-points setting symbols and colors
        symbols = np.array(["circle", "square", "diamond"])
        fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color=y_pred_gnb, symbol=symbols[y])),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color=y_pred_lda, symbol=symbols[y])),
                      row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=gnb.mu_[:,0], y=gnb.mu_[:,1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color='red', symbol='x', size=10)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                 mode="markers", showlegend=False,
                                 marker=dict(color='red', symbol='x', size=10)),
                      row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for idx, label in enumerate(classes):
            fig.add_trace(get_ellipse(gnb.mu_[idx], np.diag(gnb.vars_[idx])), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[idx], lda.cov_), row=1, col=2)

        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
