import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
# from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import functools


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_err = np.zeros(n_learners)
    test_err = np.zeros(n_learners)

    for i in range(n_learners):
        train_err[i] = adaboost.partial_loss(train_X, train_y, i+1)
        test_err[i] = adaboost.partial_loss(test_X, test_y, i+1)

    fig1 = go.Figure([go.Scatter(y=train_err, mode="lines", name=r"$\text{Train Errors}$"),
                      go.Scatter(y=test_err, mode="lines", name=r"$\text{Test Errors}$")])

    # without noise
    fig1.update_layout(title_text=
                          r"$\text{Adaboost errors as a function of number of fitted learners (without noise)}$",
                          xaxis_title=r"$\text{Number of learners}$", yaxis_title=r"$\text{Missclasification Error}$")

    # with noise
    # fig1.update_layout(title_text=
    #                       r"$\text{Adaboost errors as a function of number of fitted learners (noise level 0.4)}$",
    #                       xaxis_title=r"$\text{Number of learners}$", yaxis_title=r"$\text{Missclasification Error}$")

    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} learners}}$" for t in T],
                         horizontal_spacing=0.01, vertical_spacing=0.03)

    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(functools.partial(adaboost.partial_predict, T=t), lims[0], lims[1]),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) +1)

    fig2.update_layout(title=r"$\text{Decision boundaries of models (without noise)}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_err_model = np.argmin(test_err)

    fig3 = go.Figure([decision_surface(functools.partial(adaboost.partial_predict, T=min_err_model), lims[0], lims[1]),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    fig3.update_layout(
        title=rf"$\textbf{{Decision boundaries of the best ensemble: {min_err_model} (without noise). Accuracy: {1 - test_err[min_err_model]}}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig3.show()

    # Question 4: Decision surface with weighted samples
    last_iteration_D = (adaboost.D_ / np.max(adaboost.D_)) * 10

    fig4 = go.Figure([decision_surface(functools.partial(adaboost.partial_predict, T=n_learners), lims[0], lims[1]),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y, size=last_iteration_D, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    # without noise
    fig4.update_layout(
        title=rf"$\textbf{{Decision boundaries of the full ensemble, whit {n_learners} iterations (without noise)}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    # with noise
    # fig4.update_layout(
    #     title=rf"$\textbf{{Decision boundaries of the full ensemble, whit {n_learners} iterations (noise level 0.4)}}$",
    #     margin=dict(t=100)) \
    #     .update_xaxes(visible=False).update_yaxes(visible=False)

    fig4.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    #fit_and_evaluate_adaboost(noise=0.4)