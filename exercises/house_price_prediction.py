from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
from pandas import DataFrame
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

CUR_YEAR = 2022

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates(subset='id')

    # delete rows with invalid values:
    cleaned_df = df.drop(df[(df['price'] <= 0) |
                            (df['bedrooms'] == 0) |
                            (df['sqft_living'] <= 0) |
                            (df['floors'] == 0) |
                            (df['id'] == 0)|
                            (df['bedrooms'] > 15)].index)

    # modify date format:
    cleaned_df['date'] = cleaned_df['date'].astype(str)
    cleaned_df['date'] = cleaned_df['date'].apply(lambda x: x[:4])

    # calculate house age:
    cur_year = np.full((cleaned_df.shape[0], 1), CUR_YEAR)
    house_age = cur_year - cleaned_df[['yr_built']]
    cleaned_df['house_age'] = house_age

    # remove last years of renovated feature and add numbers of years from last renovated feature:
    last_renovated = cleaned_df[['yr_built', 'yr_renovated']].max(axis=1)
    years_from_last_renovated = cleaned_df['date'].astype(int) - last_renovated
    cleaned_df.pop('date')
    cleaned_df['years_from_last_renovated'] = years_from_last_renovated

    # remove unnecessary features:
    cleaned_df.pop('zipcode')
    cleaned_df.pop('id')
    cleaned_df.pop('sqft_living15')
    cleaned_df.pop('sqft_lot15')
    cleaned_df.pop('sqft_lot')
    cleaned_df.pop('yr_renovated')
    cleaned_df.pop('yr_built')
    cleaned_df.pop('waterfront')

    response = cleaned_df['price']
    cleaned_df.pop('price')
    samples_matrix = cleaned_df
    return samples_matrix, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        feature_content = X[feature].values
        xy_cov = np.cov(feature_content, y)[0][1]
        x_sigma = np.std(feature_content)
        y_sigma = np.std(y)
        pearson_cor = xy_cov / (x_sigma * y_sigma)

        plot_name = "Feature Name: " + feature + ", Pearson Correlation: " + str("{:.3f}".format(pearson_cor))

        fig = go.Figure([go.Scatter(x=feature_content, y=y,
                                    marker=dict(color="blue", opacity=.7),
                                    mode="markers")],
                        layout=go.Layout(title=plot_name,
                                         xaxis={"title": feature},
                                         yaxis={"title": "Price"},
                                         height=400))
        pio.write_image(fig, output_path+'\%s.png' %feature, format='png')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r'C:\Users\edeno\IML.HUJI\datasets\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # create training and test sets - add the response vector to the samples matrix
    train_set = train_X.copy()
    train_set[train_y.name] = train_y.values

    test_set = test_X.copy()
    test_set[test_y.name] = test_y.values

    LR_model = LinearRegression()
    loss_data = pd.DataFrame({'percent': [], 'mean': [], 'var': []})  # df of percent, average_loss, variance_loss

    for p in range(10, 101):
        loss_arr = np.array([])
        for i in range(10):

            p_train_set = train_set.sample(frac=p/100)
            p_train_y = p_train_set[train_y.name]
            p_train_set.pop(train_y.name)
            p_train_X = p_train_set

            LR_model._fit(p_train_X.to_numpy(), p_train_y.to_numpy())

            loss = LR_model._loss(test_X.to_numpy(), test_y.to_numpy())
            loss_arr = np.append(loss_arr, loss)

        average_loss = loss_arr.mean()
        variance_loss = loss_arr.std()
        loss_data = loss_data.append({"percent": p, "mean": average_loss, "var": variance_loss}, ignore_index=True)

    fig = go.Figure([go.Scatter(x=loss_data['percent'], y=loss_data['mean'], mode="markers+lines",
                                name="Mean Loss", line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=loss_data["percent"], y=(loss_data["mean"] - 2 * loss_data["var"]),
                                fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=loss_data["percent"], y=(loss_data["mean"] + 2 * loss_data["var"]),
                                fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(title=r"$\text{Mean Loss as a function of Train-Data Percentage}$",
                                     xaxis_title="Percentage",
                                     yaxis_title="Mean Loss"))
    pio.write_image(fig, r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures\loss.png', format='png')


