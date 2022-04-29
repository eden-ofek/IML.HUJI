import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna()
    df = df.drop(df[df['Temp'] <= (-60)].index)
    df["DayOfYear"] = df["Date"].dt.day_of_year

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    init_data = load_data(r'C:\Users\edeno\IML.HUJI\datasets\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_data = init_data.copy()[init_data['Country'] == 'Israel']
    israel_data["Year"] = israel_data["Year"].astype(str)
    fig1 = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                     title="Temperature in Israel as a function of Day of year")
    fig1.write_image(r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures\temp_in_israel.png', format='png')

    std_temp = israel_data.groupby("Month").agg({"Temp": "std"}).reset_index()

    fig2 = px.bar(std_temp, x="Month", y="Temp", text_auto=True,
                  title="Standard deviation of the daily temperatures in Israel")
    fig2.write_image(r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures\std_temp_by_month.png', format='png')

    # Question 3 - Exploring differences between countries
    grouped_data = init_data.copy().groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()
    fig3 = px.line(grouped_data, x="Month", y="mean", color="Country", error_y="std")
    fig3.update_layout(title='Average and std temperature by month',
                      xaxis_title='Month',
                      yaxis_title='Average Temperature')
    fig3.write_image(r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures\eva_std_temp_by_month.png', format='png')

    # Question 4 - Fitting model for different values of `k`
    y = israel_data["Temp"]
    X = israel_data.copy()
    X.pop("Temp")

    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)
    train_X = train_X["DayOfYear"]
    test_X = test_X["DayOfYear"]

    israel_loss_data = pd.DataFrame({"k": [], "loss": []})  # df of k : loss
    for k in range(1, 11):
        PL_model = PolynomialFitting(k)
        PL_model._fit(train_X.to_numpy(), train_y.to_numpy())
        loss = PL_model._loss(test_X.to_numpy(), test_y.to_numpy()).round(2)
        israel_loss_data = israel_loss_data.append({"k": k, "loss": loss}, ignore_index=True)

    fig4 = px.bar(israel_loss_data, x="k", y="loss", text_auto=True,
                  title="Test error for each value of k")
    fig4.update_layout(xaxis_title='Polynomial Degree',
                       yaxis_title='Loss')
    fig4.write_image(r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures\k_loss_israel.png', format='png')

    # Question 5 - Evaluating fitted model on different countries
    PL_model = PolynomialFitting(5)
    PL_model._fit(X["DayOfYear"].to_numpy(), y.to_numpy())
    countries_loss_data = pd.DataFrame({"Country": [], "Loss": []})  # df of country : loss
    for country in ["South Africa", "Jordan", "The Netherlands"]:
        data_set = init_data.copy()[init_data['Country'] == country]
        train_X = data_set["DayOfYear"]
        train_y = data_set["Temp"]
        country_loss = PL_model._loss(train_X.to_numpy(), train_y).round(2)
        countries_loss_data = countries_loss_data.append({"Country": country, "Loss": country_loss}, ignore_index=True)

    fig5 = px.bar(countries_loss_data, x="Country", y="Loss", text_auto=True,
                  title="Israel's fitted model's error over each of the other countries")
    fig5.write_image(r'C:\Users\edeno\IML.HUJI\exercises\ex2_figures\countries_loss.png', format='png')


