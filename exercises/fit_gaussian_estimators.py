from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    X = np.random.normal(mu, sigma, 1000)
    estimator = UnivariateGaussian()
    estimator.fit(X)
    print((estimator.mu_, estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    dis = []
    for i in range(10, 1000, 10):
        estimator.fit(X[0:i])
        dis.append(np.abs(estimator.mu_ - 10))

    x = range(10, 1000, 10)
    plt.plot(x, dis)
    plt.title('Expectation error by number of samples')
    plt.xlabel('Number of samples')
    plt.ylabel('Distance from true expectation')
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.plot(X, estimator.pdf(X), 'o')
    plt.title('Empirical probability density function (PDF) of samples')
    plt.xlabel('Sample')
    plt.ylabel('PDF')
    plt.show()
    

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    estimator = MultivariateGaussian()
    estimator.fit(X)

    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    samples = np.linspace(-10, 10, 200)

    likelihood_matrix = np.zeros((samples.size, samples.size))
    for f1 in range(samples.size):
        val1 = samples[f1]
        for f3 in range(samples.size):
            val3 = samples[f3]
            mu = np.array([val1, 0, val3, 0])
            likelihood_matrix[f1, f3] = estimator.log_likelihood(mu, cov, X)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=samples, y=samples, z=likelihood_matrix))
    fig.update_layout(title='Log likelihood', xaxis_title='f3', yaxis_title='f1')
    fig.show()

    # Question 6 - Maximum likelihood
    max_f1 = 0
    max_f3 = 0
    max_likelihood = -np.inf

    for f1 in range(samples.size):
        val1 = samples[f1]
        for f3 in range(samples.size):
            val3 = samples[f3]
            if likelihood_matrix[f1, f3] > max_likelihood:
                max_likelihood = likelihood_matrix[f1, f3]
                max_f1 = val1
                max_f3 = val3

    print("Maximum value for f1 is:", max_f1)
    print("Maximum value for f3 is:", max_f3)
    print("Maximum likelihood is:", max_likelihood)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()