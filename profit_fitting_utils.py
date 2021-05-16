import numpy as np
import pandas as pd
import scipy.optimize
import seaborn as sns
from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose



# functions for fitting

def log_fun_5pl(x, L, k, t_0, c, g):
    return L / (1. + np.exp(-k * (x - t_0))) ** g +c

def log_fun_5pl_sym(x, L, k, t_0, c, g):
    return (L -c)/ (1. + np.exp(-k * (x - t_0))) ** g + c


def log_fun_4pl(x, L, k, t_0, c):
    return L / (1. + np.exp(-k * (x - t_0))) + c


#reading and plotting raw data
def read_data(prd):
    data = pd.read_excel("DATA_COMPANY_ABC.xlsx", sheet_name="P{}".format(prd)).set_index('date').T \
        .reset_index().rename(columns={"index": "Date"}, index={"date": "Month"})
    data["DateTime"] = pd.to_datetime(data.Date, format="%b %Y")
    return data


def plot_indicator(data, ax, column, prd, factor=1_000_000):
    sns.lineplot(x=data.Date, y=data[column] / factor, data=data, ax=ax)

    ax.set_title("{column} for product {name}, mln".format(column=column, name=prd))

    ax.set_xticks(data.Date[6::12])
    ax.set_xticklabels(data.Date[6::12])
    return ax

# get initial results

import itertools
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt

plt.style.use("seaborn")
import pandas as pd
from sklearn.linear_model import LinearRegression


def plot_ratios(data, min_value=0, max_value=None):
    """ Plot the proportional growth rate with respect to t,
        on the interval (min_value, max_value).
    """
    data = data[data > min_value]
    if max_value is not None:
        data = data[data < max_value]
    slopes = 0.5 * (data.diff(1) - data.diff(-1))
    ratios = slopes / data
    x = data.values[1:-1]
    y = ratios.values[1:-1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    _ = plt.plot(x, y, 'o')
    _ = ax.set_xlabel('Y(t)')
    _ = ax.set_ylabel('Ratios of slopes to function values')
    plt.show()
    return x, y


def linear_regression(x, y):
    """ Find the coefficients of the linear function  y=ax + b,
        using a linear regression.
    """
    X = x.reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True, normalize=True)
    _ = reg.fit(X, y)
    a = reg.coef_[0]
    b = reg.intercept_
    y_hat = a * x + b
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    _ = plt.plot(x, y, 'o')

    _ = plt.plot(x, y_hat, '-', label='Linear regression')
    _ = ax.set_xlabel('D(t)')
    _ = ax.set_ylabel('Ratios of slopes to function values')
    _ = ax.legend()
    plt.show()
    return a, b



#plot results
def plot_forecast_comparison(product, preds, metrics="Revenue"):
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.plot(product[[metrics, "DateTime"]].set_index("DateTime"), color='black', marker='o', markersize=2.5,
             label='True')
    colors = iter(["blue", "red", "green", "orange"])
    for label in preds.keys():
        plt.plot(preds[label], color=next(colors), label=label)

    ax.set_title("{} predictions for 10 years".format(metrics))
    plt.legend()
    plt.tight_layout()
    plt.show()



# Decomposition

def _decompose(ts, period=12, model="additive"):
    decomposition = seasonal_decompose(ts, period=period, model=model, extrapolate_trend='freq')

    trend = decomposition.trend
    seasonality = decomposition.seasonal
    residuals = decomposition.resid
    residuals.dropna(inplace=True)

    return trend, seasonality, residuals


def plot_decomposition(ts, trend, seasonality, residuals):

    # Original
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='upper left')

    # Trend
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')

    # Seasonality
    plt.subplot(413)
    plt.plot(seasonality, label='Seasonality')
    plt.legend(loc='upper left')

    # Resudials
    plt.subplot(414)
    plt.plot(residuals, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()


# Fitting logistic function

def fit_other_params(k, L, data, min_value=0,fit_fun = "4PL", max_value=None, tm=0, bounds = None,p0 = None):
    """ Find a value of t_m and c such that the logistic curve is as close
        as possible to the data on the given interval.
        :parameter data: series with datetime index
    """
    c = min(data)
    data = data[data > min_value]
    if max_value is not None:
        data = data[data < max_value]

    if fit_fun == "4PL":
        f = log_fun_4pl
        if not p0:
            p0 = [L,k,tm,c]
        if not bounds:
            bounds = [[max(data)*1.3, -np.inf,-np.inf,-np.inf],[max(data)*2,np.inf,np.inf,np.inf]]
    elif fit_fun == "5PL":
        f = log_fun_5pl
        if not p0:
            p0 = [L, k, tm, c,1]
        if not bounds:
            bounds = [[max(data)*1.3, 0, 0,min(data), 1], [max(data) * 2, 0.1, np.inf, np.inf,5]]

    params_fitted, _ = scipy.optimize.curve_fit(f,np.arange(data.shape[0]),data.values,
                                                p0 = p0, bounds = bounds, method='trf')
    res = f(np.arange(data.shape[0]),*params_fitted)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    _ = plt.plot(res, 'o',label = "Fitted")
    _ = plt.plot(data.values, 'd', label = 'True')
    plt.legend(loc = "upper left")
    plt.show()
    result = pd.Series(res, index = data.index)
    return params_fitted,result


def plot_forecast_comparison(product, preds, metrics="Revenue"):
    """
    Plot prediction till the last data index date
    :param product: px or py
    :param preds: dictionary with curve name and time series values
    :param metrics: revenue, costs etc
    :return:
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.plot(product[[metrics, "DateTime"]].set_index("DateTime"), color='black', marker='o', markersize=2.5,
             label='True')
    colors = iter(["blue", "red", "green", "orange"])
    for label in preds.keys():
        plt.plot(preds[label], color=next(colors), label=label)

    ax.set_title("{} predictions for 10 years".format(metrics))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_forecast_bounds(product, preds, metrics="Revenue"):
    """
    Plot prediction till the last data index date
    :param product: px or py
    :param preds: dictionary with curve name and time series values
    :param metrics: revenue, costs etc
    :return:
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.plot(product[[metrics, "DateTime"]].set_index("DateTime"), color='black', marker='o', markersize=2.5,
             label='True')


    for label in preds.keys():
        plt.plot(preds[label], color='C0', label=label)

    ax.fill_between(preds[label],preds[label]*1.1 , (y + ci), color='b', alpha=.1)
    ax.set_title("{} predictions for 10 years".format(metrics))
    plt.legend()
    plt.tight_layout()
    plt.show()



