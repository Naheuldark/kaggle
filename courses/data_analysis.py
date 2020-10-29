import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from scipy import stats
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from courses.data_processing import read_data

pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.1)


def display_correlation(X):
    """
    Display correlation as heatmap

    :param X: training data
    """
    correlation_train = X.corr()
    mask = np.triu(correlation_train.corr())
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_train,
                annot=True,
                fmt='.1f',
                cmap='coolwarm',
                square=True,
                mask=mask,
                linewidths=1,
                cbar=False)
    plt.show()


def display_categorical(X, y):
    """
    Display categorical variables using box plots

    :param X: training data
    :param y: column containing target data
    """
    fig, axes = plt.subplots(14, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(X.select_dtypes(include=['object']).columns, axes):
        sortd = X.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i,
                    y=y,
                    data=X,
                    palette='plasma',
                    order=sortd.index,
                    ax=j)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()
    plt.show()


def display_numerical(X, y):
    """
    Display numerical variables using scatter plots and polynomial regression

    :param X: training data
    :param y: column containing target data
    """
    fig, axes = plt.subplots(12, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(X.select_dtypes(include=['number']).columns, axes):
        sns.regplot(x=i,
                    y=y,
                    data=X,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#e74c3c',
                    line_kws={'color': 'black'},
                    scatter_kws={'alpha': 0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
    plt.show()


def data_analysis(train_file, index, target):
    """
    Perform Exploratory Data Analysis on the data

    :param train_file: CSV containing training data
    :param index: column used as id for each row
    :param target: column used as prediction target
    """
    X = read_data(train_file, index)

    display_correlation(X)
    display_categorical(X, target)
    display_numerical(X, target)

    exit(0)
