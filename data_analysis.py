import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from scipy.stats import norm, probplot

from sklearn.feature_selection import mutual_info_regression

from data_processing import read_data

pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.1)

SMALL_SIZE = 5
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def display_mutual_information(X, y):
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc("axes", labelweight="bold", titleweight="bold", titlepad=10)

    X = X.copy()
    for colname in X.select_dtypes(["object", "category", "float64"]):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=True)

    width = np.arange(len(mi_scores))
    ticks = list(mi_scores.index)
    plt.barh(width, mi_scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()


def display_correlation(X, y):
    """
    Display correlation as heatmap

    :param X: training data
    :param y: column containing target data
    """
    correlation_train = X.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_train,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                square=True,
                mask=np.triu(correlation_train.corr()),
                linewidths=1,
                cbar=False)
    plt.show()

    print("Highly correlated features (corr > 0.6)")
    corr_map = np.tril(correlation_train)
    for row in range(corr_map.shape[0] - 1):
        for col in range(corr_map.shape[1]):
            if 0.6 < corr_map[row][col] < 1.0:
                print("\t{:<15}\t{:<15}".format(correlation_train.index[row], correlation_train.columns[col]))
    print()

    print(y + " Correlation Analysis")
    print(">> Highly correlated features")
    for col in range(corr_map.shape[1]):
        if 0.6 < corr_map[corr_map.shape[0] - 1][col]:
            print("\t{:<15}".format(correlation_train.columns[col]))

    print(">> Features more likely to be dropped")
    for col in range(corr_map.shape[1]):
        if 0.2 > corr_map[corr_map.shape[0] - 1][col]:
            print("\t{:<15}".format(correlation_train.columns[col]))


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


def display_target(X, y, title):
    """
    Display target using several plots

    :param X: training data
    :param y: column containing target data
    :param title: title of the whole figure
    """
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    grid = GridSpec(ncols=3, nrows=2, figure=fig)

    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(X.loc[:, y],
                 hist=True,
                 kde=True,
                 fit=norm,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Actual', 'Normal'])

    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('Probability Plot')
    probplot(X.loc[:, y].fillna(np.mean(X.loc[:, y])), plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(X.loc[:, y], orient='v', color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{title}', fontsize=24)
    plt.show()


def data_analysis(train_file, index, target):
    """
    Perform Exploratory Data Analysis on the data

    :param train_file: CSV containing training data
    :param index: column used as id for each row
    :param target: column used as prediction target
    """
    X = read_data(train_file, index)
    y = X[target]
    X.drop(target, axis=1, inplace=True)

    display_mutual_information(X, y)
    # display_correlation(X.join(y), target)
    # display_categorical(X.join(y), target)
    # display_numerical(X.join(y), target)
    #
    # display_target(X.join(y), target, target + ' Before Log Transformation')
    # y = np.log1p(y)
    # display_target(X.join(y), target, target + ' After Log Transformation')
