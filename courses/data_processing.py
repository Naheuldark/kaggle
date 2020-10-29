import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def read_data(file, index):
    """
    Read CSV data and return an Panda Dataframe
    :param file: CSV file
    :param index: column used as id for each row
    :return: pd.Dataframe
    """
    return pd.read_csv(file, index_col=index)


def import_data(train_file, test_file, target, index):
    """
    Import training and testing data

    :param train_file: CSV file containing the training data
    :param test_file: CSV file containing the test data
    :param target: column used as model target
    :param index: column used as id for each row
    :return: X, y, test_data
    """
    # Read the data
    train_data = read_data(train_file, index)
    test_data = read_data(test_file, index)

    # Remove rows with missing target
    train_data.dropna(axis=0, subset=[target], inplace=True)
    y = train_data[target]

    # Separate target from predictors
    X = train_data.drop([target], axis=1)

    return X, y, test_data


def get_columns(X):
    """
    Select columns to work with

    :param X: imported training data
    :return: all columns, numerical columns, categorical columns
    """
    numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    columns = numerical_cols + categorical_cols

    return columns, numerical_cols, categorical_cols


def get_preprocessor(numerical_cols, categorical_cols):
    """
    Compute the preprocessor for the data

    :param numerical_cols: columns containing numerical data
    :param categorical_cols: columns containing categorical data with low cardinality
    :return: preprocessor
    """
    # Preprocessing for numerical data
    # numerical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scale', StandardScaler())
    # ])
    numerical_transformer = 'passthrough'

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor


def preprocess(train_file, test_file, target, index, ratio):
    """
    Preprocess the data and extract training data, validation data and test data

    :param train_file: CSV file containing the training data
    :param test_file: CSV file containing the test data
    :param target: column used as model target
    :param index: column used as id for each row
    :param ratio: ratio of data belonging to the training set (the rest will be used for validation) [0 < r < 1]
    :return: train_X, val_X, train_y, val_y, test_X, preprocessor
    """
    assert 0.0 < ratio < 1.0

    X, y, test_X = import_data(train_file, test_file, target, index)

    columns, num_cols, cat_cols = get_columns(X)

    # Split training data into training and validation using [ratio]
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=ratio, test_size=1 - ratio, random_state=0)

    return train_X[columns].copy(), \
           val_X[columns].copy(), \
           train_y, \
           val_y, \
           test_X[columns].copy(), \
           get_preprocessor(num_cols, cat_cols)


def preprocess_xgboost(train_file, test_file, target, index):
    """
    Preprocess the data and extract training data, validation data and test data for XGBoost

    :param train_file: CSV file containing the training data
    :param test_file: CSV file containing the test data
    :param target: column used as model target
    :param index: column used as id for each row
    :return: train_X, train_y, test_X, preprocessor
    """
    X, y, test_X = import_data(train_file, test_file, target, index)

    columns, num_cols, cat_cols = get_columns(X)

    return X[columns].copy(), y, test_X[columns].copy(), get_preprocessor(num_cols, cat_cols)
