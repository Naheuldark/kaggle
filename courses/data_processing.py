import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def process_missing_values(train_X, val_X, test_X):
    """
    Use imputation to handle missing value, fit the transform on training data and apply it as well on validation and test
    :param train_X: training data
    :param val_X: validation data
    :param test_X: testing data
    :return: processed_train_X, processed_val_X, processed_test_X
    """
    imputer = SimpleImputer(strategy='median')

    processed_train_X = pd.DataFrame(imputer.fit_transform(train_X))
    processed_train_X.columns = train_X.columns

    processed_val_X = pd.DataFrame(imputer.transform(val_X))
    processed_val_X.columns = val_X.columns

    processed_test_X = pd.DataFrame(imputer.transform(test_X))
    processed_test_X.columns = test_X.columns

    return processed_train_X, processed_val_X, processed_test_X


def process_categorical_values(train_X, val_X, test_X):
    """
    Use imputation to handle missing value, fit the transform on training data and apply it as well on the validation
    :param train_X: training data
    :param val_X: validation data
    :return: processed_train_X, processed_val_X
    """


    return processed_train_X, processed_val_X, processed_test_X


def preprocess(train_file, test_file, target, index, ratio):
    """
    Preprocess the data and extract training data, validation data and test data
    :param train_file: CSV file containing the training data
    :param test_file: CSV file containing the test data
    :param target: column used as model target
    :param index: column used as id for each row
    :param ratio: ratio of data belonging to the training set (the rest will be used for validation) [0 < r < 1]
    :return: train_X, val_X, train_y, val_y, test_X
    """
    assert 0.0 < ratio < 1.0

    # Read the data
    train_data = pd.read_csv(train_file, index_col=index)
    test_X = pd.read_csv(test_file, index_col=index)

    # Remove rows with missing target
    train_data.dropna(axis=0, subset=[target], inplace=True)
    y = train_data[target]

    # Separate target from predictors
    X = train_data.drop([target], axis=1)

    # Split training data into training and validation using [ratio]
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=ratio, test_size=1-ratio)

    # Process the data
    train_X, val_X, test_X = process_missing_values(train_X, val_X, test_X)
    train_X, val_X, test_X = process_categorical_values(train_X, val_X, test_X)

    return train_X, val_X, train_y, val_y, test_X
