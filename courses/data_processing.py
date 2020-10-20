import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def check_data(train_X, val_X, test_X):
    """
    Perform checks on the preprocessed data
    :param train_X: processed training data
    :param val_X: processed validation data
    :param test_X: processed testing data
    """
    # Assert there is no missing values anymore
    assert train_X.isnull().sum()[train_X.isnull().sum() > 0].empty
    assert val_X.isnull().sum()[val_X.isnull().sum() > 0].empty
    assert test_X.isnull().sum()[test_X.isnull().sum() > 0].empty

    # Assert there are only numerical values left
    assert train_X.select_dtypes(include=['object']).empty
    assert val_X.select_dtypes(include=['object']).empty
    assert test_X.select_dtypes(include=['object']).empty


def process_categorical_values(train_X, val_X, test_X, cat_cols):
    """
    Process categorical missing values and convert them to numerical
    Use label encoding and one hot encoding to handle categorical values
    :param train_X: training data
    :param val_X: validation data
    :param test_X: testing data
    :param cat_cols: columns containing categorical values only
    :return: processed_cat_train_X, processed_cat_val_X, processed_cat_test_X
    """
    cat_train_X = train_X[cat_cols]
    processed_cat_train_X = cat_train_X.copy()
    for col in cat_train_X.columns:
        processed_cat_train_X[col].values[:] = 0
    processed_cat_train_X.index = train_X.index
    processed_cat_train_X = processed_cat_train_X.astype('int64')

    cat_val_X = val_X[cat_cols]
    processed_cat_val_X = cat_val_X.copy()
    for col in cat_val_X.columns:
        processed_cat_val_X[col].values[:] = 0
    processed_cat_val_X.index = val_X.index
    processed_cat_val_X = processed_cat_val_X.astype('int64')

    cat_test_X = test_X[cat_cols]
    processed_cat_test_X = cat_test_X.copy()
    for col in cat_test_X.columns:
        processed_cat_test_X[col].values[:] = 0
    processed_cat_test_X.index = test_X.index
    processed_cat_test_X = processed_cat_test_X.astype('int64')

    check_data(processed_cat_train_X, processed_cat_val_X, processed_cat_test_X)
    return processed_cat_train_X, processed_cat_val_X, processed_cat_test_X


def process_numerical_values(train_X, val_X, test_X, num_cols):
    """
    Process numerical missing values
    Use imputation to handle missing value, fit the transform on training data and apply it on validation and test
    :param train_X: training data
    :param val_X: validation data
    :param test_X: testing data
    :param num_cols: columns containing numerical values only
    :return: processed_num_train_X, processed_num_val_X, processed_num_test_X
    """
    imputer = SimpleImputer(strategy='median')

    num_train_X = train_X[num_cols]
    processed_num_train_X = pd.DataFrame(imputer.fit_transform(num_train_X))
    processed_num_train_X.columns = num_train_X.columns
    processed_num_train_X.index = train_X.index

    num_val_X = val_X[num_cols]
    processed_num_val_X = pd.DataFrame(imputer.transform(num_val_X))
    processed_num_val_X.columns = num_val_X.columns
    processed_num_val_X.index = val_X.index

    num_test_X = test_X[num_cols]
    processed_num_test_X = pd.DataFrame(imputer.transform(num_test_X))
    processed_num_test_X.columns = num_test_X.columns
    processed_num_test_X.index = test_X.index

    check_data(processed_num_train_X, processed_num_val_X, processed_num_test_X)
    return processed_num_train_X, processed_num_val_X, processed_num_test_X


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
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=ratio, test_size=1 - ratio)

    # Process the data
    num_cols = [col for col in train_X.columns if train_X[col].dtype != 'object']
    cat_cols = [col for col in train_X.columns if train_X[col].dtype == 'object']
    assert (len(num_cols) + len(cat_cols)) == train_X.shape[1]

    num_train_X, num_val_X, num_test_X = process_numerical_values(train_X, val_X, test_X, num_cols)
    cat_train_X, cat_val_X, cat_test_X = process_categorical_values(train_X, val_X, test_X, cat_cols)

    return num_train_X.join(cat_train_X.set_index(train_X.index), on=train_X.index), \
           num_val_X.join(cat_val_X.set_index(val_X.index), on=val_X.index), \
           train_y, \
           val_y, \
           num_test_X.join(cat_test_X.set_index(test_X.index), on=test_X.index)
