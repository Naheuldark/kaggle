import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def compute_preprocessor(numerical_cols, categorical_cols):
    """
    Compute the preprocessor for the data
    :param numerical_cols: columns containing numerical data
    :param categorical_cols: columns containing categorical data with low cardinality
    :return: preprocessor
    """
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')

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

    # Read the data
    train_data = pd.read_csv(train_file, index_col=index)
    test_data = pd.read_csv(test_file, index_col=index)

    # Remove rows with missing target
    train_data.dropna(axis=0, subset=[target], inplace=True)
    y = train_data[target]

    # Separate target from predictors
    X = train_data.drop([target], axis=1)

    # Split training data into training and validation using [ratio]
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=ratio, test_size=1 - ratio)

    # Process the data
    # Select numerical columns
    numerical_cols = [col for col in train_X.columns if
                      train_X[col].dtype in ['int64', 'float64']]

    # Select categorical columns with relatively low cardinality
    categorical_cols = [col for col in train_X.columns if
                        train_X[col].nunique() < 10 and
                        train_X[col].dtype == 'object']

    return train_X[categorical_cols + numerical_cols].copy(), \
           val_X[categorical_cols + numerical_cols].copy(), \
           train_y, \
           val_y, \
           test_data[categorical_cols + numerical_cols].copy(), \
           compute_preprocessor(numerical_cols, categorical_cols)
