import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def preprocess(train_file, test_file, features, target):
    """
    Preprocess the data and extract training data, validation data and test data
    :param train_file: CSV file containing the training data
    :param test_file: CSV file containing the test data
    :param features: list of features (column names present in both training AND test data)
    :param target: column used as model target
    :return: train_X, val_X, train_y, val_y, test_X, test_data
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    y = train_data[target]
    X = train_data[features]

    test_X = test_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y)

    return train_X, val_X, train_y, val_y, test_X, test_data


def train(model, train_X, train_y, test_X, test_y):
    """
    Train the specified model and returns the trained model
    :param model: model to train
    :param train_X: features used to train the model
    :param train_y: target used to train the model
    :param test_X: features used for the prediction
    :param test_y: target used to compute the MAE
    :return: model
    """
    model.fit(train_X, train_y)

    mae = mean_absolute_error(model.predict(test_X), test_y)
    print("Validation MAE ({:}): {:,.0f}".format(model, mae))

    return model


def test(model, test_X, test_data):
    """
    Test the specified model and generate the CSV submission file
    :param model: trained model to test
    :param test_X: features used to test the model
    :return:
    """
    test_predictions = model.predict(test_X)

    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_predictions})
    output.to_csv("data/home_iowa/submission.csv", index=False)


###################
# DATA PROCESSING #
###################

# Data files
home_file = "data/home_iowa/train.csv"
test_file = "data/home_iowa/test.csv"
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
target = 'SalePrice'

train_X, val_X, train_y, val_y, test_X, test_data = preprocess(home_file, test_file, features, target)

##################
# MODEL TRAINING #
##################

# DecisionTreeRegressor
train(DecisionTreeRegressor(), train_X, train_y, val_X, val_y)

# DecisionTreeRegressor using best value for max_leaf_nodes
train(DecisionTreeRegressor(max_leaf_nodes=100), train_X, train_y, val_X, val_y)

# RandomForestRegressor
trained_model = train(RandomForestRegressor(), train_X, train_y, val_X, val_y)

###########
# TESTING #
###########

test(trained_model, test_X, test_data)