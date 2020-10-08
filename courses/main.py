import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def preprocess(train_file, test_file, features, target):
    """
    Preprocess the data and extract training data, validation data and test data
    :return: train_X, val_X, train_y, val_y, test_X
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    y = train_data[target]
    X = train_data[features]

    test_X = test_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y)

    return train_X, val_X, train_y, val_y, test_X


def train(model):
    """
    Train the specified model and returns the prediction
    :return: prediction
    """


###################
# DATA PROCESSING #
###################

# Data files
home_file = "data/home_iowa/train.csv"
test_file = "data/home_iowa/test.csv"
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
target = 'SalePrice'

train_X, val_X, train_y, val_y, test_X = preprocess(home_file, test_file, features, target)

##################
# MODEL TRAINING #
##################

# DecisionTreeRegressor
iowa_model = DecisionTreeRegressor()
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE (DecisionTreeRegressor): {:,.0f}".format(val_mae))

# DecisionTreeRegressor using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100)
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE (DecisionTreeRegressor w/ max_leaf_nodes=100): {:,.0f}".format(val_mae))

# RandomForestRegressor
iowa_model = RandomForestRegressor()
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE (RandomForestRegressor): {:,.0f}".format(val_mae))

###########
# TESTING #
###########

test_predictions = iowa_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_predictions})
output.to_csv("data/home_iowa/submission.csv", index=False)
