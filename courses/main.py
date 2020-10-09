from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from courses.data_processing import preprocess
from courses.model_training import train, test

###################
# DATA PROCESSING #
###################

# Data files
home_file = "data/home_iowa/train.csv"
test_file = "data/home_iowa/test.csv"
target = 'SalePrice'
index = 'Id'

train_X, val_X, train_y, val_y, test_X = preprocess(home_file, test_file, target, index, 0.8)

##################
# MODEL TRAINING #
##################

models = [
    # DecisionTreeRegressor
    train(DecisionTreeRegressor(), train_X, train_y, val_X, val_y),
    train(DecisionTreeRegressor(max_leaf_nodes=100), train_X, train_y, val_X, val_y),

    # RandomForestRegressor
    train(RandomForestRegressor(), train_X, train_y, val_X, val_y),
    train(RandomForestRegressor(n_estimators=50), train_X, train_y, val_X, val_y),
    train(RandomForestRegressor(n_estimators=100), train_X, train_y, val_X, val_y),
    train(RandomForestRegressor(n_estimators=100, criterion='mae'), train_X, train_y, val_X, val_y),
    train(RandomForestRegressor(n_estimators=200, min_samples_split=20), train_X, train_y, val_X, val_y),
    train(RandomForestRegressor(n_estimators=100, max_depth=7), train_X, train_y, val_X, val_y)
]

###########
# TESTING #
###########

test(models, test_X, target, index, "data/home_iowa")
