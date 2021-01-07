from courses.data_processing import *
from courses.model_training import *
from courses.data_analysis import *

###################
# DATA PROCESSING #
###################

# Data files
home_file = "data/home_iowa/train.csv"
test_file = "data/home_iowa/test.csv"
target = 'SalePrice'
index = 'Id'

train_X, train_y, test_X, preprocessor = preprocess(home_file, test_file, target, index)

data_analysis(home_file, index, target)

##################
# MODEL TRAINING #
##################

model = XGBRegressor(learning_rate=0.01, n_jobs=4)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Optimize and fine tune model hyper-parameters
best_model = optimize(pipeline, train_X, train_y)
# params = {
#     'n_estimators': 1916,
#     'learning_rate': 0.01,
#     'max_depth': 5,
#     'min_child_weight': 0,
#     'subsample': 0.35,
#     'colsample_bytree': 0.4
# }

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

pipeline.fit(train_X, train_y)

###########
# TESTING #
###########

test(pipeline, test_X, target, index, "data/home_iowa")
