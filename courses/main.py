from courses.data_processing import *
from courses.model_training import *

###################
# DATA PROCESSING #
###################

# Data files
home_file = "data/home_iowa/train.csv"
test_file = "data/home_iowa/test.csv"
target = 'SalePrice'
index = 'Id'

train_X, train_y, test_X, preprocessor = preprocess_xgboost(home_file, test_file, target, index)

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

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

pipeline.fit(train_X, train_y)

###########
# TESTING #
###########

test(model, test_X, target, index, "data/home_iowa")
