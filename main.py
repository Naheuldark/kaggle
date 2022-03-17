from data_processing import *
from model_training import *
from data_analysis import *

# From https://www.kaggle.com/naheuldark/upd-interm-ml-course-wrap-up-mae-13031-top-2/edit

####################
# DATA ENGINEERING #
####################

# Data files
home_file = "data/home_iowa/train.csv"
test_file = "data/home_iowa/test.csv"
target = 'SalePrice'
index = 'Id'

# Data Analysis
data_analysis(home_file, index, target)

# Data Processing
train_X, train_y, test_X, preprocessor = preprocess(home_file, test_file, target, index)

##################
# MODEL TRAINING #
##################

# Optimize and fine tune model hyper-parameters
# ---------------------------------------------
# model = XGBRegressor(learning_rate=0.01, n_jobs=4)
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', model)
# ])
# best_params = optimize(pipeline, train_X, train_y)

best_params = {
    'n_estimators': 1916,
    'learning_rate': 0.01,
    'max_depth': 5,
    'min_child_weight': 0,
    'subsample': 0.35,
    'colsample_bytree': 0.4
}

best_model = XGBRegressor(**best_params, n_jobs=4, random_state=0)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

pipeline.fit(train_X, train_y)

###########
# TESTING #
###########

test(pipeline, test_X, target, index, "data/home_iowa")
