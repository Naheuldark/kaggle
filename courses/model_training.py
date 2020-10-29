import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor, DMatrix, cv


def optimize(pipeline, train_X, train_y):
    """
    Optimize XGBoost Regressor hyper-parameters

    :param pipeline: pipeline containing the initial model
    :param train_X: training data
    :param train_y: training targets
    :return:
    """
    parameter_space = {}
    fine_tune_range = [-1, 0, 1]
    fine_fine_tune_range = [-0.1, -0.05, 0.0, 0.05, 0.1]

    # 1. Find the optimal number of estimators
    # ----------------------------------------
    # Search for the best number of estimators within 200 to 2000 in steps of 200.
    parameter_space['model__n_estimators'] = [n for n in range(150, 3001, 150)]
    print("Initial parameter search space: ", parameter_space)

    # Initializing the grid search.
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    grid_search = GridSearchCV(pipeline,
                               param_grid=parameter_space,
                               scoring='neg_mean_absolute_error',
                               cv=folds,
                               n_jobs=4,
                               verbose=1)

    grid_search.fit(train_X, train_y)
    print("Best found parameter values: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print()

    # Fix n_estimators to the best found value
    parameter_space['model__n_estimators'] = [grid_search.best_params_['model__n_estimators']]

    # 2.1 Find the best combination of max_depth and min_child_weight
    # ---------------------------------------------------------------
    # Add max_depth and min_child_weight with possible values 1, 4, 7 each to the search.
    parameter_space['model__max_depth'] = [x for x in [1, 4, 7]]
    parameter_space['model__min_child_weight'] = [x for x in [1, 4, 7]]
    print("Updated parameter space: ", parameter_space)

    grid_search.fit(train_X, train_y)
    print("Best found parameter values: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print()

    # 2.2 Fine tune the combination of max_depth and min_child_weight
    # ---------------------------------------------------------------
    parameter_space['model__max_depth'] = [grid_search.best_params_['model__max_depth'] + i
                                           for i in fine_tune_range]
    parameter_space['model__min_child_weight'] = [grid_search.best_params_['model__min_child_weight'] + i
                                                  for i in fine_tune_range]
    print("Parameter search space: ", parameter_space)

    grid_search.fit(train_X, train_y)
    print("Best found parameter values: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print()

    # Fix max_depth and min_child_weight with the best found values
    parameter_space['model__max_depth'] = [grid_search.best_params_['model__max_depth']]
    parameter_space['model__min_child_weight'] = [grid_search.best_params_['model__min_child_weight']]

    # 3.1 Find the best combination of subsample and colsample_bytree
    # ---------------------------------------------------------------
    # Add subsample and colsample_bytree with possible values 0.6 and 0.9 each.
    parameter_space['model__subsample'] = [x for x in [0.3, 0.6, 0.9]]
    parameter_space['model__colsample_bytree'] = [x for x in [0.3, 0.6, 0.9]]
    print("Parameter search space: ", parameter_space)

    grid_search.fit(train_X, train_y)
    print("Best found parameter values: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print()

    # 3.2 Fine tune the combination of subsample and colsample_bytree
    # ---------------------------------------------------------------
    parameter_space['model__subsample'] = [grid_search.best_params_['model__subsample'] + i
                                           for i in fine_fine_tune_range]
    parameter_space['model__colsample_bytree'] = [grid_search.best_params_['model__colsample_bytree'] + i
                                                  for i in fine_fine_tune_range]
    print("Parameter search space: ", parameter_space)

    grid_search.fit(train_X, train_y)

    parameter_space['model__subsample'] = [grid_search.best_params_['model__subsample']]
    parameter_space['model__colsample_bytree'] = [grid_search.best_params_['model__colsample_bytree']]

    # 4. Find exact optimal of estimators using early_stopping
    # --------------------------------------------------------
    print("Parameter search space: ", parameter_space)

    # Setting up parameter dict with found optimal values
    params = {
        'max_depth': parameter_space['model__max_depth'][0],
        'min_child_weight': parameter_space['model__min_child_weight'][0],
        'eta': 0.01,  # learning rate
        'subsample': parameter_space['model__subsample'][0],
        'colsample_bytree': parameter_space['model__colsample_bytree'][0]
    }

    cv_results = cv(params,
                    DMatrix(pd.get_dummies(train_X), label=train_y),
                    num_boost_round=10000,
                    seed=0,
                    nfold=5,
                    metrics='mae',
                    early_stopping_rounds=100
                    )

    mae = cv_results['test-mae-mean'].min()
    opt_n_estimators = cv_results['test-mae-mean'].argmin()

    print("Optimal number of estimators: ", opt_n_estimators)
    print("Score: ", mae)

    params = {
        'n_estimators': opt_n_estimators,
        'learning_rate': 0.01,
        'max_depth': parameter_space['model__max_depth'][0],
        'min_child_weight': parameter_space['model__min_child_weight'][0],
        'subsample': parameter_space['model__subsample'][0],
        'colsample_bytree': parameter_space['model__colsample_bytree'][0]
    }
    print(">>> Final hyper-parameters: ", params)

    return XGBRegressor(**params, n_jobs=4, random_state=0)


def test(model, test_X, target, index, out_path):
    """
    Test the specified model and generate the CSV submission file

    :param model: trained model to test
    :param test_X: features used to test the model
    :param target: column used as model target
    :param index: column used to index data
    :param out_path: path to store the submission CSV file
    """
    test_predictions = model.predict(test_X)

    output = pd.DataFrame({index: test_X.index,
                           target: test_predictions})
    output.to_csv(out_path + "/submission.csv", index=False)
