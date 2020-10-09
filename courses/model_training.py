import math
import pandas as pd
from sklearn.metrics import mean_absolute_error


def train(model, train_X, train_y, test_X, test_y):
    """
    Train the specified model and returns the trained model
    :param model: model to train
    :param train_X: features used to train the model
    :param train_y: target used to train the model
    :param test_X: features used for the prediction
    :param test_y: target used to compute the MAE
    :return: model, mae
    """
    model.fit(train_X, train_y)

    mae = mean_absolute_error(model.predict(test_X), test_y)
    print("Validation MAE: {:,.0f} ({:})".format(mae, model))

    return model, mae


def compute_best_model(models):
    """
    Compute the best model based on the MAE
    :param models: array of all models as tuple (trained model, MAE)
    :return: the model with the lowest MAE
    """
    best_mae = math.inf
    best_model = None
    for model, mae in models:
        if mae < best_mae:
            best_mae = mae
            best_model = model

    assert best_model is not None
    print("\nUsing model {:} with lowest MAE {:,.0f}".format(best_model, best_mae))

    return best_model


def test(models, test_X, target, index, out_path):
    """
    Test the specified model and generate the CSV submission file
    :param models: trained model to test
    :param test_X: features used to test the model
    :param target: column used as model target
    :param index: column used to index data
    :param out_path: path to store the submission CSV file
    """
    best_model = compute_best_model(models)
    test_predictions = best_model.predict(test_X)

    output = pd.DataFrame({index: test_X.index,
                           target: test_predictions})
    output.to_csv(out_path + "/submission.csv", index=False)
