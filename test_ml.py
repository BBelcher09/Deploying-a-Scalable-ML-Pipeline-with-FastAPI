import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def mock_data():
    """
    Fixture to create mock data for testing
    """
    # 5 samples with 3 features simulating age and two categorical variables
    X = np.array ([[25, 1, 0],
                   [30, 0, 1],
                   [22, 1, 0],
                   [35, 0, 1],
                   [28, 1, 0]
    ])
    #Binary targets
    y = np.array([0, 1, 0, 1, 0])
    return X, y
                   
def test_train_model_algorithm(mock_data):
    """
    Test the train_model function with mock data to see if it returns a trained RandomForestClassifier model.
    """
    X, y = mock_data
    model = train_model(X, y)

    #Assert that the returned model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "The model returned is not a RandomForestClassifier instance."

def test_inference_output(mock_data):
    """
    Test the inference function with mock data to see if it returns the expected output.
    """
    X, y = mock_data
    model = train_model(X, y)
    preds = inference(model, X)

    #Assert type is a numpy array and has the same length as input data
    assert isinstance(preds, np.ndarray), "The predictions are not a numpy array."
    assert len(preds) == len(X), "The number of predictions does not match the number of input samples."

def test_compute_model_metrics_values():
    """
    Test the compute_model_metrics function with mock data to see if it returns the expected metrics.
    """
    # Mock predictions and true labels
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Assert that the metrics are within expected ranges
    assert precision == 1.0, f"Expected precision of 1.0, but got {precision}."
    assert recall == 1.0, f"Expected recall of 1.0, but got {recall}."
    assert fbeta == 1.0, f"Expected fbeta of 1.0, but got {fbeta}."
    

