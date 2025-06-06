import pytest

# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data

data = pd.read_csv("data/census.csv")
cat_features = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country"
]
sample = data.sample(n=100, random_state=42)

X, y, encoder, lb = process_data(
    sample,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_type():
    """
    Tests that train_model returns a RandomForestClassifier
    """
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    


# TODO: implement the second test. Change the function name and input as needed
def test_inference_output_shape():
    """
    Tests that inference returns a prediction array of the correct length
    """
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test compute model metrics returns float between 0 and 1.
    """
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    for metric in [precision, recall, fbeta]:
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0
