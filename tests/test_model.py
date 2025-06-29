# tests/test_model.py

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


def test_model_prediction():
    # Load model and encoder
    model = joblib.load("models/model.joblib")
    encoder = joblib.load("models/encoder.joblib")

    # Load sample data
    iris = load_iris(as_frame=True)
    df = iris.frame
    X_sample = df.drop("target", axis=1).iloc[:5]

    # Predict
    preds = model.predict(X_sample)

    # Assertions
    assert len(preds) == 5
    assert preds.dtype in [int, float]
