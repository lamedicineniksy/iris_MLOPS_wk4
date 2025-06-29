from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_and_evaluate_model():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target


    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
   


    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    
    os.makedirs("models", exist_ok=True)

    
    
    joblib.dump(model, "models/model.joblib")
    joblib.dump(encoder, "models/encoder.joblib")

    return accuracy

if __name__ == "__main__":
    acc = train_and_evaluate_model()
    print(f"Model accuracy: {acc}")
