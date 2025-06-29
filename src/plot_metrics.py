# src/plot_metrics.py

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def plot_metrics():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load trained model
    model = joblib.load("model.joblib")
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("metrics.png")

    # Write report
    with open("report.md", "w") as f:
        f.write(f"# Model Report\n\n")
        f.write(f"**Accuracy:** {acc:.2f}\n\n")
        f.write(f"![Confusion Matrix](metrics.png)")

if __name__ == "__main__":
    plot_metrics()
