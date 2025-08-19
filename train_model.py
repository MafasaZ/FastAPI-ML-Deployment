# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Use clean feature names to match your API fields
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    bundle = {
        "pipeline": pipe,
        "feature_names": feature_names,
        "target_names": target_names,
        "problem_type": "classification",
        "model_type": "LogisticRegression",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "test_metrics": {
            "accuracy": float(acc)
        }
    }

    joblib.dump(bundle, "model.pkl")
    print("Saved model.pkl")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
