# Iris ML Model API (FastAPI)

A minimal FastAPI service that serves a trained Iris classifier as a web API.

## Problem Description
Predict Iris species (setosa, versicolor, virginica) from 4 numerical features:
- sepal_length, sepal_width, petal_length, petal_width (cm)

Dataset: `sklearn.datasets.load_iris()` (built-in)

## Model Choice
- **Pipeline**: StandardScaler → LogisticRegression
- Rationale: fast to train, robust baseline, supports `predict_proba` for confidence.
- Test Accuracy (example): ~0.93–1.00 depending on split.

## Project Structure

-main.py
-train_model.py
-model.pkl
-requirements.txt
-README.md
