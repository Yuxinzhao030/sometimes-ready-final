from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import evaluate_model


def run_logistic_regression(
    feature_dir="data/features",
    train_path="data/processed/train.csv",
    test_path="data/processed/test.csv",
):
    feature_dir = Path(feature_dir)

    X_train = joblib.load(feature_dir / "X_train_tfidf.pkl")
    X_test = joblib.load(feature_dir / "X_test_tfidf.pkl")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    results = evaluate_model(
        model=model,
        model_name="TF-IDF + Logistic Regression",
        save_name="tfidf_logistic",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    return results


if __name__ == "__main__":
    run_logistic_regression()
