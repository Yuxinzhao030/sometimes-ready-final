import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from XGBoost import get_xgboost_predictions


def load_data():
    """
    Load train/test CSV files.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    train_path = "../train_test/train.csv"
    test_path = "../train_test/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["full_text"].astype(str).values
    y_train = train_df["label"].values

    X_test = test_df["full_text"].astype(str).values
    y_test = test_df["label"].values

    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    print(f"Train size: {len(X_train):,}, Fake ratio: {y_train.mean():.2%}")
    print(f"Test size: {len(X_test):,}, Fake ratio: {y_test.mean():.2%}")
    print("=" * 60)

    return X_train, y_train, X_test, y_test
