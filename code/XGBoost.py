import pandas as pd
import numpy as np
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt


# 1. Data Loading
def load_data():
    """Load train/test CSV files. Returns (X_train, y_train, X_test, y_test)."""
    train_path = "train_test/train.csv"
    test_path = "train_test/test.csv"

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


# 2. TF-IDF Vectorization
# TODO: After Person 1 completes data.py, replace this entire function with:
#       from src.data import build_tfidf

def build_tfidf(X_train, X_test, max_features=5000):
    """Convert text to TF-IDF vectors. Returns (X_train_vec, X_test_vec, vectorizer)."""
    print("\n" + "=" * 60)
    print("TF-IDF VECTORIZATION")
    print("=" * 60)
    print(f"max_features: {max_features}")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Feature dimension: {X_train_vec.shape[1]:,}")
    print("=" * 60)

    return X_train_vec, X_test_vec, vectorizer


# 3. XGBoost Training
def train_xgboost(X_train_vec, y_train, X_test_vec, y_test, params=None):
    """Train XGBoost, evaluate on test set. Returns (model, metrics, y_pred, y_pred_proba)."""
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_vec, y_train)

    print("=" * 60)

    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\nRESULTS:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 60)

    return model, metrics, y_pred, y_pred_proba


# 4. Interface for Person 3 (SBERT + XGBoost)
def get_xgboost_predictions(X_train_vec, y_train, X_test_vec):
    """Person 3: Train XGBoost on SBERT vectors. Returns (y_pred, y_pred_proba)."""
    params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

    return y_pred, y_pred_proba


# 5. Complete Pipeline
def run_tfidf_xgboost(save_results=True):
    """Run full pipeline: load data -> TF-IDF -> XGBoost -> save results."""
    print("\n" + "=" * 70)
    print("TF-IDF + XGBoost Pipeline")
    print("=" * 70)

    # 5.1 Load data
    X_train, y_train, X_test, y_test = load_data()

    # 5.2 TF-IDF vectorization
    # TODO: After Person 1 completes data.py, replace with:
    #       X_train_vec, X_test_vec = build_tfidf(X_train, X_test)
    X_train_vec, X_test_vec, vectorizer = build_tfidf(X_train, X_test)

    # 5.3 Train and evaluate
    model, metrics, y_pred, y_pred_proba = train_xgboost(
        X_train_vec, y_train, X_test_vec, y_test
    )

    # 5.4 Save results
    if save_results:
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        with open("results/xgboost_results.json", "w") as f:
            json.dump(metrics, f, indent=4)

        joblib.dump(model, "models/xgboost_model.pkl")
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

        print("\n[SAVED] results/xgboost_results.json")
        print("[SAVED] models/xgboost_model.pkl")
        print("[SAVED] models/tfidf_vectorizer.pkl")

    return metrics, model, y_test, y_pred, y_pred_proba


# 6. Visualization
def plot_all_results(model, y_test, y_pred, y_pred_proba, save_dir="results"):
    """Generate confusion matrix, ROC curve, and feature importance plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # Plot 2: ROC Curve
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"XGBoost (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve.png", dpi=150)
    plt.close()

    # Plot 3: Feature Importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(
        model, max_num_features=30, importance_type="weight", ax=plt.gca()
    )
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png", dpi=150)
    plt.close()

    print(f"\n[SAVED] 3 plots to {save_dir}/")


# 7. Main Entry Point
if __name__ == "__main__":
    metrics, model, y_test, y_pred, y_pred_proba = run_tfidf_xgboost(
        save_results=True
    )
    plot_all_results(model, y_test, y_pred, y_pred_proba)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 70)