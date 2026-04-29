import pandas as pd
import numpy as np
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

from tfidf_features import build_tfidf



# 1. XGBoost Training

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
            "tree_method": "hist", 
            "n_jobs": 2,
        }

    print("\n" + "=" * 60)
    print("TF-IDF + XGBoost Results")
    print("---")

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))
    print("=" * 60)

    return model, metrics, y_pred, y_pred_proba


# 2. Prepare XGBoost for SBERT

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


# 3. Complete Pipeline

def run_tfidf_xgboost(save_results=True):
    """Run full pipeline: TF-IDF (Person1) -> XGBoost -> save results."""
    print("\n" + "=" * 70)
    print("TF-IDF + XGBoost Pipeline")
    print("=" * 70)

    X_train_vec, y_train, X_test_vec, y_test = build_tfidf()

    model, metrics, y_pred, y_pred_proba = train_xgboost(
        X_train_vec, y_train, X_test_vec, y_test
    )

    if save_results:
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        with open("results/xgboost_results.json", "w") as f:
            json.dump(metrics, f, indent=4)

        joblib.dump(model, "models/xgboost_model.pkl")

        print("\n[SAVED] results/xgboost_results.json")
        print("[SAVED] models/xgboost_model.pkl")

    return metrics, model, y_test, y_pred, y_pred_proba


# 4. Visualization

def plot_all_results(model, y_test, y_pred, y_pred_proba, save_dir="results"):
    """Generate confusion matrix, ROC curve, and feature importance plots."""
    os.makedirs(save_dir, exist_ok=True)

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
    plt.savefig(f"{save_dir}/xgboost_confusion_matrix.png", dpi=150)
    plt.close()

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
    plt.savefig(f"{save_dir}/xgboost_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=30, importance_type="weight", ax=plt.gca())
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/xgboost_feature_importance.png", dpi=150)
    plt.close()

    print(f"\n[SAVED] 3 plots to {save_dir}/")


# 5. Main Entry Point

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