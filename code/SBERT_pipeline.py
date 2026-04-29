
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from sbert_features import encode_sbert
from gaussian_nb import get_gaussian_nb_predictions
from logistic_regression import get_logistic_regression_predictions
from XGBoost import get_xgboost_predictions


# ======================
# PATHS
# ======================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_DIR / "data" / "processed" / "test.csv"

RESULTS_CSV_DIR = PROJECT_DIR / "results" / "csv"
RESULTS_FIG_DIR = PROJECT_DIR / "results" / "figures"

RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# LOAD DATA
# ======================

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

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


# ======================
# EVALUATION
# ======================

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def add_model_name(metrics, model_name):
    metrics["model"] = model_name
    return metrics


# ======================
# SAVE RESULTS
# ======================

def save_results(results):
    json_path = RESULTS_CSV_DIR / "sbert_results.json"
    csv_path = RESULTS_CSV_DIR / "sbert_summary.csv"

    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)

    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


# ======================
# PLOTS
# ======================

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])

    path = RESULTS_FIG_DIR / f"{safe_name}_confusion.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[SAVED] {path}")


def plot_roc_curve(y_true, model_outputs):
    plt.figure(figsize=(7, 6))

    for output in model_outputs:
        fpr, tpr, _ = roc_curve(y_true, output["y_prob"])
        auc = roc_auc_score(y_true, output["y_prob"])

        plt.plot(
            fpr,
            tpr,
            label=f"{output['model']} (AUC={auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.title("ROC Curve - SBERT Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)

    path = RESULTS_FIG_DIR / "sbert_roc_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[SAVED] {path}")


# ======================
# MAIN PIPELINE
# ======================

def run_sbert_pipeline():
    X_train, y_train, X_test, y_test = load_data()

    print("\n" + "=" * 60)
    print("SBERT ENCODING")
    print("=" * 60)

    X_train_emb = encode_sbert(X_train)
    X_test_emb = encode_sbert(X_test)

    results = []
    roc_outputs = []

    # Logistic Regression
    print("\n" + "=" * 60)
    print("SBERT + LOGISTIC REGRESSION")
    print("=" * 60)

    log_pred, log_prob = get_logistic_regression_predictions(
        X_train_emb,
        y_train,
        X_test_emb,
    )

    log_metrics = add_model_name(
        evaluate(y_test, log_pred, log_prob),
        "SBERT + Logistic Regression",
    )

    print(log_metrics)

    results.append(log_metrics)
    roc_outputs.append(
        {
            "model": "SBERT Logistic",
            "y_prob": log_prob,
        }
    )

    plot_confusion_matrix(
        y_test,
        log_pred,
        "SBERT Logistic",
    )

    # XGBoost
    print("\n" + "=" * 60)
    print("SBERT + XGBOOST")
    print("=" * 60)

    xgb_pred, xgb_prob = get_xgboost_predictions(
        X_train_emb,
        y_train,
        X_test_emb,
    )

    xgb_metrics = add_model_name(
        evaluate(y_test, xgb_pred, xgb_prob),
        "SBERT + XGBoost",
    )

    print(xgb_metrics)

    results.append(xgb_metrics)
    roc_outputs.append(
        {
            "model": "SBERT XGBoost",
            "y_prob": xgb_prob,
        }
    )

    plot_confusion_matrix(
        y_test,
        xgb_pred,
        "SBERT XGBoost",
    )

    # Gaussian Naive Bayes
    print("\n" + "=" * 60)
    print("SBERT + GAUSSIAN NAIVE BAYES")
    print("=" * 60)

    nb_pred, nb_prob = get_gaussian_nb_predictions(
        X_train_emb,
        y_train,
        X_test_emb,
    )

    nb_metrics = add_model_name(
        evaluate(y_test, nb_pred, nb_prob),
        "SBERT + GaussianNB",
    )

    print(nb_metrics)

    results.append(nb_metrics)
    roc_outputs.append(
        {
            "model": "SBERT GaussianNB",
            "y_prob": nb_prob,
        }
    )

    plot_confusion_matrix(
        y_test,
        nb_pred,
        "SBERT GaussianNB",
    )

    save_results(results)
    plot_roc_curve(y_test, roc_outputs)

    print("\n" + "=" * 60)
    print("FINAL SBERT SUMMARY")
    print("=" * 60)
    print(pd.DataFrame(results))

    return results


if __name__ == "__main__":
    run_sbert_pipeline()
