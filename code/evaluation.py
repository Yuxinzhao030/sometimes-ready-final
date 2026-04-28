import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


RESULTS_DIR = "results"
# add paths:
# RESULT_FILES = [
#     "results/tfidf_logistic_results.csv",
#     "results/xgboost_results.csv",
#     "results/sbert_results.csv",
#     "results/bert_results.csv",
# ]
RESULT_FILES = []

def evaluate_classification(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = None, None, None, None

    results = {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return pd.DataFrame([results])


def save_metrics(metrics_df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def collect_results(result_files=None):

    if result_files is None:
        result_files = RESULT_FILES

    if len(result_files) == 0:
        print("No result files provided yet.")
        print("Later, add result file paths to RESULT_FILES.")
        return pd.DataFrame()

    all_results = []

    for file_path in result_files:
        if not os.path.exists(file_path):
            print(f"Warning: file not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        all_results.append(df)

    if len(all_results) == 0:
        print("No valid result files were found.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    combined.to_csv(output_path, index=False)

    print(f"Saved combined results to {output_path}")
    return combined


def plot_model_comparison(results_df, metric="f1"):

    if results_df.empty:
        print("No results to plot.")
        return

    if "model" not in results_df.columns or metric not in results_df.columns:
        print(f"Cannot plot. Required columns: model and {metric}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(f"Model Comparison by {metric.upper()}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, f"model_comparison_{metric}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to {output_path}")


def main():
    combined_results = collect_results()
    if not combined_results.empty:
        plot_model_comparison(combined_results, metric="accuracy")
        plot_model_comparison(combined_results, metric="f1")


if __name__ == "__main__":
    main()