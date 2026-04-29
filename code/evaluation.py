import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RESULTS_DIR = Path("results/csv")
SUMMARY_PATH = RESULTS_DIR / "summary_results.csv"


def evaluate_predictions(
    y_test,
    y_pred,
    y_prob,
    model_name,
    save_name,
    results_dir=RESULTS_DIR,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    else:
        print("ROC-AUC:   N/A")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    results = pd.DataFrame(
        {
            "model": [model_name],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "roc_auc": [roc_auc],
        }
    )

    save_path = results_dir / f"{save_name}_results.csv"
    results.to_csv(save_path, index=False)

    print(f"\nSaved results to {save_path}")
    return results


def collect_results(results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results folder does not exist: {results_dir}")
        return pd.DataFrame()

    result_files = list(results_dir.glob("*_results.csv"))

    result_files = [
        file for file in result_files
        if file.name != "summary_results.csv"
    ]

    if len(result_files) == 0:
        print("No result CSV files found.")
        return pd.DataFrame()

    all_results = []

    for file in result_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if len(all_results) == 0:
        print("No valid result files were loaded.")
        return pd.DataFrame()

    summary = pd.concat(all_results, ignore_index=True)

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for col in metric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    if "f1" in summary.columns:
        summary = summary.sort_values(by="f1", ascending=False)

    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary results to {SUMMARY_PATH}")

    return summary


def print_best_model(summary_df):
    if summary_df.empty:
        print("No summary results available.")
        return

    if "f1" not in summary_df.columns:
        print("F1 column not found. Cannot select best model.")
        return

    best = summary_df.sort_values(by="f1", ascending=False).iloc[0]

    print("\n" + "=" * 60)
    print("Best Model Based on F1 Score")
    print("=" * 60)
    print(f"Model:     {best['model']}")
    print(f"Accuracy:  {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall:    {best['recall']:.4f}")
    print(f"F1-score:  {best['f1']:.4f}")

    if "roc_auc" in best.index and pd.notna(best["roc_auc"]):
        print(f"ROC-AUC:   {best['roc_auc']:.4f}")


def main():
    summary = collect_results()
    print_best_model(summary)

    if not summary.empty:
        print("\nFinal Model Comparison:")
        print(summary)


if __name__ == "__main__":
    main()