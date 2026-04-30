import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

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


FIGURES_DIR = Path("results/figures")


def save_confusion_matrix(y_true, y_pred, model_name, save_name):
    """Save a confusion matrix plot for one model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=True,
        annot_kws={"size": 12},
    )

    plt.title(model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    save_path = FIGURES_DIR / f"{save_name}_confusion_matrix.png"

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[SAVED] {save_path}")


def evaluate_predictions(
    y_test,
    y_pred,
    y_prob,
    model_name,
    save_name,
    results_dir=RESULTS_DIR,
):
    """Evaluate model predictions, save metrics to CSV, and save a confusion matrix plot."""
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

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        save_name=save_name,
    )
    print(f"\nSaved results to {save_path}")
    return results


def collect_results(results_dir=RESULTS_DIR):
    """Collect all model result CSV files and create a summary results table."""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results folder does not exist: {results_dir}")
        return pd.DataFrame()

    all_csv_files = list(results_dir.glob("*.csv"))

    skip_files = {
        "summary_results.csv",
        "bert_predictions.csv",
    }

    all_results = []

    for file in all_csv_files:
        if file.name in skip_files:
            print(f"Skipped {file}")
            continue

        try:
            df = pd.read_csv(file)

            df.columns = [col.strip().lower() for col in df.columns]

            if "f1_score" in df.columns and "f1" not in df.columns:
                df = df.rename(columns={"f1_score": "f1"})

            required_metrics = {"accuracy", "precision", "recall", "f1"}
            if not required_metrics.issubset(set(df.columns)):
                print(f"Skipped {file}: not a classification result file")
                continue

            if "model" not in df.columns:
                df["model"] = (
                    file.stem
                    .replace("_results", "")
                    .replace("_summary", "")
                    .replace("_", " ")
                    .title()
                )

            df["source_file"] = file.name
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

    summary["model"] = summary["model"].astype(str).str.strip()

    summary = summary.sort_values(by="f1", ascending=False)
    summary = summary.drop_duplicates(subset=["model"], keep="first")

    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary results to {SUMMARY_PATH}")

    return summary


def print_best_model(summary_df):
    """Print the best model based on F1 score."""
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


def plot_roc_curve(y_true, model_outputs, title, save_name):
    """Plot and save ROC curves for multiple models using predicted probabilities."""
    results_fig_dir = Path("results/figures")
    results_fig_dir.mkdir(parents=True, exist_ok=True)

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
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)

    path = results_fig_dir / save_name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[SAVED] {path}")


def print_top_tfidf_features(model, vectorizer, top_n=20):
    """Print top TF-IDF features associated with real and fake news."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_fake_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_real_idx = np.argsort(coefficients)[:top_n]

    print("\n" + "=" * 60)
    print("Top TF-IDF Features for Fake News")
    print("=" * 60)

    for feature, coef in zip(
        feature_names[top_fake_idx],
        coefficients[top_fake_idx],
    ):
        print(f"{feature:<30} {coef:.4f}")

    print("\n" + "=" * 60)
    print("Top TF-IDF Features for Real News")
    print("=" * 60)

    for feature, coef in zip(
        feature_names[top_real_idx],
        coefficients[top_real_idx],
    ):
        print(f"{feature:<30} {coef:.4f}")


def main():
    summary = collect_results()
    print_best_model(summary)

    if not summary.empty:
        print("\nFinal Model Comparison:")
        print(summary)


if __name__ == "__main__":
    main()
