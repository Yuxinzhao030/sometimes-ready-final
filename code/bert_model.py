import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

RESULTS_DIR = "results/csv"
FIGURES_DIR = "results/figures"
MODEL_DIR = "models/bert_fake_news"

MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"
LABEL_COL = "label"


def load_train_test_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_cols = [TEXT_COL, LABEL_COL]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in train.csv")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in test.csv")

    train_df = train_df[[TEXT_COL, LABEL_COL]].dropna()
    test_df = test_df[[TEXT_COL, LABEL_COL]].dropna()

    train_df[TEXT_COL] = train_df[TEXT_COL].astype(str)
    test_df[TEXT_COL] = test_df[TEXT_COL].astype(str)

    train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)
    test_df[LABEL_COL] = test_df[LABEL_COL].astype(int)

    return train_df, test_df


def tokenize_data(train_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(batch):
        return tokenizer(
            batch[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column(LABEL_COL, "labels")
    test_dataset = test_dataset.rename_column(LABEL_COL, "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"],
    )

    display.plot(values_format="d")
    plt.title("BERT Confusion Matrix")
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "bert_confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cm


def save_metrics_bar_chart(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = [results[m] for m in metrics]

    plt.figure(figsize=(7, 5))
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Fine-tuned BERT Performance")
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "bert_metrics_bar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_prediction_distribution(y_true, y_pred):
    pred_df = pd.DataFrame({
        "True Label": y_true,
        "Predicted Label": y_pred,
    })

    counts = pred_df.value_counts().reset_index(name="count")
    counts["group"] = (
        "True " + counts["True Label"].astype(str)
        + " / Pred " + counts["Predicted Label"].astype(str)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(counts["group"], counts["count"])
    plt.ylabel("Count")
    plt.title("BERT Prediction Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "bert_prediction_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_bert_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_df, test_df = load_train_test_data()

    print("Tokenizing data...")
    train_dataset, test_dataset, tokenizer = tokenize_data(train_df, test_df)

    print("Loading BERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="models/bert_logs",
        logging_steps=500,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training BERT...")
    trainer.train()

    print("Evaluating BERT...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    cm = save_confusion_matrix(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    results = {
        "model": "Fine-tuned DistilBERT",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
    }

    save_metrics_bar_chart(results)
    save_prediction_distribution(y_true, y_pred)

    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(RESULTS_DIR, "bert_results.csv"), index=False)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR, "bert_predictions.csv"), index=False)

    print("BERT results:")
    print(results_df)

    return results_df


if __name__ == "__main__":
    run_bert_pipeline()
