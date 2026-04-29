import os
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Relative paths from the project root
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
RESULTS_DIR = "results/csv"
MODEL_DIR = "models/bert_fake_news"

MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "full_text"
LABEL_COL = "label"


def load_train_test_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Load local train/test CSV files.

    Expected columns:
    - full_text: news article text
    - label: 1 = fake news, 0 = real news
    """

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


def run_bert_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
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
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="results/models/bert_logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
    eval_results = trainer.evaluate()

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    cm = confusion_matrix(y_true, y_pred)

    results = {
        "model": "Fine-tuned DistilBERT",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )[1],
        "f1": precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )[2],
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
    }

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