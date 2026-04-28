from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_DIR = Path("data/features")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


X_train = joblib.load(FEATURE_DIR / "X_train_tfidf.pkl")
X_test = joblib.load(FEATURE_DIR / "X_test_tfidf.pkl")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["label"]
y_test = test_df["label"]


model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("TF-IDF + Logistic Regression Results")
print("-----------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


results = pd.DataFrame(
    {
        "model": ["TF-IDF + Logistic Regression"],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "roc_auc": [roc_auc],
    }
)

results.to_csv(RESULTS_DIR / "tfidf_logistic_results.csv", index=False)
joblib.dump(model, RESULTS_DIR / "tfidf_logistic_model.pkl")

print("\nSaved results to results/tfidf_logistic_results.csv")
print("Saved model to results/tfidf_logistic_model.pkl")
