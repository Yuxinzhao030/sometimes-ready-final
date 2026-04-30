from data_cleaning import clean_data
from tfidf_features import build_tfidf

from logistic_regression import get_logistic_regression_predictions
from naive_bayes import get_naive_bayes_predictions
from gaussian_nb import get_gaussian_nb_predictions
from XGBoost import get_xgboost_predictions

from sbert_features import encode_sbert
from bert_model import run_bert_pipeline
from anomaly_detection import run_anomaly_detection

from evaluation import (
    evaluate_predictions,
    collect_results,
    print_best_model,
    plot_roc_curve,
)


print("=" * 60)
print("Step 1: Data Cleaning")
print("=" * 60)

train_df, test_df = clean_data()


print("\n" + "=" * 60)
print("Step 2: Build TF-IDF Features")
print("=" * 60)

X_train_tfidf, y_train, X_test_tfidf, y_test = build_tfidf()


print("\n" + "=" * 60)
print("Step 3: TF-IDF + Logistic Regression")
print("=" * 60)

y_pred_tfidf_logistic, y_prob_tfidf_logistic = (
    get_logistic_regression_predictions(
        X_train_tfidf,
        y_train,
        X_test_tfidf,
    )
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_tfidf_logistic,
    y_prob=y_prob_tfidf_logistic,
    model_name="TF-IDF + Logistic Regression",
    save_name="tfidf_logistic",
)


print("\n" + "=" * 60)
print("Step 4: TF-IDF + MultinomialNB")
print("=" * 60)

y_pred_tfidf_nb, y_prob_tfidf_nb = (
    get_naive_bayes_predictions(
        X_train_tfidf,
        y_train,
        X_test_tfidf,
    )
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_tfidf_nb,
    y_prob=y_prob_tfidf_nb,
    model_name="TF-IDF + MultinomialNB",
    save_name="tfidf_multinomial_nb",
)


print("\n" + "=" * 60)
print("Step 5: TF-IDF + XGBoost")
print("=" * 60)

y_pred_tfidf_xgb, y_prob_tfidf_xgb = (
    get_xgboost_predictions(
        X_train_tfidf,
        y_train,
        X_test_tfidf,
    )
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_tfidf_xgb,
    y_prob=y_prob_tfidf_xgb,
    model_name="TF-IDF + XGBoost",
    save_name="tfidf_xgboost",
)

plot_roc_curve(
    y_true=y_test,
    model_outputs=[
        {"model": "Logistic Regression", "y_prob": y_prob_tfidf_logistic},
        {"model": "MultinomialNB", "y_prob": y_prob_tfidf_nb},
        {"model": "XGBoost", "y_prob": y_prob_tfidf_xgb},
    ],
    title="ROC Curve - TF-IDF Models",
    save_name="tfidf_roc_curve.png",
)

print("\n" + "=" * 60)
print("Step 6: Build SBERT Features")
print("=" * 60)

X_train_text = train_df["text"].astype(str).values
X_test_text = test_df["text"].astype(str).values

X_train_sbert = encode_sbert(X_train_text)
X_test_sbert = encode_sbert(X_test_text)


print("\n" + "=" * 60)
print("Step 7: SBERT + Logistic Regression")
print("=" * 60)

y_pred_sbert_logistic, y_prob_sbert_logistic = (
    get_logistic_regression_predictions(
        X_train_sbert,
        y_train,
        X_test_sbert,
    )
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_sbert_logistic,
    y_prob=y_prob_sbert_logistic,
    model_name="SBERT + Logistic Regression",
    save_name="sbert_logistic",
)


print("\n" + "=" * 60)
print("Step 8: SBERT + XGBoost")
print("=" * 60)

y_pred_sbert_xgb, y_prob_sbert_xgb = (
    get_xgboost_predictions(
        X_train_sbert,
        y_train,
        X_test_sbert,
    )
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_sbert_xgb,
    y_prob=y_prob_sbert_xgb,
    model_name="SBERT + XGBoost",
    save_name="sbert_xgboost",
)


print("\n" + "=" * 60)
print("Step 9: SBERT + GaussianNB")
print("=" * 60)

y_pred_sbert_gnb, y_prob_sbert_gnb = (
    get_gaussian_nb_predictions(
        X_train_sbert,
        y_train,
        X_test_sbert,
    )
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_sbert_gnb,
    y_prob=y_prob_sbert_gnb,
    model_name="SBERT + GaussianNB",
    save_name="sbert_gaussian_nb",
)

plot_roc_curve(
    y_true=y_test,
    model_outputs=[
        {"model": "Logistic Regression", "y_prob": y_prob_sbert_logistic},
        {"model": "XGBoost", "y_prob": y_prob_sbert_xgb},
        {"model": "GaussianNB", "y_prob": y_prob_sbert_gnb},
    ],
    title="ROC Curve - SBERT Models",
    save_name="sbert_roc_curve.png",
)


print("\n" + "=" * 60)
print("Step 10: Fine-tuned BERT")
print("=" * 60)

run_bert_pipeline()


print("\n" + "=" * 60)
print("Step 11: Collect Summary Results")
print("=" * 60)

summary = collect_results()
print_best_model(summary)

if not summary.empty:
    print("\nFinal Model Comparison:")
    print(summary)


print("\n" + "=" * 60)
print("Step 12: TF-IDF Anomaly Detection")
print("=" * 60)

run_anomaly_detection(X_train_tfidf, y_train, X_test_tfidf, y_test,)


print("\n" + "=" * 60)
print("Pipeline Finished")
print("=" * 60)
