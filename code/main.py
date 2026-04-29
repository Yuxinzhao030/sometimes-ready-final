from data_cleaning import clean_data
from tfidf_features import build_tfidf
from logistic_regression import get_logistic_regression_predictions
from naive_bayes import get_naive_bayes_predictions
from evaluate import evaluate_predictions


clean_data()

X_train, y_train, X_test, y_test = build_tfidf()


# Logistic Regression
y_pred_lr, y_prob_lr = get_logistic_regression_predictions(
    X_train,
    y_train,
    X_test,
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_lr,
    y_prob=y_prob_lr,
    model_name="TF-IDF + Logistic Regression",
    save_name="tfidf_logistic",
)


# Naive Bayes
y_pred_nb, y_prob_nb = get_naive_bayes_predictions(
    X_train,
    y_train,
    X_test,
)

evaluate_predictions(
    y_test=y_test,
    y_pred=y_pred_nb,
    y_prob=y_prob_nb,
    model_name="TF-IDF + Naive Bayes",
    save_name="tfidf_naive_bayes",
)
