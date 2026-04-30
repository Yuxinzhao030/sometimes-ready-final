from sklearn.linear_model import LogisticRegression


def get_logistic_regression_predictions(X_train, y_train, X_test, return_model=False):
    """Train Logistic Regression and return class predictions and fake-news probabilities."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    if return_model:
        return y_pred, y_prob, model

    return y_pred, y_prob
