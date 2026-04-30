from sklearn.naive_bayes import MultinomialNB


def get_naive_bayes_predictions(X_train, y_train, X_test):
    """Train Multinomial Naive Bayes and return class predictions and fake-news probabilities."""
    model = MultinomialNB(alpha=1.0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob
