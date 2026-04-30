import xgboost as xgb

def get_xgboost_predictions(X_train, y_train, X_test):
    params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob
