
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from XGBoost import get_xgboost_predictions


# ======================
# LOAD DATA
# ======================
def load_data():
    train_path = "../train_test/train.csv"
    test_path = "../train_test/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["full_text"].astype(str).values
    y_train = train_df["label"].values

    X_test = test_df["full_text"].astype(str).values
    y_test = test_df["label"].values

    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    print(f"Train size: {len(X_train):,}, Fake ratio: {y_train.mean():.2%}")
    print(f"Test size: {len(X_test):,}, Fake ratio: {y_test.mean():.2%}")
    print("=" * 60)

    return X_train, y_train, X_test, y_test


# ======================
# EVALUATION
# ======================
def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ======================
# SBERT ENCODING 
# ======================
def encode_sbert(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings


# ======================
# MODELS
# ======================
def run_sbert_logistic(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("SBERT + LOGISTIC")
    print("=" * 60)

    X_train_emb = encode_sbert(X_train)
    X_test_emb = encode_sbert(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_emb, y_train)

    y_pred = model.predict(X_test_emb)
    y_prob = model.predict_proba(X_test_emb)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob)
    metrics["model"] = "SBERT + Logistic"

    print(metrics)

    return metrics, y_pred, y_prob


def run_sbert_xgboost(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("SBERT + XGBOOST")
    print("=" * 60)

    X_train_emb = encode_sbert(X_train)
    X_test_emb = encode_sbert(X_test)

    y_pred, y_prob = get_xgboost_predictions(
        X_train_emb,
        y_train,
        X_test_emb,
    )

    metrics = evaluate(y_test, y_pred, y_prob)
    metrics["model"] = "SBERT + XGBoost"

    print(metrics)

    return metrics, y_pred, y_prob


# ======================
# SAVE RESULTS
# ======================
def save_results(results):
    import json
    import pandas as pd

    json_path = "../results/sbert_results.json"
    csv_path = "../results/sbert_summary.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


# ======================
# PLOTS
# ======================
def plot_confusion_matrix(y_true, y_pred, model_name):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    safe = model_name.lower().replace(" ", "_")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title(model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = f"../results/{safe}_confusion.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[SAVED] {path}")


def plot_roc(y_true, outputs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.figure()

    for m in outputs:
        fpr, tpr, _ = roc_curve(y_true, m["y_prob"])
        auc = roc_auc_score(y_true, m["y_prob"])
        plt.plot(fpr, tpr, label=f"{m['model']} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("SBERT ROC")

    path = "../results/sbert_roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[SAVED] {path}")


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    log_m, log_pred, log_prob = run_sbert_logistic(
        X_train, y_train, X_test, y_test
    )

    xgb_m, xgb_pred, xgb_prob = run_sbert_xgboost(
        X_train, y_train, X_test, y_test
    )

    results = [log_m, xgb_m]

    save_results(results)

    plot_confusion_matrix(y_test, log_pred, "SBERT Logistic")
    plot_confusion_matrix(y_test, xgb_pred, "SBERT XGBoost")

    plot_roc(
        y_test,
        [
            {"model": "SBERT Logistic", "y_prob": log_prob},
            {"model": "SBERT XGBoost", "y_prob": xgb_prob},
        ],
    )
