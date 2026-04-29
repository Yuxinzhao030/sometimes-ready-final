import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_path="data/processed/train.csv",
    test_path="data/processed/test.csv",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    stop_words="english",
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_text = train_df["text"].fillna("")
    X_test_text = test_df["text"].fillna("")

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("Train size:", len(X_train_text))
    print("Test size:", len(X_test_text))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    print("\nTF-IDF built successfully")
    print("Train shape:", X_train_tfidf.shape)
    print("Test shape:", X_test_tfidf.shape)

    return X_train_tfidf, y_train, X_test_tfidf, y_test


if __name__ == "__main__":
    build_tfidf()
