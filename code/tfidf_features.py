from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["full_text"].fillna("")
X_test = test_df["full_text"].fillna("")

print("Train size:", len(X_train))
print("Test size:", len(X_test))


vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    stop_words="english",
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF built successfully")
print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)


joblib.dump(X_train_tfidf, FEATURE_DIR / "X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, FEATURE_DIR / "X_test_tfidf.pkl")

joblib.dump(vectorizer, FEATURE_DIR / "tfidf_vectorizer.pkl")

print("\nSaved:")
print("data/features/X_train_tfidf.pkl")
print("data/features/X_test_tfidf.pkl")
print("data/features/tfidf_vectorizer.pkl")
