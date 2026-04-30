import numpy as np

from sklearn.linear_model import LogisticRegression
from tfidf_features import build_tfidf


def main():
    (
        X_train,
        y_train,
        _,
        _,
        vectorizer,
    ) = build_tfidf(return_vectorizer=True)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_n = 20

    top_fake_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_real_idx = np.argsort(coefficients)[:top_n]

    print("\n" + "=" * 60)
    print("Top Words Associated with Fake News")
    print("=" * 60)

    for word, coef in zip(
        feature_names[top_fake_idx],
        coefficients[top_fake_idx],
    ):
        print(f"{word:<30} {coef:.4f}")

    print("\n" + "=" * 60)
    print("Top Words Associated with Real News")
    print("=" * 60)

    for word, coef in zip(
        feature_names[top_real_idx],
        coefficients[top_real_idx],
    ):
        print(f"{word:<30} {coef:.4f}")


if __name__ == "__main__":
    main()