import os
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(
    data_path="./data/raw/WELFake_Dataset.csv",
    train_path="data/processed/train.csv",
    test_path="data/processed/test.csv",
    test_size=0.2,
    random_state=42,
):
    df = pd.read_csv(data_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["text"] = df["text"].str.replace(
        r"^[^.!?]*\(Reuters\)\s*[-–—]\s*",
        "",
        regex=True,
        case=False,
    )

    df["text"] = df["text"].str.replace(
        r"^\(?Reuters\)?\s*[-–—]\s*",
        "",
        regex=True,
        case=False,
    )

    patterns = [
        r"\[[^\]]{1,120}\]",
        r"\([^\)]{1,120}\)",
        r"http\S+",
        r"www\S+",
        r"\bfeatured image[s]?\b",
        r"\bgetty image[s]?\b",
        r"\bpic\.twitter\S*\b",
        r"\bimage[s]?\b",
        r"\bvideo[s]?\b",
        r"\bwatch\b",
        r"\bread\b",
        r"\bsource\b",
        r"\bwire\b",
        r"\bflickr\b",
        r"\bscreenshot[s]?\b",
        r"\bscreen capture\b",
        r"\bscreen\b",
        r"\bcapture\b",
        r"\btwitter\.?com\b",
        r"\btwitter\b",
        r"\bfollow twitter\b",
        r"\bfollow\b",
        r"\bshare\b",
        r"\breuters\b",
        r"\bbreitbart\b",
        r"\b21wire\b",
        r"\byoutube\b",
    ]

    for pattern in patterns:
        df["text"] = df["text"].str.replace(
            pattern,
            " ",
            regex=True,
            case=False,
        )

    df["text"] = (
        df["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["text_len"] = df["text"].str.len()

    df = df[df["text_len"] >= 200].copy()
    df = df.drop_duplicates(subset="text").copy()

    df = df[["text", "label"]].reset_index(drop=True)

    print("Clean data shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts(normalize=True))

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved train:", train_path)
    print("Saved test:", test_path)

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nTest label distribution:")
    print(test_df["label"].value_counts(normalize=True))

    return train_df, test_df


if __name__ == "__main__":
    clean_data()
