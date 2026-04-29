import torch
from sentence_transformers import SentenceTransformer


def encode_sbert(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device,
    )

    return model.encode(
        list(texts),
        batch_size=64 if device == "cuda" else 32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )