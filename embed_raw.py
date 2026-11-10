import json

import numpy as np
from sentence_transformers import SentenceTransformer

from embed import ARTIFACTS, MODEL_NAME


def main():
    documents = json.loads(
        (ARTIFACTS / "documents.json").read_text(encoding="utf-8")
    )
    queries = json.loads(
        (ARTIFACTS / "queries.json").read_text(encoding="utf-8")
    )

    base_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(ARTIFACTS / "model_cache"),
    )

    layers = list(base_model.children())
    print("Model layers:", [type(layer).__name__ for layer in layers])

    if type(layers[-1]).__name__ != "Normalize":
        raise ValueError("Expected a final Normalize layer.")

    raw_model = SentenceTransformer(
        modules=layers[:-1],
        device="cpu",
    )

    texts = [doc["text"] for doc in documents]

    for name, inputs in (
        ("raw_vectors", texts),
        ("raw_query_vectors", queries),
    ):
        vectors = raw_model.encode(
            inputs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        assert vectors.shape == (len(inputs), 384)
        assert np.isfinite(vectors).all()

        norms = np.linalg.norm(vectors, axis=1)
        assert np.all(norms > 0)

        np.save(ARTIFACTS / f"{name}.npy", vectors)
        print(
            f"{name}: shape={vectors.shape}, "
            f"length min={norms.min():.4f}, "
            f"mean={norms.mean():.4f}, "
            f"max={norms.max():.4f}"
        )


if __name__ == "__main__":
    main()