import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    documents = json.loads(
        (ARTIFACTS / "documents.json").read_text(encoding="utf-8")
    )
    texts = [doc["text"] for doc in documents]
    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=ARTIFACTS / "model_cache",
        device="cpu",
    )
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    vectors = vectors.astype(np.float32)
    assert vectors.shape == (len(documents), 384)
    assert np.isfinite(vectors).all()

    np.save((ARTIFACTS / "vectors.npy"),vectors)
    ids = np.array([doc["id"] for doc in documents], dtype=np.int64)
    np.save(ARTIFACTS / "vector_ids.npy",ids)
    norms = np.linalg.norm(vectors,axis=1
    )
    print(f"Saved vectors with shape: {vectors.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(
        f"Vector lengths: min={norms.min():.6f}, "
        f"mean={norms.mean():.6f}, max={norms.max():.6f}"
    )

if __name__ == "__main__":
    main()
