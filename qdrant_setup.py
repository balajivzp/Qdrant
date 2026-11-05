import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

COLLECTIONS = {
    "capstone_cosine" : models.Distance.COSINE,
    "capstone_euclid" : models.Distance.EUCLID,
    "capstone_dot" : models.Distance.DOT,
}

def main():
    documents = json.loads(
        (ARTIFACTS / "documents.json").read_text(encoding="utf-8")
    )

    vectors = np.load(ARTIFACTS / "vectors.npy")
    ids = np.load(ARTIFACTS / "vector_ids.npy")

    assert vectors.shape == (len(documents), 384)
    assert np.isfinite(vectors).all()
    assert np.array_equal(ids, [doc["id"] for doc in documents])

    payloads = [
        {"text" : doc["text"], "category" : doc["category"]}
        for doc in documents
    ]

    client = QdrantClient(url="http://localhost:6333",timeout=60)

    for name, distance in COLLECTIONS.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=distance
                )
            )
        config = client.get_collection(name).config.params.vectors
        if config.size != 384 or config.distance != distance:
            raise ValueError(f"Unexpected vector configuration in {name}")
        print(f"Uploading {len(documents)} documents to {name}...")

        client.upload_collection(
            collection_name=name,
            vectors=vectors,
            ids=ids.tolist(),
            payload=payloads,
            batch_size=64,
            wait=True,
        )

        count = client.count(collection_name=name,exact=True).count
        assert count == len(documents)
        print(f"{name}: {count} points")
    client.close()



if __name__ == "__main__":
    main()
