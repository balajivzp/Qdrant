import json
import time

import numpy as np
from qdrant_client import QdrantClient, models

from qdrant_setup import ARTIFACTS

CONFIGS = {
    "capstone_cosine": {"m": 16, "ef_construct": 100},
    "capstone_cosine_weak": {"m": 4, "ef_construct": 16},
}


def main():
    documents = json.loads(
        (ARTIFACTS / "documents.json").read_text(encoding="utf-8")
    )
    vectors = np.load(ARTIFACTS / "vectors.npy")
    ids = np.load(ARTIFACTS / "vector_ids.npy")

    assert vectors.shape == (len(documents), 384)
    assert np.array_equal(ids, [doc["id"] for doc in documents])

    payloads = [
        {"text": doc["text"], "category": doc["category"]}
        for doc in documents
    ]

    client = QdrantClient(url="http://localhost:6333", timeout=60)

    for name, settings in CONFIGS.items():
        hnsw = models.HnswConfigDiff(
            **settings,
            full_scan_threshold=10,
        )
        optimizer = models.OptimizersConfigDiff(indexing_threshold=1)

        if client.collection_exists(name):
            config = client.get_collection(name).config.params.vectors
            assert config.size == 384
            assert config.distance == models.Distance.COSINE

            client.update_collection(
                collection_name=name,
                hnsw_config=hnsw,
                optimizers_config=optimizer,
            )
        else:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=hnsw,
                optimizers_config=optimizer,
            )

        count = client.count(collection_name=name, exact=True).count
        if count == 0:
            client.upload_collection(
                collection_name=name,
                vectors=vectors,
                ids=ids.tolist(),
                payload=payloads,
                batch_size=64,
                wait=True,
            )
        elif count != len(documents):
            raise ValueError(f"Unexpected point count in {name}: {count}")

        deadline = time.monotonic() + 300

        while True:
            info = client.get_collection(name)
            indexed = info.indexed_vectors_count or 0
            print(
                f"{name}: indexed={indexed}/{len(documents)}, "
                f"status={info.status}"
            )

            if (
                indexed >= len(documents)
                and info.status == models.CollectionStatus.GREEN
            ):
                break

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Indexing did not finish for {name}")

            time.sleep(2)

        count = client.count(collection_name=name, exact=True).count
        assert count == len(documents)

    client.close()
    print("Both HNSW collections are ready.")


if __name__ == "__main__":
    main()
