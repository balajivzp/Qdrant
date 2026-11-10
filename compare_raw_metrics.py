import json

import numpy as np
from qdrant_client import QdrantClient, models

from qdrant_setup import ARTIFACTS, ROOT

COLLECTIONS = {
    "capstone_raw_cosine": models.Distance.COSINE,
    "capstone_raw_euclid": models.Distance.EUCLID,
    "capstone_raw_dot": models.Distance.DOT,
}


def main():
    documents = json.loads(
        (ARTIFACTS / "documents.json").read_text(encoding="utf-8")
    )
    queries = json.loads(
        (ARTIFACTS / "queries.json").read_text(encoding="utf-8")
    )
    vectors = np.load(ARTIFACTS / "raw_vectors.npy")
    query_vectors = np.load(ARTIFACTS / "raw_query_vectors.npy")
    ids = np.load(ARTIFACTS / "vector_ids.npy")

    assert vectors.shape == (len(documents), 384)
    assert query_vectors.shape == (len(queries), 384)
    assert np.array_equal(ids, [doc["id"] for doc in documents])

    payloads = [
        {"text": doc["text"], "category": doc["category"]}
        for doc in documents
    ]

    client = QdrantClient(url="http://localhost:6333", timeout=60)

    for name, distance in COLLECTIONS.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=distance,
                ),
            )

        config = client.get_collection(name).config.params.vectors
        assert config.size == 384 and config.distance == distance

        print(f"Uploading to {name}...")
        client.upload_collection(
            collection_name=name,
            vectors=vectors,
            ids=ids.tolist(),
            payload=payloads,
            batch_size=64,
            wait=True,
        )
        assert client.count(name, exact=True).count == len(documents)

    rows = []

    for query, vector in zip(queries, query_vectors):
        print(f"\nQuery: {query}")

        for name, distance in COLLECTIONS.items():
            points = client.query_points(
                collection_name=name,
                query=vector.tolist(),
                search_params=models.SearchParams(exact=True),
                limit=5,
                with_payload=True,
            ).points

            matches = [
                {
                    "rank": rank,
                    "id": point.id,
                    "score": point.score,
                    "category": point.payload["category"],
                    "text": point.payload["text"],
                }
                for rank, point in enumerate(points, start=1)
            ]
            assert len(matches) == 5

            rows.append({
                "query": query,
                "collection": name,
                "metric": distance.value,
                "exact": True,
                "matches": matches,
            })
            print(f"  {distance.value}: {[p.id for p in points]}")

    client.close()

    output = ROOT / "results" / "distance_metrics_raw.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps({
            "embedding_variant": "Final Normalize layer removed",
            "results": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()