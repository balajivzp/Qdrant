import json
from statistics import mean, median
from time import perf_counter

import numpy as np
from qdrant_client import QdrantClient, models

from embed import ARTIFACTS, ROOT
from setup_hnsw import CONFIGS


def measure(client, collection, vector, params):
    def search():
        return client.query_points(
            collection_name=collection,
            query=vector,
            search_params=params,
            limit=5,
            with_payload=False,
        ).points

    # Warm up before measuring.
    for _ in range(3):
        search()

    latencies = []
    for _ in range(10):
        start = perf_counter()
        points = search()
        latencies.append((perf_counter() - start) * 1000)

    return points, latencies


def main():
    queries = json.loads(
        (ARTIFACTS / "queries.json").read_text(encoding="utf-8")
    )
    vectors = np.load(ARTIFACTS / "query_vectors.npy")
    assert vectors.shape == (len(queries), 384)

    client = QdrantClient(url="http://localhost:6333", timeout=60)

    for collection in CONFIGS:
        info = client.get_collection(collection)
        assert info.status == models.CollectionStatus.GREEN
        assert info.indexed_vectors_count >= 6000
        assert client.count(collection, exact=True).count == 6000

    methods = [
        ("exact", "capstone_cosine", models.SearchParams(exact=True))
    ]
    for collection in CONFIGS:
        for ef in (16, 64, 128):
            methods.append((
                f"{collection}_ef{ef}",
                collection,
                models.SearchParams(exact=False, hnsw_ef=ef),
            ))

    rows = []

    for query, vector in zip(queries, vectors):
        ground_truth = None

        for method, collection, params in methods:
            points, latencies = measure(
                client, collection, vector.tolist(), params
            )
            ids = [point.id for point in points]
            assert len(ids) == 5

            if method == "exact":
                ground_truth = set(ids)

            overlap = len(set(ids) & ground_truth) / 5

            rows.append({
                "query": query,
                "method": method,
                "collection": collection,
                "search_params": params.model_dump(exclude_none=True),
                "ids": ids,
                "scores": [point.score for point in points],
                "overlap_at_5": overlap,
                "latency_ms": latencies,
                "median_latency_ms": median(latencies),
            })

            print(
                f"{method}: overlap={overlap:.0%}, "
                f"median={median(latencies):.3f} ms"
            )

    configurations = {
        name: client.get_collection(name).config.model_dump(mode="json")
        for name in CONFIGS
    }
    client.close()

    output = ROOT / "results" / "hnsw.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps({
            "configurations": configurations,
            "warmups": 3,
            "repeats": 10,
            "results": rows,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nAverages across five queries:")
    for method, _, _ in methods:
        group = [row for row in rows if row["method"] == method]
        print(
            f"{method}: "
            f"overlap={mean(r['overlap_at_5'] for r in group):.0%}, "
            f"latency={mean(r['median_latency_ms'] for r in group):.3f} ms"
        )


if __name__ == "__main__":
    main()