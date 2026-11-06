import json
from statistics import mean, median
from time import perf_counter

import joblib
import numpy as np

from ivf import search_ivf
from qdrant_setup import ARTIFACTS, ROOT


def main():
    index = joblib.load(ARTIFACTS / "ivf_index.joblib")
    queries = json.loads(
        (ARTIFACTS / "queries.json").read_text(encoding="utf-8")
    )
    vectors = np.load(ARTIFACTS / "query_vectors.npy")

    hnsw = json.loads(
        (ROOT / "results" / "hnsw.json").read_text(encoding="utf-8")
    )
    ground_truth = {
        row["query"]: set(row["ids"])
        for row in hnsw["results"]
        if row["method"] == "exact"
    }

    assert vectors.shape == (len(queries), 384)
    rows = []

    for query, vector in zip(queries, vectors):
        expected = ground_truth[query]

        # Verify every query against Qdrant's saved exact results.
        full = search_ivf(
            index, vector, nprobe=len(index["centroids"])
        )
        assert set(full["ids"]) == expected

        for nprobe in (1, 8):
            for _ in range(3):
                search_ivf(index, vector, nprobe)

            latencies = []
            for _ in range(10):
                start = perf_counter()
                result = search_ivf(index, vector, nprobe)
                latencies.append((perf_counter() - start) * 1000)

            assert len(result["ids"]) == 5
            overlap = len(set(result["ids"]) & expected) / 5

            rows.append({
                "query": query,
                "nprobe": nprobe,
                "ids": result["ids"],
                "scores": result["scores"],
                "candidate_count": result["candidate_count"],
                "overlap_at_5": overlap,
                "latency_ms": latencies,
                "median_latency_ms": median(latencies),
            })

            print(
                f"nprobe={nprobe}: overlap={overlap:.0%}, "
                f"candidates={result['candidate_count']}, "
                f"median={median(latencies):.3f} ms"
            )

    output = ROOT / "results" / "ivf.json"
    output.write_text(
        json.dumps({
            "n_clusters": len(index["centroids"]),
            "metric": "cosine",
            "warmups": 3,
            "repeats": 10,
            "results": rows,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nAverages across five queries:")
    for nprobe in (1, 8):
        group = [row for row in rows if row["nprobe"] == nprobe]
        print(
            f"nprobe={nprobe}: "
            f"overlap={mean(r['overlap_at_5'] for r in group):.0%}, "
            f"latency={mean(r['median_latency_ms'] for r in group):.3f} ms"
        )


if __name__ == "__main__":
    main()