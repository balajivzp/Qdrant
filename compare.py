import json
import re
import argparse
import joblib

from compare_raw_metrics import COLLECTIONS as RAW_COLLECTIONS
from ivf import search_ivf
from setup_hnsw import CONFIGS

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from embed import ARTIFACTS, MODEL_NAME, ROOT
from qdrant_setup import COLLECTIONS

def main():
    query_text = (ROOT / "queries.md").read_text(encoding="utf-8")
    queries = re.findall(r"^\d+\.\s+(.+)$", query_text, flags=re.MULTILINE)
    assert len(queries) == 5, "Expected five number queries"

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=(ARTIFACTS / "model_cache")
    )

    query_vector = model.encode(
        queries,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(ARTIFACTS / "query_vectors.npy", query_vector)

    (ARTIFACTS / "queries.json").write_text(
        json.dumps(queries,indent=2),
        encoding="utf-8"
    )

    client = QdrantClient(url="http://localhost:6333", timeout=60)
    results = []

    for query, vector in zip(queries, query_vector):
        print(f"\nQuery: {query}")

        for collection, distance in COLLECTIONS.items():
            response = client.query_points(
                collection_name=collection,
                query=vector.tolist(),
                search_params=models.SearchParams(exact=True),
                limit=5,
                with_payload=True,
            )

            matches = []
            print(f"\n  {distance.value}:")

            for rank, point in enumerate(response.points, start=1):
                matches.append({
                    "rank": rank,
                    "id": point.id,
                    "score": point.score,
                    "category": point.payload["category"],
                    "text": point.payload["text"],
                })
                print(
                    f"    {rank}. ID={point.id} "
                    f"score={point.score:.6f} "
                    f"category={point.payload['category']}"
                )

            results.append({
                            "query": query,
                            "collection": collection,
                            "metric": distance.value,
                            "exact": True,
                            "matches": matches,
                        })
    client.close()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output = results_dir / "distance_metrics.json"
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output}")

def run_demo(query_number):
    queries = json.loads(
        (ARTIFACTS / "queries.json").read_text(encoding="utf-8")
    )
    vectors = np.load(ARTIFACTS / "query_vectors.npy")
    raw_vectors = np.load(ARTIFACTS / "raw_query_vectors.npy")
    index = joblib.load(ARTIFACTS / "ivf_index.joblib")

    position = query_number - 1
    query = queries[position]
    vector = vectors[position]
    raw_vector = raw_vectors[position]

    print(f"\nQuery {query_number}: {query}")

    def show(label, ids, scores, ground_truth=None):
        print(f"\n{label}")
        if ground_truth is not None:
            overlap = len(set(ids) & ground_truth) / 5
            print(f"Overlap with normalized exact search: {overlap:.0%}")

        for rank, (point_id, score) in enumerate(zip(ids, scores), 1):
            print(f"  {rank}. ID={point_id}, score={score:.6f}")

    client = QdrantClient(url="http://localhost:6333", timeout=60)

    try:
        for collection in CONFIGS:
            info = client.get_collection(collection)
            assert info.status == models.CollectionStatus.GREEN
            assert info.indexed_vectors_count >= len(index["ids"])
            assert info.config.hnsw_config.full_scan_threshold == 10

        print("\n--- Metric comparison: unnormalized embeddings ---")

        for collection, distance in RAW_COLLECTIONS.items():
            points = client.query_points(
                collection_name=collection,
                query=raw_vector.tolist(),
                search_params=models.SearchParams(exact=True),
                limit=5,
                with_payload=False,
            ).points

            show(
                f"{distance.value}, exact",
                [point.id for point in points],
                [point.score for point in points],
            )

        print("\n--- Index comparison: normalized cosine embeddings ---")

        points = client.query_points(
            collection_name="capstone_cosine",
            query=vector.tolist(),
            search_params=models.SearchParams(exact=True),
            limit=5,
            with_payload=False,
        ).points

        ground_truth = {point.id for point in points}
        assert len(ground_truth) == 5

        show(
            "Exact cosine baseline",
            [point.id for point in points],
            [point.score for point in points],
            ground_truth,
        )

        for collection in CONFIGS:
            for ef in (16, 64, 128):
                points = client.query_points(
                    collection_name=collection,
                    query=vector.tolist(),
                    search_params=models.SearchParams(
                        exact=False,
                        hnsw_ef=ef,
                    ),
                    limit=5,
                    with_payload=False,
                ).points

                show(
                    f"{collection}, ef={ef}",
                    [point.id for point in points],
                    [point.score for point in points],
                    ground_truth,
                )

        for nprobe in (1, 8):
            result = search_ivf(index, vector, nprobe=nprobe, k=5)

            show(
                f"IVF, nprobe={nprobe}, "
                f"candidates={result['candidate_count']}",
                result["ids"],
                result["scores"],
                ground_truth,
            )
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-number",
        type=int,
        choices=range(1, 6),
        help="Run all methods for one saved query, numbered 1–5.",
    )
    args = parser.parse_args()

    if args.query_number is None:
        main()
    else:
        run_demo(args.query_number)
