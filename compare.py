import json
import re

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

if __name__ == "__main__":
    main()
