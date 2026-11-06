import joblib
import numpy as np
from sklearn.cluster import KMeans

from embed import ARTIFACTS


def build_ivf_index(vectors, ids, n_clusters=32):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    assert np.all(norms > 0)
    vectors = vectors / norms

    clustering = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    labels = clustering.fit_predict(vectors)

    members = [
        np.flatnonzero(labels == cluster)
        for cluster in range(n_clusters)
    ]

    return {
        "vectors": vectors,
        "ids": ids,
        "centroids": clustering.cluster_centers_,
        "members": members,
    }


def search_ivf(index, query_vector, nprobe, k=5):
    centroids = index["centroids"]

    if not 1 <= nprobe <= len(centroids):
        raise ValueError("nprobe must be between 1 and the cluster count.")
    if k < 1:
        raise ValueError("k must be positive.")

    query = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        raise ValueError("Query vector must be nonzero.")
    query = query / norm

    centroid_distances = np.sum((centroids - query) ** 2, axis=1)
    selected = np.argsort(centroid_distances)[:nprobe]

    candidates = np.concatenate([
        index["members"][cluster] for cluster in selected
    ])

    scores = index["vectors"][candidates] @ query
    order = np.argsort(-scores)[:k]
    rows = candidates[order]

    return {
        "ids": index["ids"][rows].tolist(),
        "scores": scores[order].tolist(),
        "candidate_count": len(candidates),
    }


def main():
    vectors = np.load(ARTIFACTS / "vectors.npy")
    ids = np.load(ARTIFACTS / "vector_ids.npy")
    queries = np.load(ARTIFACTS / "query_vectors.npy")

    assert len(vectors) == len(ids)
    print("Building 32 clusters...")
    index = build_ivf_index(vectors, ids)
    joblib.dump(index, ARTIFACTS / "ivf_index.joblib")

    for nprobe in (1, 8, 32):
        result = search_ivf(index, queries[0], nprobe)
        print(
            f"nprobe={nprobe}: "
            f"candidates={result['candidate_count']}, "
            f"IDs={result['ids']}"
        )

    # Searching every cluster must match local exhaustive cosine search.
    query = queries[0] / np.linalg.norm(queries[0])
    exact_scores = index["vectors"] @ query
    exact_ids = ids[np.argsort(-exact_scores)[:5]].tolist()
    full_search = search_ivf(index, queries[0], nprobe=32)

    assert set(full_search["ids"]) == set(exact_ids)
    print("Full-probe correctness check passed.")


if __name__ == "__main__":
    main()