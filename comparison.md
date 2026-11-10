# Search comparison

## Experiment setup

- Dataset: 6,000 documents from 20 Newsgroups.
- Model: all-MiniLM-L6-v2, 384 dimensions.
- Five fixed queries; five results per query.
- Index experiments use normalized embeddings and cosine similarity.
- The separate raw metric experiment removes the model's final Normalize layer.
- Each latency uses ten timed searches after three warmups.
- Reported latency is the average of the five per-query medians.
- Overlap@5 is the number of shared IDs with exact top-five divided by five.

## Index results

| Method | Average overlap@5 | Latency (ms) |
|---|---:|---:|
| Exact brute-force | 100% | 1.842 |
| Normal HNSW, ef=16 | 100% | 1.793 |
| Normal HNSW, ef=64 | 100% | 1.793 |
| Normal HNSW, ef=128 | 100% | 1.841 |
| Weak HNSW, ef=16 | 92% | 1.636 |
| Weak HNSW, ef=64 | 96% | 1.958 |
| Weak HNSW, ef=128 | 96% | 1.707 |
| IVF, nprobe=1 | 88% | 0.045 |
| IVF, nprobe=8 | 100% | 0.174 |

Normal HNSW uses m=16 and ef_construct=100.
Weak HNSW uses m=4 and ef_construct=16.
Both use full_scan_threshold=10 and indexing_threshold=1.
IVF uses 32 KMeans clusters.

## Speed and accuracy

Normal HNSW matched exact search on all five queries at every tested
ef value. Increasing ef therefore produced no additional recall on
this query set. This does not guarantee perfect recall on other queries.

Weak HNSW achieved 92% average overlap at ef=16 and 96% at ef=64.
Increasing ef to 128 did not improve it further. A larger search budget
allows the algorithm to explore more of the existing graph. However,
it does not add the links omitted when the weaker graph was built.
Changing m or ef_construct would require rebuilding the index.

The custom IVF index improved from 88% overlap at nprobe=1 to 100%
at nprobe=8. It examined about 185 candidate vectors per query at the
lower setting and 1,624 at the higher setting. Searching additional
clusters reduced the chance of excluding a true nearest neighbor.
Scoring more candidates increased measured latency from approximately
0.045 ms to 0.174 ms.

These observations illustrate the accuracy-versus-work tradeoff:
approximate methods can miss relevant candidates when exploration
is limited, and additional exploration can recover them.

Latency comparisons require care. Qdrant measurements include local
HTTP communication and server processing, while IVF runs directly
inside Python. Consequently, IVF's lower measured latency does not
establish an inherent algorithmic advantage. HNSW latency also varied
rather than increasing consistently with ef. The dataset contains
only 6,000 documents and five queries, so these findings describe this
experiment rather than general performance guarantees.

## Distance metric findings

With the original unit-length embeddings, all three metrics returned
the same rankings. For unit vectors, dot product equals cosine
similarity, and squared Euclidean distance equals 2 - 2 × cosine.

I then removed the model's final Normalize layer and re-embedded the
same documents and queries. Document vector lengths varied, allowing
magnitude to affect the results.

For "What maintenance should I perform on a used motorcycle?", exact
search returned these top documents:

| Metric | Top document ID | Score |
|---|---:|---:|
| Cosine | 5377 | 0.501555 |
| Euclidean | 5673 | 5.230222 |
| Dot product | 5685 | 9.923678 |

Cosine compares direction independently of magnitude. Dot product
depends on direction and magnitude. Euclidean distance measures
straight-line separation and therefore also depends on vector length.

Within this raw-vector experiment, the documents, input vectors,
queries, and exact search method were identical across metrics.
Only the distance metric changed. This demonstrates that metric
choice and index choice are separate decisions.

Higher cosine and dot scores are better; lower Euclidean distances
are better. Scores from different metrics are not directly comparable.