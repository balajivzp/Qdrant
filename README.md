# Qdrant semantic search capstone

Embed 6,000 documents from 20 Newsgroups and compare distance metrics,
Qdrant exact search, two HNSW graphs, and a custom IVF-style index.

The project uses five fixed queries from [queries.md](queries.md).
Measured results and their interpretation are in [comparison.md](comparison.md).

## Requirements

- Python 3.10 or newer. The recorded environment uses Python 3.13.5.
- Docker with Docker Compose; on macOS, start Docker Desktop first.
- Internet access for the initial dataset and model downloads.
- Python package versions are recorded in [requirements.txt](requirements.txt).
- The experiments were run with Qdrant 1.20.0. The Compose file uses the
  unversioned official image; use `qdrant/qdrant:v1.20.0` in that file when
  reproducing the experiments with the same server version.

Run all commands below from the repository root.

## Install and start Qdrant

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
docker compose up -d
docker compose ps
curl --fail http://localhost:6333/
```

Open the [Qdrant dashboard](http://localhost:6333/dashboard).
Verification note: the dashboard endpoint returned HTTP 200 during setup;
the server API and collection searches were also verified successfully.

REST and dashboard traffic use `http://localhost:6333`; gRPC uses port 6334.
The local server uses HTTP, so an HTTPS URL produces a TLS connection error.

The Compose mount persists collections in `qdrant_storage/`. Stop the service
with `docker compose down`; restart it with `docker compose up -d`. The
mounted data remains on disk.

## Reproduce the experiments

Run these commands in order, allowing each command to finish successfully:

```bash
# Prepare the reproducible text sample and normalized document embeddings.
python data.py
python embed.py

# Create three metric collections, embed the five queries, and run exact search.
python qdrant_setup.py
python compare.py

# Build the two HNSW graphs and measure their overlap and latency.
python setup_hnsw.py
python benchmark_hnsw.py

# Build the custom IVF index and benchmark two search budgets.
python ivf.py
python benchmark_ivf.py

# Repeat the metric comparison with naturally varying vector magnitudes.
python embed_raw.py
python compare_raw_metrics.py
```

The first downloads and document embedding runs take longer than subsequent
searches. Embeddings, model cache, and the IVF index are saved in `artifacts/`.
This directory and `qdrant_storage/` are ignored by Git; a fresh checkout
regenerates them using the commands above.

`compare.py` without arguments creates the saved query embeddings required
by later scripts. If you change the dataset, model, or query order, regenerate
the dependent embeddings, collections, indexes, and benchmark results together.
`setup_hnsw.py` skips uploads to populated collections and checks their counts;
it assumes those collections contain the same dataset.

The scripts target the `capstone_*` collection names and leave unrelated
collections alone. Rerunning uploads with the same document IDs replaces those
points. Rebuilt approximate indexes and repeated timings can differ from the
committed results. Update the report if new measurements change its findings.

## Live review demo

After completing setup, run:

```bash
source .venv/bin/activate
docker compose up -d
python compare.py --query-number 2
```

Choose any query number from 1 through 5. The demo uses cached embeddings for
that saved query and executes live searches, printing five IDs and scores for
each of 12 result groups:

- Three exact metric searches using unnormalized embeddings.
- One exact cosine baseline using normalized embeddings.
- Normal and weak HNSW graphs, each at `hnsw_ef=16,64,128`.
- Custom IVF at `nprobe=1,8`, including the number of candidates examined.

The index methods also print overlap with the normalized exact baseline.
This mode does not regenerate embeddings or overwrite benchmark result files.

For query 2, the recorded raw metric top IDs are 5377 for cosine, 5673 for
Euclidean, and 5685 for dot product. IVF finds 40% of the exact top five at
`nprobe=1` and 100% at `nprobe=8` in the recorded run.

## Files and outputs

| File | Purpose or output |
|---|---|
| `data.py` | Load, clean, and sample documents; save `artifacts/documents.json` |
| `embed.py` | Save `vectors.npy` and the aligned `vector_ids.npy` in `artifacts/` |
| `queries.md` | Five numbered natural-language queries |
| `qdrant_setup.py` | Populate normalized Cosine, Euclidean, and Dot collections |
| `compare.py` | Generate cached queries and metric results, or run the live demo |
| `setup_hnsw.py` | Prepare normal and weak cosine HNSW collections |
| `benchmark_hnsw.py` | Save `results/hnsw.json` |
| `ivf.py` | Implement clustering and candidate search; save `artifacts/ivf_index.joblib` |
| `benchmark_ivf.py` | Save `results/ivf.json` using saved Qdrant exact IDs as ground truth |
| `embed_raw.py` | Save document and query embeddings with the final Normalize layer removed |
| `compare_raw_metrics.py` | Populate separate raw-vector collections and save their exact results |
| `results/distance_metrics.json` | Exact top-five results with normalized embeddings |
| `results/distance_metrics_raw.json` | Exact top-five results with unnormalized embeddings |
| `comparison.md` | Combined measurements and explanations |

## Experimental choices

- The dataset uses the training split, removes headers, footers, and quotes,
  drops empty documents, and uses shuffle seed 42 before selecting 6,000 texts.
- The embedding model is `sentence-transformers/all-MiniLM-L6-v2` with 384
  dimensions. All embedding scripts use the CPU.
- The original model includes a Normalize layer. Unit vectors give equivalent
  cosine, dot, and Euclidean rankings, apart from ties and numerical effects.
  The raw experiment removes that layer for both documents and queries; it
  supplies the same raw vectors to all three metric collections.
- Higher cosine and dot scores are better. Lower Euclidean distances are better.
  Scores from different metrics are not directly comparable.
- Index experiments consistently use normalized vectors and cosine similarity.
  Normal HNSW uses `m=16, ef_construct=100`; weak HNSW uses `m=4,
  ef_construct=16`.
- Both HNSW collections use `indexing_threshold=1` and
  `full_scan_threshold=10` to make this small dataset suitable for indexed
  search. The scripts wait for green status and check indexed-vector counts.
  Counts can include replaced entries and exceed the live point count.
- IVF uses 32 KMeans clusters, seed 42, and 10 initializations. It selects
  centroids by squared Euclidean distance, then scores only their members
  with cosine similarity. Searching all 32 clusters is checked against
  exhaustive search; the IVF benchmark checks all five queries against
  saved Qdrant exact results.
- Overlap@5 is the shared result-ID count divided by five. Benchmarks perform
  three warmups and ten timed searches per query and configuration. The report
  averages the five per-query median latencies.
- Index construction, embedding, and file loading are outside timed searches.
  Qdrant timings include local HTTP overhead; IVF timings are in-process.
  Five queries on 6,000 documents provide a small experimental comparison.

## Review preparation

Be able to explain each script and these observations in your own words:

1. Low `nprobe` can exclude the cluster containing a true nearest neighbor.
   Raising it searches more clusters and scores more candidates.
2. Dot product incorporates vector magnitude, whereas cosine normalizes it
   away. The raw-vector motorcycle query demonstrates different top results.
3. Increasing the weak graph's `m` toward 16 adds connectivity when the graph
   is rebuilt. Raising search-time `hnsw_ef` explores the existing graph more
   thoroughly but does not add links.
4. Qdrant's dense vector index is HNSW; this project's IVF implementation is
   separate Python code. A system might choose an IVF design when its cluster
   probing, batched scans, or combination with compression suits its workload.
   This small benchmark does not establish which design wins at production
   scale, and this implementation does not include compression.

## Troubleshooting

- Connection refused: start Docker Desktop and run `docker compose up -d`.
  Inspect `docker compose logs --tail=50 qdrant` if startup fails.
- Missing artifact files: follow the reproduction sequence in order.
- HNSW readiness assertions: run `python setup_hnsw.py` and let indexing finish.
- PyTorch's macOS redirect message is informational. Check for a subsequent
  traceback and confirm the script produces its expected output.
