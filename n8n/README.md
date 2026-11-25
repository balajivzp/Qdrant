# Qdrant capstone in n8n

The workflows use native n8n trigger, Edit Fields, and HTTP Request nodes. The
loopback Python worker runs the repository's existing scripts in the project
virtual environment. This preserves the dataset, model, collection names,
benchmark timing boundaries, and report formats.

## Start the Python worker

From the repository root, with the [project prerequisites](../README.md#requirements)
installed and Qdrant running on port 6333:

```bash
.venv/bin/python -m qdrant_capstone.integrations.n8n_worker
```

Keep the worker running while using the workflows. It listens on
`127.0.0.1:8765`, so the supplied workflows expect n8n to run directly on the
same host. Qdrant and the worker can be checked at
`http://127.0.0.1:6333/` and `http://127.0.0.1:8765/health`. First runs need
internet access for any model and dataset files missing from the local caches.

## Import and run

Import both workflow JSON files via **Import from File** in the n8n editor.
For a local npm installation you can also run
`n8n import:workflow --separate --input=n8n` from the repository root.

1. Open **Qdrant Capstone | Reproduce Benchmarks** and click **Execute Workflow**.
   The ten stages run in the repository's documented order. The final node
   returns paths to the four files under `results/`. As with the original
   commands, a rerun refreshes the artifacts and reports and upserts points into
   the existing `capstone_*` collections.
2. Open **Qdrant Capstone | Live Query Demo**. In **Choose Query Number (1–5)**,
   set the saved query number and run the workflow. The final node returns
   ranked IDs and scores for all 12 search configurations, overlap@5 where
   applicable, and candidate counts for IVF. The default query is 2. An existing
   set of artifacts and indexes from the original scripts also works.

Start a benchmark from its manual trigger. The worker permits one benchmark
run at a time because runs share artifact files and Qdrant collections. A failed
Python stage releases the run. If an n8n execution is canceled between stages,
the worker releases the run after ten minutes of inactivity. The worker waits
for an in-progress Python stage to finish before accepting another stage.
