"""Run the fixed Qdrant capstone stages for local n8n workflows."""

import argparse
import json
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic

from qdrant_capstone.config import ROOT
STAGES = {
    "sample": ("qdrant_capstone.data.prepare", None),
    "embed": ("qdrant_capstone.embeddings.normalized", None),
    "normalized-collections": ("qdrant_capstone.indexes.collections", None),
    "normalized-metrics": ("qdrant_capstone.search.compare", "results/distance_metrics.json"),
    "hnsw-setup": ("qdrant_capstone.indexes.hnsw", None),
    "hnsw-benchmark": ("qdrant_capstone.benchmarks.hnsw", "results/hnsw.json"),
    "ivf-setup": ("qdrant_capstone.indexes.ivf", None),
    "ivf-benchmark": ("qdrant_capstone.benchmarks.ivf", "results/ivf.json"),
    "raw-embed": ("qdrant_capstone.embeddings.raw", None),
    "raw-metrics": ("qdrant_capstone.search.raw_metrics", "results/distance_metrics_raw.json"),
}
STAGE_ORDER = tuple(STAGES)
IDLE_TIMEOUT = 600
STAGE_TIMEOUT = 3600


class Pipeline:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.run_id = None
        self.next_stage = 0
        self.last_stage_finished = 0

    def execute(self, stage, payload):
        if stage not in STAGES and stage != "demo":
            return HTTPStatus.NOT_FOUND, {"error": "Unknown stage."}
        run_id = payload.get("run_id")
        if stage != "demo" and (not isinstance(run_id, str) or not run_id):
            return HTTPStatus.BAD_REQUEST, {"error": "A run_id is required."}
        if stage == "demo":
            number = payload.get("query_number")
            if type(number) is not int or not 1 <= number <= 5:
                return HTTPStatus.BAD_REQUEST, {"error": "Choose an integer from 1 to 5."}

        if not self.lock.acquire(blocking=False):
            return HTTPStatus.CONFLICT, {"error": "A capstone stage is already running."}
        try:
            if monotonic() - self.last_stage_finished >= IDLE_TIMEOUT:
                self.run_id = None
            if self.run_id is None and stage == STAGE_ORDER[0]:
                self.run_id = run_id
                self.next_stage = 0

            if stage == "demo":
                allowed = self.run_id is None
            else:
                allowed = self.run_id == run_id and stage == STAGE_ORDER[self.next_stage]
            if not allowed:
                return HTTPStatus.CONFLICT, {"error": "Another run owns the pipeline or the stage is out of order."}

            args = ["qdrant_capstone.search.compare", "--query-number", str(number), "--json"] if stage == "demo" else [STAGES[stage][0]]
            try:
                process = subprocess.run(
                    [sys.executable, "-m", *args],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=STAGE_TIMEOUT,
                )
                if process.returncode:
                    self.run_id = None
                    return HTTPStatus.INTERNAL_SERVER_ERROR, {
                        "stage": stage,
                        "error": (process.stderr or process.stdout)[-6000:],
                    }
            except (OSError, subprocess.TimeoutExpired) as error:
                self.run_id = None
                return HTTPStatus.INTERNAL_SERVER_ERROR, {"stage": stage, "error": str(error)}

            if stage == "demo":
                try:
                    return HTTPStatus.OK, json.loads(process.stdout)
                except ValueError:
                    return HTTPStatus.INTERNAL_SERVER_ERROR, {
                        "stage": stage,
                        "error": "The demo did not return valid JSON.",
                    }

            self.next_stage += 1
            self.last_stage_finished = monotonic()
            complete = self.next_stage == len(STAGE_ORDER)
            if complete:
                self.run_id = None
                self.next_stage = 0
            result = {
                "run_id": run_id,
                "stage": stage,
                "complete": complete,
                "log": process.stdout.splitlines()[-15:],
            }
            if complete:
                result["result_files"] = [
                    str(self.root / output) for _, output in STAGES.values() if output
                ]
            elif STAGES[stage][1]:
                result["result_file"] = str(self.root / STAGES[stage][1])
            return HTTPStatus.OK, result
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(HTTPStatus.OK, {"status": "ok"})
        else:
            self.reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

    def do_POST(self):
        if self.headers.get_content_type() != "application/json":
            self.reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use application/json."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            self.reply(HTTPStatus.BAD_REQUEST, {"error": "Expected a JSON object."})
            return
        status, result = self.server.pipeline.execute(self.path.removeprefix("/"), payload)
        self.reply(status, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository checkout to run")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.pipeline = Pipeline(args.root.resolve())
        print(f"Qdrant capstone worker listening on http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
