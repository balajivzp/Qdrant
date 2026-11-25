"""Shared paths and environment configuration for all entry points."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Explicit shell settings take precedence over the local, untracked .env file.
load_dotenv(ROOT / ".env", override=False)


def create_qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
    )
