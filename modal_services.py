"""HTTP client functions for Modal-hosted Docling extraction and TEI embedding.

Modal app definitions live in ~/servers/rag-backend/modal_deploy.py.
This module contains only the HTTP client code used by ingest.py and server.py.

Usage:
    # Deploy services (from rag-backend workspace)
    cd ~/servers/rag-backend && ./deploy.sh

    # Use from ingest.py
    python ingest.py --root /path/to/docs \
        --extract-provider modal --modal-extract-url <url> \
        --embed-provider tei --tei-embed-url <url>
"""

import math
import time

import requests
from requests.adapters import HTTPAdapter
from pathlib import Path

# Connection pooling for reuse across multiple calls
_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
        _SESSION.mount("http://", adapter)
        _SESSION.mount("https://", adapter)
    return _SESSION


def _retry_post(url, max_retries=3, backoff=2.0, **kwargs):
    """POST with exponential backoff for transient Modal errors (cold starts, 503s)."""
    session = _get_session()
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.post(url, **kwargs)
            if resp.status_code == 503 and attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
    # Should not reach here, but just in case
    if last_exc:
        raise last_exc


def embed_texts_tei(
    tei_url: str,
    texts: list,
    batch_size: int = 128,
    timeout: int = 120,
    normalize: bool = True,
) -> list:
    """Embed texts via TEI /embed endpoint (Modal-hosted or local).

    Args:
        tei_url: Base URL of TEI service (e.g., http://localhost:8080)
        texts: List of texts to embed
        batch_size: Texts per request
        timeout: HTTP timeout in seconds
        normalize: L2-normalize vectors

    Returns:
        List of embedding vectors
    """
    embeddings = []
    endpoint = f"{tei_url.rstrip('/')}/embed"
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _retry_post(
            endpoint,
            json={"inputs": batch},
            timeout=timeout,
        )
        batch_vecs = resp.json()

        if normalize:
            normalized = []
            for vec in batch_vecs:
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                normalized.append([v / norm for v in vec])
            batch_vecs = normalized

        embeddings.extend(batch_vecs)

    return embeddings


def extract_via_modal(file_path: str, modal_url: str) -> dict:
    """Upload file to Modal Docling endpoint, return doc_dict for local block processing.

    Args:
        file_path: Local path to the document
        modal_url: URL of the Modal Docling endpoint

    Returns:
        Dict with 'doc_dict', 'filename', 'size_bytes', 'needs_ocr', 'status'
    """
    p = Path(file_path)
    file_bytes = p.read_bytes()

    endpoint = modal_url.rstrip("/")
    if not endpoint.endswith("/extract"):
        endpoint = f"{endpoint}/extract"

    resp = _retry_post(
        endpoint,
        files={"file": (p.name, file_bytes)},
        timeout=600,  # 10 min for large PDFs
    )
    result = resp.json()

    if result.get("status") == "error":
        raise RuntimeError(f"Modal extraction failed: {result.get('error')}")

    return result
