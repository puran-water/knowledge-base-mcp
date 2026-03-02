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


def _retry_post(url, max_retries=5, backoff=2.0, **kwargs):
    """POST with exponential backoff for transient Modal errors (cold starts, 429s, 502s, 503s)."""
    session = _get_session()
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.post(url, **kwargs)
            if resp.status_code in (429, 502, 503) and attempt < max_retries - 1:
                # Respect Retry-After header if present, otherwise exponential backoff
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    delay = min(float(retry_after), 30.0)
                else:
                    delay = backoff * (2 ** attempt)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
        except requests.exceptions.ReadTimeout as exc:
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
    max_chars: int = 500,
) -> list:
    """Embed texts via TEI /embed endpoint (Modal-hosted or local).

    Args:
        tei_url: Base URL of TEI service (e.g., http://localhost:8080)
        texts: List of texts to embed
        batch_size: Texts per request
        timeout: HTTP timeout in seconds
        normalize: L2-normalize vectors
        max_chars: Truncate texts longer than this to avoid exceeding model's
            max token limit (512 tokens for Arctic Embed XS ≈ 1500 chars for
            technical text with ~3 chars/token)

    Returns:
        List of embedding vectors
    """
    # Truncate texts that exceed model's token limit
    truncated = 0
    safe_texts = []
    for t in texts:
        if len(t) > max_chars:
            safe_texts.append(t[:max_chars])
            truncated += 1
        else:
            safe_texts.append(t)
    if truncated:
        print(f"  TEI: truncated {truncated}/{len(texts)} texts to {max_chars} chars")

    def _embed_batch(batch_texts):
        """Embed a batch, splitting on 413 errors (token limit exceeded)."""
        resp = _retry_post(
            endpoint,
            json={"inputs": batch_texts},
            timeout=timeout,
        )
        return resp.json()

    def _embed_batch_safe(batch_texts, depth=0):
        """Embed with automatic bisection on 413 errors."""
        try:
            return _embed_batch(batch_texts)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 413:
                if len(batch_texts) == 1:
                    # Single text still too long — aggressively truncate
                    shorter = batch_texts[0][:max_chars // 2]
                    print(f"  TEI: single text 413, truncating to {len(shorter)} chars")
                    return _embed_batch([shorter])
                # Split batch in half and retry each
                mid = len(batch_texts) // 2
                print(f"  TEI: 413 on batch of {len(batch_texts)}, splitting at depth {depth}")
                left = _embed_batch_safe(batch_texts[:mid], depth + 1)
                right = _embed_batch_safe(batch_texts[mid:], depth + 1)
                return left + right
            raise

    embeddings = []
    endpoint = f"{tei_url.rstrip('/')}/embed"
    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i : i + batch_size]
        batch_vecs = _embed_batch_safe(batch)

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
