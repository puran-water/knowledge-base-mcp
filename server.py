import json
import os
import logging
import asyncio
import time
import re
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import requests
import httpx
from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
import sqlite3

from document_store import DocumentStore, get_subjects_from_context
from models import (
    IngestExtractInput,
    IngestValidateExtractionInput,
    IngestChunkInput,
    IngestGenerateMetadataInput,
    IngestAssessQualityInput,
    IngestEnhanceInput,
    IngestUpsertInput,
    IngestUpsertBatchInput,
    IngestGenerateSummaryInput,
    IngestCorpusUpsertInput,
    KBSearchInput,
    KBSparseInput,
    KBSparseSpladeInput,
    KBDenseInput,
    KBHybridInput,
    KBRerankInput,
    KBColbertInput,
    KBBatchInput,
    KBQualityInput,
    KBHintInput,
    KBTableInput,
    KBOpenInput,
    KBNeighborsInput,
    KBSummaryInput,
    KBOutlineInput,
    KBEntitiesInput,
    KBLinkoutsInput,
    KBGraphInput,
    KBPromoteInput,
    KBDemoteInput,
)
from graph_builder import (
    entity_linkouts as graph_entity_linkouts,
    list_entities as graph_list_entities,
    neighbors as graph_neighbors,
)
from summary_index import query_summaries, upsert_summary_entry
from lexical_index import ALIASES, fetch_texts_by_chunk_ids
from sparse_expansion import SparseExpander
from ingest_blocks import (
    extract_document_blocks,
    chunk_blocks,
    _serialize_blocks,
    _deserialize_blocks,
)
from metadata_schema import generate_metadata
from ingest_core import upsert_document_artifacts
from datetime import datetime


# Response profile enum for token-efficient responses
class ResponseProfile(str, Enum):
    """Response payload size profiles for retrieval tools.

    SLIM: Minimal fields (chunk_id, text, doc_id, path, section_path, pages, score)
          Default profile - optimized for token efficiency (90% reduction)

    FULL: All structural metadata including tables, bboxes, element_ids
          Use when table reconstruction or figure citations needed

    DIAGNOSTIC: Full metadata + provenance + score breakdowns
                Use for quality audits and debugging extraction issues
    """
    SLIM = "slim"
    FULL = "full"
    DIAGNOSTIC = "diagnostic"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_flag(name: str, *, fallback: Optional[str] = None, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None and fallback:
        raw = os.getenv(fallback)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}

# ---- Env & config -----------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "snowflake-arctic-embed:xs")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", os.getenv("OLLAMA_MODEL_GENERATE", "llama3:8b"))
TEI_RERANK_URL = os.getenv("TEI_RERANK_URL", "http://localhost:8087")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_METRIC = os.getenv("QDRANT_METRIC", "cosine")
COLBERT_URL = os.getenv("COLBERT_URL")
COLBERT_TIMEOUT = int(os.getenv("COLBERT_TIMEOUT", "60"))
FTS_DB_PATH = os.getenv("FTS_DB_PATH", "data/fts.db")
GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "data/graph.db")
SUMMARY_DB_PATH = os.getenv("SUMMARY_DB_PATH", "data/summary.db")
SPARSE_EXPANDER = SparseExpander(os.getenv("SPARSE_EXPANDER", "none"))
SPARSE_QUERY_TOP_K = int(os.getenv("SPARSE_QUERY_TOP_K", "48"))
HAVE_SPARSE_SPLADE = SPARSE_EXPANDER.enabled
HAVE_COLBERT = bool(COLBERT_URL)
INGEST_EMBED_BATCH = int(os.getenv("INGEST_EMBED_BATCH", "32"))
INGEST_EMBED_TIMEOUT = int(os.getenv("INGEST_EMBED_TIMEOUT", "120"))
INGEST_EMBED_PARALLEL = int(os.getenv("INGEST_EMBED_PARALLEL", "1"))
INGEST_EMBED_THREADS = int(os.getenv("INGEST_EMBED_THREADS", "8"))
INGEST_EMBED_KEEPALIVE = os.getenv("INGEST_EMBED_KEEPALIVE", "1h")
INGEST_EMBED_FORCE_PER_ITEM = _env_flag("INGEST_EMBED_FORCE_PER_ITEM", default=False)
INGEST_EMBED_ROBUST = _env_flag("INGEST_EMBED_ROBUST", default=False)

# JSON like: {"kb":{"collection":"snowflake_kb","title":"Company KB"}}
# Backcompat: allow STELLA_SCOPES if NOMIC_KB_SCOPES not set
SCOPES_ENV = os.getenv("NOMIC_KB_SCOPES") or os.getenv("STELLA_SCOPES") or '{"kb":{"collection":"snowflake_kb","title":"Company KB"}}'
SCOPES: Dict[str, Dict[str, Any]] = json.loads(SCOPES_ENV)

# ---- Utilities --------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("kb-mcp")

qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
mcp = FastMCP(name="knowledge-base", version="1.0.0", instructions="Vector search with Qdrant + Ollama embeddings and TEI reranker")

# Collection-specific document stores (replaces global doc_store)
_doc_stores: Dict[str, DocumentStore] = {}

def _get_doc_store(collection_name: str) -> DocumentStore:
    """Get or create a DocumentStore for the given collection."""
    if not collection_name:
        # If no collection specified, use default from first SCOPE
        collection_name = next(iter(SCOPES.values()))["collection"]

    if collection_name not in _doc_stores:
        fts_db_path = _get_fts_db_path(collection_name)
        _doc_stores[collection_name] = DocumentStore(fts_db_path)

    return _doc_stores[collection_name]

ARTIFACT_DIR = Path(os.getenv("INGEST_ARTIFACT_DIR", "data/ingest_artifacts"))
PLAN_DIR = Path(os.getenv("INGEST_PLAN_DIR", "data/ingest_plans"))
INGEST_MODEL_VERSION = os.getenv("INGEST_MODEL_VERSION", "structured_ingest_v1")
INGEST_PROMPT_SHA = os.getenv("INGEST_PROMPT_SHA", "sha_deterministic_chunking_v1")
MAX_METADATA_BYTES = int(os.getenv("MAX_METADATA_BYTES", "8192"))
MAX_METADATA_CALLS_PER_DOC = int(os.getenv("MAX_METADATA_CALLS_PER_DOC", "2"))
MAX_CANARY_QUERIES = int(os.getenv("MAX_CANARY_QUERIES", "6"))
CANARY_TIMEOUT = int(os.getenv("CANARY_TIMEOUT", "30"))

SESSION_PRIORS: Dict[str, Dict[str, float]] = {}


def _lookup_chunk_by_element(element_id: str, fts_db_path: Optional[str] = None) -> Optional[str]:
    if not element_id:
        return None
    db_path = fts_db_path or FTS_DB_PATH
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id FROM fts_chunks WHERE element_ids LIKE ? LIMIT 1",
            (f'%"{element_id}"%',),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Rerank constraints to avoid payload-too-large
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "700"))
RERANK_MAX_ITEMS = int(os.getenv("RERANK_MAX_ITEMS", "16"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
# Neighbor packaging and scoring controls
NEIGHBOR_CHUNKS = int(os.getenv("NEIGHBOR_CHUNKS", "1"))
ANSWERABILITY_THRESHOLD = float(os.getenv("ANSWERABILITY_THRESHOLD", "0.0"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "0"))
DECAY_STRENGTH = float(os.getenv("DECAY_STRENGTH", "0.0"))
MIX_W_BM25 = _env_float("MIX_W_BM25", 0.2)
MIX_W_DENSE = _env_float("MIX_W_DENSE", 0.3)
MIX_W_RERANK = _env_float("MIX_W_RERANK", 0.5)
THIN_PAYLOAD_ENABLED = _env_flag("THIN_PAYLOAD", fallback="THIN_VECTOR_PAYLOAD")
AUR_AUDIT_PATH = Path(os.getenv("AUDIT_LOG_PATH", "logs/audit.log"))
ALLOWED_ROOTS = [Path(p).resolve() for p in os.getenv("ALLOWED_DOC_ROOTS", "").split(os.pathsep) if p.strip()]


def l2norm(vec: List[float]) -> List[float]:
    import math
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


QUALITY_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")


def _tokenize_quality(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in QUALITY_TOKEN_RE.findall(text)]


def _coverage_ratio(tokens: List[str], text: Optional[str]) -> float:
    if not tokens:
        return 0.0
    content = (text or "").lower()
    if not content:
        return 0.0
    hits = sum(1 for tok in tokens if tok in content)
    return hits / max(len(tokens), 1)


def _summarize(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "avg": round(sum(values) / len(values), 4),
        "count": len(values),
    }


async def embed_query(text: str, normalize: bool = True) -> List[float]:
    """Generate embedding vector for query text using Ollama.

    Uses native async httpx for non-blocking HTTP calls.

    Args:
        text: Query text to embed
        normalize: Whether to L2-normalize the vector (default: True)

    Returns:
        List[float]: Embedding vector
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": OLLAMA_MODEL, "input": [text]},
            )
            if r.status_code == 404:
                # Fallback to older Ollama API
                r2 = await client.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": OLLAMA_MODEL, "prompt": text},
                )
                r2.raise_for_status()
                v = r2.json().get("embedding")
            else:
                r.raise_for_status()
                v = r.json().get("embeddings", [[]])[0]
        return l2norm(v) if normalize else v
    except Exception:
        logger.exception("embed_query failed")
        raise


# ---- Local lexical helpers --------------------------------------------------
def _fts_search(query: str, limit: int, fts_db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    db_path = fts_db_path or FTS_DB_PATH
    if not os.path.exists(db_path):
        return []
    try:
        from lexical_index import search as fts_search
        try:
            return fts_search(db_path, query, limit)
        except Exception as exc:
            if "syntax error" in str(exc).lower():
                safe = re.sub(r"[\^`~!@#$%&*()+={}[\]|\\:;'<>,.?/]", " ", query).strip()
                if safe:
                    return fts_search(db_path, safe, limit)
            raise
    except Exception as e:
        logger.warning("FTS search failed: %s", e)
        return []


def _sparse_terms_search(query_terms: Dict[str, float], limit: int, fts_db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    db_path = fts_db_path or FTS_DB_PATH
    if not os.path.exists(db_path) or not query_terms:
        return []
    try:
        from lexical_index import sparse_search
        return sparse_search(db_path, query_terms, limit)
    except Exception as exc:
        logger.warning("Sparse-term search failed: %s", exc)
        return []


def _encode_sparse_query(text: str) -> Dict[str, float]:
    if not HAVE_SPARSE_SPLADE or not text:
        return {}
    return SPARSE_EXPANDER.encode_dict(text, top_k=SPARSE_QUERY_TOP_K)


async def _run_colbert(
    collection: str,
    query: str,
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not HAVE_COLBERT:
        return [{"error": "colbert_service_unavailable"}]
    fts_db_path = _get_fts_db_path(collection)
    limit = max(return_k, min(retrieve_k, 64))
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=float(COLBERT_TIMEOUT)) as client:
            resp = await client.post(
                f"{COLBERT_URL.rstrip('/')}/query",
                json={
                    "collection": collection,
                    "query": query,
                    "k": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict):
            hits = data.get("results") or data.get("hits") or []
        else:
            hits = data
        if not isinstance(hits, list):
            raise ValueError("Unexpected ColBERT response format")
    except Exception as exc:
        timings["colbert_ms"] = timings.get("colbert_ms", 0.0) + (time.perf_counter() - start) * 1000.0
        return [{"error": "colbert_query_failed", "detail": str(exc)}]
    timings["colbert_ms"] = timings.get("colbert_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    scored: List[Dict[str, Any]] = []
    chunk_ids: List[str] = []
    for entry in hits:
        if not isinstance(entry, dict):
            continue
        chunk_id = entry.get("chunk_id")
        doc_id = entry.get("doc_id")
        if not chunk_id:
            continue
        chunk_ids.append(str(chunk_id))
        scored.append(
            {
                "chunk_id": str(chunk_id),
                "doc_id": doc_id,
                "colbert_score": float(entry.get("score", 0.0) or 0.0),
            }
        )
    if not scored:
        return []
    payload_map = fetch_texts_by_chunk_ids(fts_db_path, chunk_ids)
    col_scores = [row["colbert_score"] for row in scored]
    stats = (min(col_scores), max(col_scores)) if col_scores else None
    rows: List[Dict[str, Any]] = []
    for info in scored[:return_k]:
        chunk_id = info["chunk_id"]
        base = payload_map.get(chunk_id, {})
        col_score = info["colbert_score"]
        decay = _decay_factor(base.get("mtime"))
        score = _combined_score(
            bm25=None,
            dense=col_score,
            rerank=None,
            bm_stats=None,
            dense_stats=stats,
            rerank_stats=None,
            decay=decay,
        )
        prior_mult = _prior_multiplier(base.get("doc_id"), subjects, boosts)
        score *= prior_mult
        row = {
            "score": score,
            "final_score": score,
            "dense_score": col_score,
            "colbert_score": col_score,
            "decay_factor": decay,
            "id": chunk_id,
            "chunk_id": chunk_id,
            "doc_id": base.get("doc_id"),
            "path": base.get("path"),
            "chunk_start": base.get("chunk_start"),
            "chunk_end": base.get("chunk_end"),
            "text": base.get("text"),
            "section_path": base.get("section_path"),
            "element_ids": base.get("element_ids"),
            "bboxes": base.get("bboxes"),
            "types": base.get("types"),
            "source_tools": base.get("source_tools"),
            "table_headers": base.get("table_headers"),
            "table_units": base.get("table_units"),
            "chunk_profile": base.get("chunk_profile"),
            "plan_hash": base.get("plan_hash"),
            "model_version": base.get("model_version"),
            "prompt_sha": base.get("prompt_sha"),
            "doc_metadata": base.get("doc_metadata"),
            "scores": _score_breakdown(
                bm25=None,
                dense=col_score,
                rrf=None,
                rerank=None,
                decay=decay,
                final=score,
            ),
        }
        _set_page_fields(row, base)
        _ensure_score_bucket(row)["prior"] = prior_mult
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    _annotate_rows(rows, query)
    return rows


def _fts_neighbors(doc_id: str, chunk_start: int, n: int, fts_db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    db_path = fts_db_path or FTS_DB_PATH
    if n <= 0 or not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM fts_chunks WHERE doc_id = ? AND chunk_start < ? ORDER BY chunk_start DESC LIMIT ?",
            (doc_id, int(chunk_start), int(n)),
        )
        prev_rows = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT * FROM fts_chunks WHERE doc_id = ? AND chunk_start >= ? ORDER BY chunk_start ASC LIMIT ?",
            (doc_id, int(chunk_start), int(n + 1)),
        )
        next_rows = [dict(r) for r in cur.fetchall()]
        rows = list(reversed(prev_rows)) + next_rows
        # Drop dup
        seen, out = set(), []
        for r in rows:
            key = (r.get("chunk_id"), r.get("chunk_start"))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
    except Exception as e:
        logger.debug("fts_neighbors failed: %s", e)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _rrf_fuse(lists: List[List[Dict[str, Any]]], id_getter) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for li in lists:
        for rank, x in enumerate(li, start=1):
            xid = id_getter(x)
            if xid is None:
                continue
            if xid not in items:
                items[xid] = dict(x)
            else:
                for k, v in x.items():
                    if k not in items[xid]:
                        items[xid][k] = v
            scores[xid] = scores.get(xid, 0.0) + 1.0 / (HYBRID_RRF_K + rank)
    fused = list(items.values())
    for it in fused:
        try:
            key = it.get("chunk_id")
        except Exception:
            key = None
        it["_rrf_score"] = scores.get(str(key), 0.0)
    fused.sort(key=lambda d: d.get("_rrf_score", 0.0), reverse=True)
    return fused
def _decay_factor(mtime: Optional[int]) -> float:
    if not mtime or DECAY_HALF_LIFE_DAYS <= 0:
        return 1.0
    age_days = max(0.0, (time.time() - float(mtime)) / 86400.0)
    try:
        import math
        factor = math.pow(2.0, -age_days / DECAY_HALF_LIFE_DAYS)
    except Exception:
        factor = 1.0
    if DECAY_STRENGTH <= 0:
        return 1.0
    if DECAY_STRENGTH >= 1:
        return factor
    return (1.0 - DECAY_STRENGTH) + DECAY_STRENGTH * factor


def _parse_page_numbers(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value if isinstance(v, (int, float))]
    if isinstance(value, str):
        digits = re.findall(r"\d+", value)
        return [int(d) for d in digits]
    if isinstance(value, (int, float)):
        return [int(value)]
    return []


def _canonical_page_numbers(source: Any) -> List[int]:
    if isinstance(source, dict):
        candidate = source.get("page_numbers")
        if candidate in (None, "", []):
            candidate = source.get("pages")
    else:
        candidate = source
    return _parse_page_numbers(candidate)


def _set_page_fields(row: Dict[str, Any], source: Any) -> None:
    pages = _canonical_page_numbers(source)
    row["page_numbers"] = pages
    row["pages"] = list(pages)


async def _run_canaries(doc_id: str, chunk_payload: Dict[str, Any]) -> Dict[str, Any]:
    collection = None
    plan = _load_plan(doc_id)
    if isinstance(plan, dict):
        collection = plan.get("collection")
    if not collection:
        for cfg in SCOPES.values():
            coll = cfg.get("collection")
            if coll:
                collection = coll
                break
    if not collection:
        return {"available": False, "reason": "unknown_collection"}

    config_path = _canary_config_path(collection)
    if not config_path.exists():
        return {"available": False, "reason": "no_config"}

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("failed to load canary config: %s", exc)
        return {"available": False, "reason": "config_error"}

    collection_config = config.get(collection) or {}
    queries = collection_config.get("queries") or []
    if not queries:
        return {"available": False, "reason": "no_queries"}

    results = []
    limit = min(MAX_CANARY_QUERIES, len(queries))
    subjects = [f"canary:{doc_id}"]
    for entry in queries[:limit]:
        prompt = entry.get("prompt") if isinstance(entry, dict) else entry
        if not prompt:
            continue
        mode = entry.get("mode") if isinstance(entry, dict) else "hybrid"
        top_k = entry.get("top_k", 5) if isinstance(entry, dict) else 5
        retrieve_k = entry.get("retrieve_k", 12) if isinstance(entry, dict) else 12
        return_k = entry.get("return_k", 5) if isinstance(entry, dict) else 5
        try:
            rows = await search(
                Context(subjects=subjects),
                prompt,
                mode=mode,
                top_k=top_k,
                retrieve_k=retrieve_k,
                return_k=return_k,
            )
        except Exception as exc:
            results.append({"query": prompt, "error": str(exc)})
            continue
        hits = [row for row in rows if isinstance(row, dict) and row.get("doc_id") == doc_id and not row.get("abstain")]
        results.append({
            "query": prompt,
            "mode": mode,
            "hits": len(hits),
            "best_score": _best_score(hits),
        })

    return {
        "available": True,
        "queries": results,
        "limit": limit,
    }


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


WHY_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
WHY_MAX_MATCHES = 6


def _score_breakdown(
    *,
    bm25: Optional[float],
    dense: Optional[float],
    rrf: Optional[float],
    rerank: Optional[float],
    decay: Optional[float],
    final: Optional[float],
) -> Dict[str, Optional[float]]:
    return {
        "bm25": bm25,
        "dense": dense,
        "rrf": rrf,
        "rerank": rerank,
        "decay": decay,
        "final": final,
    }


def _build_why_annotations(query: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not query or not isinstance(row, dict):
        return []
    annotations: List[Dict[str, Any]] = []
    try:
        terms = {tok.lower() for tok in WHY_TOKEN_RE.findall(query)}
    except Exception:
        terms = set()
    text = ""
    try:
        text = str(row.get("text") or "").lower()
    except Exception:
        text = ""
    if terms and text:
        matches = sorted({tok for tok in terms if tok and tok in text})[:WHY_MAX_MATCHES]
        if matches:
            annotations.append({"matched_terms": matches})
    section_path = row.get("section_path")
    if isinstance(section_path, list) and section_path and terms:
        hits = [seg for seg in section_path if isinstance(seg, str) and any(tok in seg.lower() for tok in terms)]
        if hits:
            annotations.append({"section_path_hits": hits})
    types = row.get("types") or []
    if isinstance(types, list) and any(isinstance(t, str) and "table" in t.lower() for t in types):
        element_ids = row.get("element_ids")
        note: Dict[str, Any] = {"table": True}
        if isinstance(element_ids, list) and element_ids:
            note["element_id"] = element_ids[0]
        annotations.append(note)
    return annotations


def _annotate_rows(rows: List[Dict[str, Any]], query: str) -> None:
    if not query:
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("error"):
            continue
        row["why"] = _build_why_annotations(query, row)


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _doc_id_for_path(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, _file_uri(path)))


def _is_allowed_path(path: Optional[str]) -> bool:
    if not ALLOWED_ROOTS:
        return True
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except Exception:
        return False
    for root in ALLOWED_ROOTS:
        try:
            if resolved == root or resolved.is_relative_to(root):
                return True
        except AttributeError:
            try:
                resolved.relative_to(root)
                return True
            except Exception:
                continue
        except Exception:
            continue
    return False


def _session_key(subjects: List[str]) -> str:
    if not subjects:
        return "*"
    return "|".join(sorted(subjects))


def _update_session_prior(subjects: List[str], doc_id: str, delta: float) -> float:
    key = _session_key(subjects)
    priors = SESSION_PRIORS.setdefault(key, {})
    doc_id = str(doc_id)
    current = priors.get(doc_id, 0.0) + float(delta)
    current = max(-0.9, min(4.0, current))
    if abs(current) < 1e-6:
        priors.pop(doc_id, None)
    else:
        priors[doc_id] = current
    return max(0.1, 1.0 + current)


def _prior_multiplier(doc_id: Optional[str], subjects: List[str], boosts: Optional[Dict[str, float]] = None) -> float:
    multiplier = 1.0
    if doc_id:
        doc_id = str(doc_id)
        key = _session_key(subjects)
        prior_delta = SESSION_PRIORS.get(key, {}).get(doc_id, 0.0)
        multiplier *= max(0.1, 1.0 + prior_delta)
        if boosts:
            boost_val = boosts.get(doc_id)
            if boost_val is not None:
                try:
                    multiplier *= max(0.0, float(boost_val))
                except Exception:
                    pass
    return multiplier


def _parse_scope_boosts(scope: Optional[Dict[str, Any]]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    if not isinstance(scope, dict):
        return boosts
    for doc_id, val in (scope.get("boost") or {}).items():
        try:
            boosts[str(doc_id)] = max(0.0, float(val))
        except Exception:
            continue
    for doc_id, val in (scope.get("demote") or {}).items():
        try:
            boosts[str(doc_id)] = max(0.0, float(val))
        except Exception:
            continue
    return boosts


def _sanitize_triage_for_plan(triage: Dict[str, Any]) -> Dict[str, Any]:
    pages_out: List[Dict[str, Any]] = []
    for entry in (triage or {}).get("pages", []):
        if not isinstance(entry, dict):
            continue
        page = entry.get("page")
        try:
            page_int = int(page)
        except Exception:
            page_int = None
        pages_out.append(
            {
                "page": page_int,
                "route": entry.get("route"),
                "text_chars": entry.get("text_chars"),
                "images": entry.get("images"),
                "vector_lines": entry.get("vector_lines"),
                "multicolumn_score": entry.get("multicolumn_score"),
                "has_table_token": entry.get("has_table_token"),
                "has_figure_token": entry.get("has_figure_token"),
            }
        )
    return {"pages": pages_out}


def _compute_plan_hash(payload: Dict[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _plan_file_path(doc_id: str) -> Path:
    _ensure_dir(PLAN_DIR)
    return PLAN_DIR / f"{doc_id}.plan.json"


def _load_plan(doc_id: str) -> Dict[str, Any]:
    path = _plan_file_path(doc_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    legacy_path = PLAN_DIR / f"{doc_id}.json"
    if legacy_path.exists():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_plan(doc_id: str, data: Dict[str, Any]) -> None:
    path = _plan_file_path(doc_id)
    _write_json(path, data)


def _artifact_dir(doc_id: str) -> Path:
    directory = ARTIFACT_DIR / doc_id
    _ensure_dir(directory)
    return directory


def _blocks_artifact_path(doc_id: str) -> Path:
    return _artifact_dir(doc_id) / "blocks.json"


def _chunks_artifact_path(doc_id: str) -> Path:
    return _artifact_dir(doc_id) / "chunks.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_artifact_path(artifact_path: Union[str, Path]) -> Path:
    if not artifact_path:
        raise ValueError("artifact_path is required")
    path = Path(artifact_path).expanduser().resolve()
    allowed_roots = [PLAN_DIR.resolve(), ARTIFACT_DIR.resolve()]
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        raise ValueError(f"Invalid artifact path: {artifact_path}")
    if not path.exists():
        raise ValueError(f"Artifact not found: {artifact_path}")
    return path


def _canary_config_path(collection: str, name: str = "default") -> Path:
    base = Path(os.getenv("CANARY_DIR", "config/canaries"))
    return base / collection / f"{name}.json"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _audit(event: str, payload: Dict[str, Any]) -> None:
    if not event:
        return
    try:
        _ensure_dir(AUR_AUDIT_PATH.parent)
        record = {
            "event": event,
            "ts": time.time(),
            **payload,
        }
        with AUR_AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("audit_log_failed", exc_info=True)


def _audit_subjects(subjects: List[str]) -> List[str]:
    return [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in subjects if s]


def _select_chunk_profile(blocks: List[Any]) -> str:
    if not blocks:
        return "fixed_window"
    table_rows = sum(1 for b in blocks if getattr(b, "type", "") == "table_row")
    if table_rows:
        return "table_row"
    step_like = 0
    for b in blocks:
        btype = getattr(b, "type", "")
        text = (getattr(b, "text", "") or "").strip()
        if btype == "list" and text:
            step_like += 1
        elif re.match(r"^\d+(?:\.\d+)*\s", text):
            step_like += 1
    if step_like and step_like >= max(3, len(blocks) // 4):
        return "procedure_block"
    return "heading_based"


def _summarize_chunks(chunks: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks[:limit]):
        text = chunk.get("text") or ""
        summaries.append(
            {
                "index": idx,
                "pages": chunk.get("pages"),
                "types": chunk.get("types"),
                "length": len(text),
                "preview": text[:280],
            }
        )
    return summaries


def _ensure_plan_defaults(plan_data: Dict[str, Any]) -> None:
    plan_data.setdefault("model_version", INGEST_MODEL_VERSION)
    plan_data.setdefault("prompt_sha", INGEST_PROMPT_SHA)
    calls = plan_data.get("metadata_calls")
    try:
        plan_data["metadata_calls"] = int(calls)
    except Exception:
        plan_data["metadata_calls"] = 0
    client_section = plan_data.setdefault("client_orchestration", {})
    if not isinstance(client_section, dict):
        client_section = {}
        plan_data["client_orchestration"] = client_section
    client_section.setdefault("decisions", [])


def _is_table_row(types: Any) -> bool:
    if isinstance(types, list):
        return any(str(t).lower() == "table_row" for t in types)
    if isinstance(types, str):
        return "table_row" in types.lower()
    return False


def _ensure_score_bucket(row: Dict[str, Any]) -> Dict[str, float]:
    scores = row.get("scores")
    if not isinstance(scores, dict):
        scores = {}
        row["scores"] = scores
    return scores


async def _perform_upsert(
    doc_id: str,
    collection_name: str,
    chunks_artifact: Path,
    *,
    metadata_artifact: Optional[Path] = None,
    thin_payload: Optional[bool] = None,
    skip_vectors: bool = False,
    update_graph_links: bool = True,
    update_summary_index: bool = True,
    fts_recreate: bool = False,
) -> Dict[str, Any]:
    metadata_arg: Optional[str] = metadata_artifact.as_posix() if metadata_artifact else None
    fts_db_path = _get_fts_db_path(collection_name)
    try:
        result = await asyncio.to_thread(
            upsert_document_artifacts,
            doc_id,
            collection_name,
            chunks_artifact.as_posix(),
            metadata_artifact=metadata_arg,
            ollama_url=OLLAMA_URL,
            ollama_model=OLLAMA_MODEL,
            embed_batch_size=INGEST_EMBED_BATCH,
            embed_timeout=INGEST_EMBED_TIMEOUT,
            embed_parallel=INGEST_EMBED_PARALLEL,
            embed_threads=INGEST_EMBED_THREADS,
            embed_keepalive=INGEST_EMBED_KEEPALIVE,
            embed_force_per_item=INGEST_EMBED_FORCE_PER_ITEM,
            embed_robust=INGEST_EMBED_ROBUST,
            qdrant_client=qdr,
            qdrant_metric=QDRANT_METRIC,
            fts_db_path=fts_db_path,
            fts_recreate=fts_recreate,
            thin_payload=thin_payload,
            update_graph_links=update_graph_links,
            update_summary_index=update_summary_index,
            skip_vectors=skip_vectors,
            sparse_expander=SPARSE_EXPANDER if HAVE_SPARSE_SPLADE else None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("ingest.upsert failed for %s", doc_id)
        return {"error": "upsert_failed", "detail": str(exc)}

    plan = _load_plan(doc_id)
    if plan:
        result.setdefault("plan_hash", plan.get("plan_hash"))
        result.setdefault("model_version", plan.get("model_version"))
        result.setdefault("prompt_sha", plan.get("prompt_sha"))
    result.setdefault("status", "ok")
    return result


def _record_client_decisions(
    doc_id: str,
    decisions: Optional[List[Dict[str, Any]]] = None,
    client_meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not doc_id:
        return
    if not decisions and not client_meta:
        return
    plan = _load_plan(doc_id)
    if not isinstance(plan, dict):
        plan = {"doc_id": doc_id}
    orchestr = plan.setdefault("client_orchestration", {})
    if not isinstance(orchestr, dict):
        orchestr = {}
        plan["client_orchestration"] = orchestr
    if client_meta:
        client_id = client_meta.get("client_id")
        client_model = client_meta.get("client_model")
        if client_id:
            orchestr["client_id"] = client_id
        if client_model:
            orchestr["client_model"] = client_model
    entries = orchestr.setdefault("decisions", [])
    if decisions:
        for decision in decisions:
            if isinstance(decision, dict):
                entry = dict(decision)
            else:
                entry = {"detail": str(decision)}
            entry.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")
            entries.append(entry)
    _save_plan(doc_id, plan)


@mcp.tool(name="ingest.extract_with_strategy", title="Ingest: Extract Blocks", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def ingest_extract(ctx: Context, path: str, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract document blocks using Docling.

    Processes a document file to extract structured blocks (paragraphs,
    tables, headings, figures) with metadata. Creates blocks artifact.

    Args:
        ctx: MCP context with session metadata
        path: Absolute path to document file (PDF, DOCX, etc.)
        plan: Optional extraction plan with doc_id, triage, chunk_profile

    Returns:
        Dict with doc_id, block_count, artifact path, plan_hash
        Error dict if file not found or extraction fails
    """
    plan = plan or {}
    if not path:
        return {"error": "missing_path"}
    path_obj = Path(path).expanduser()
    if not path_obj.is_file():
        return {"error": "not_found", "detail": f"file not found: {path_obj.as_posix()}"}

    doc_id_expected = plan.get("doc_id")
    doc_id = doc_id_expected or _doc_id_for_path(path_obj)
    if doc_id_expected and doc_id_expected != doc_id:
        return {"error": "doc_id_mismatch", "detail": f"plan doc_id {doc_id_expected} does not match computed {doc_id}"}

    plan_override = plan.get("triage")
    try:
        blocks, triage_info = await asyncio.to_thread(extract_document_blocks, path_obj, doc_id, plan_override)
    except TypeError:
        blocks, triage_info = await asyncio.to_thread(extract_document_blocks, path_obj, doc_id)
    except Exception as exc:
        return {"error": "extraction_failed", "detail": str(exc)}

    sanitized = _sanitize_triage_for_plan(triage_info)
    serialized_blocks = _serialize_blocks(blocks)

    artifact_path = _blocks_artifact_path(doc_id)
    artifact_payload = {
        "doc_id": doc_id,
        "path": path_obj.as_posix(),
        "triage": sanitized,
        "blocks": serialized_blocks,
        "block_count": len(serialized_blocks),
    }

    plan_data = _load_plan(doc_id)
    plan_data.update({
        "doc_id": doc_id,
        "path": path_obj.as_posix(),
        "triage": sanitized,
    })
    if plan.get("chunk_profile"):
        plan_data["chunk_profile"] = plan.get("chunk_profile")
    if isinstance(plan.get("chunk_params"), dict):
        plan_data["chunk_params"] = plan.get("chunk_params")
    _ensure_plan_defaults(plan_data)
    plan_data.setdefault("chunk_profile", plan_data.get("chunk_profile") or "auto")
    plan_data.setdefault("chunk_params", plan_data.get("chunk_params") or {})
    plan_hash = _compute_plan_hash(plan_data)
    plan_data["plan_hash"] = plan_hash
    _save_plan(doc_id, plan_data)

    artifact_payload["plan_hash"] = plan_hash
    _write_json(artifact_path, artifact_payload)

    return {
        "doc_id": doc_id,
        "block_count": len(serialized_blocks),
        "artifact": artifact_path.as_posix(),
        "plan_hash": plan_hash,
    }


@mcp.tool(name="ingest.validate_extraction", title="Ingest: Validate Extraction", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def ingest_validate_extraction(
    ctx: Context,
    artifact_ref: str,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not artifact_ref:
        return {"error": "missing_artifact"}
    try:
        artifact_path = _validate_artifact_path(artifact_ref)
    except ValueError as exc:
        return {"error": "invalid_artifact_path", "detail": str(exc)}
    data = _read_json(artifact_path)
    if not data:
        return {"error": "artifact_not_found", "detail": f"no artifact at {artifact_path.as_posix()}"}

    blocks = data.get("blocks") or []
    rule_set = rules or {}

    block_count = len(blocks)
    table_blocks = sum(1 for block in blocks if str(block.get("type")).lower() in {"table", "table_row"})
    heading_blocks = sum(1 for block in blocks if str(block.get("type")).lower() == "heading")
    text_chars = sum(len(str(block.get("text") or "")) for block in blocks)
    pages = {block.get("page") for block in blocks if block.get("page") is not None}

    issues: List[str] = []
    min_blocks = int(rule_set.get("min_blocks") or 0)
    if min_blocks and block_count < min_blocks:
        issues.append(f"min_blocks:{block_count}<{min_blocks}")
    min_text_chars = int(rule_set.get("min_text_chars") or 0)
    if min_text_chars and text_chars < min_text_chars:
        issues.append(f"min_text_chars:{text_chars}<{min_text_chars}")
    min_pages = int(rule_set.get("min_pages") or 0)
    if min_pages and len(pages) < min_pages:
        issues.append(f"min_pages:{len(pages)}<{min_pages}")
    if rule_set.get("expect_tables") and table_blocks == 0:
        issues.append("expect_tables:true but no table blocks detected")
    if rule_set.get("expect_headings") and heading_blocks == 0:
        issues.append("expect_headings:true but no heading blocks detected")

    stats = {
        "block_count": block_count,
        "table_blocks": table_blocks,
        "heading_blocks": heading_blocks,
        "text_chars": text_chars,
        "page_count": len(pages),
        "plan_hash": data.get("plan_hash"),
        "doc_id": data.get("doc_id"),
    }

    _audit("ingest_validate_extraction", {
        "artifact": artifact_path.as_posix(),
        "stats": stats,
        "issues": issues,
    })

    return {
        "valid": not issues,
        "issues": issues,
        "stats": stats,
    }


@mcp.tool(name="ingest.chunk_with_guidance", title="Ingest: Chunk Blocks", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def ingest_chunk(ctx: Context, artifacts_ref: str, profile: str = "auto", max_chars: int = 1800, overlap_sentences: int = 1) -> Dict[str, Any]:
    """Chunk extracted blocks into retrieval-sized pieces.

    Splits document blocks into chunks suitable for embedding and retrieval.
    Multiple profiles available for different document types.

    Args:
        ctx: MCP context with session metadata
        artifacts_ref: Path to blocks artifact from extraction
        profile: Chunking strategy - auto/fixed_window/heading_based/procedure_block/table_row
        max_chars: Maximum characters per chunk (default 1800, recommend 700 for reranker)
        overlap_sentences: Sentence overlap between chunks (default 1)

    Returns:
        Dict with doc_id, chunk_count, chunk_profile, plan_hash, artifact path
        Includes sample_chunks preview
    """
    if not artifacts_ref:
        return {"error": "missing_artifact"}
    try:
        artifact_path = _validate_artifact_path(artifacts_ref)
    except ValueError as exc:
        return {"error": "invalid_artifact_path", "detail": str(exc)}
    data = _read_json(artifact_path)
    if not data:
        return {"error": "artifact_not_found", "detail": f"no artifact at {artifact_path.as_posix()}"}

    doc_id = data.get("doc_id")
    if not doc_id:
        return {"error": "missing_doc_id", "detail": "artifact missing doc_id"}

    blocks_serialized = data.get("blocks") or []
    blocks = _deserialize_blocks(blocks_serialized)

    requested_profile = (profile or "auto").lower()
    if requested_profile not in {"auto", "fixed_window", "heading_based", "procedure_block", "table_row"}:
        requested_profile = "auto"
    selected_profile = _select_chunk_profile(blocks) if requested_profile == "auto" else requested_profile

    max_chars = max(200, int(max_chars))
    overlap_sentences_val = max(0, int(overlap_sentences))
    try:
        chunks, raw_text = await asyncio.to_thread(
            chunk_blocks,
            blocks,
            max_chars,
            overlap_sentences_val,
            selected_profile,
        )
    except Exception as exc:
        return {"error": "chunk_failed", "detail": str(exc)}

    content_hash = hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()

    plan_data = _load_plan(doc_id)
    if not plan_data.get("triage") and data.get("triage"):
        plan_data["triage"] = data.get("triage")
    plan_data.update(
        {
            "doc_id": doc_id,
            "path": data.get("path"),
            "chunk_profile": selected_profile,
            "chunk_params": {
                "max_chars": max_chars,
                "overlap_sentences": overlap_sentences_val,
            },
            "content_hash": content_hash,
        }
    )
    _ensure_plan_defaults(plan_data)

    chunk_artifact_path = _chunks_artifact_path(doc_id)
    chunk_payload = {
        "doc_id": doc_id,
        "path": data.get("path"),
        "triage": data.get("triage"),
        "chunk_profile": selected_profile,
        "chunk_params": {
            "max_chars": max_chars,
            "overlap_sentences": overlap_sentences_val,
        },
        "chunks": chunks,
        "chunk_count": len(chunks),
        "raw_text": raw_text,
        "content_hash": content_hash,
        "doc_metadata": plan_data.get("doc_metadata"),
        "model_version": plan_data.get("model_version"),
        "prompt_sha": plan_data.get("prompt_sha"),
        "metadata_calls": plan_data.get("metadata_calls", 0),
    }

    plan_hash = _compute_plan_hash(plan_data)
    plan_data["plan_hash"] = plan_hash
    chunk_payload["plan_hash"] = plan_hash

    _save_plan(doc_id, plan_data)
    _write_json(chunk_artifact_path, chunk_payload)

    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "chunk_profile": selected_profile,
        "plan_hash": plan_hash,
        "artifact": chunk_artifact_path.as_posix(),
        "sample_chunks": _summarize_chunks(chunks),
    }


@mcp.tool(name="ingest.generate_metadata", title="Ingest: Generate Metadata", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def ingest_generate_metadata(ctx: Context, doc_id: str, artifact_ref: Optional[str] = None, policy: str = "strict_v1") -> Dict[str, Any]:
    if not doc_id:
        return {"error": "missing_doc_id"}
    if artifact_ref:
        try:
            chunk_path = _validate_artifact_path(artifact_ref)
        except ValueError as exc:
            return {"error": "invalid_artifact_path", "detail": str(exc)}
    else:
        chunk_path = _chunks_artifact_path(doc_id)
    data = _read_json(chunk_path)
    if not data:
        return {"error": "artifact_not_found", "detail": f"no chunk artifact for {doc_id}"}

    raw_text = data.get("raw_text") or ""
    chunks = data.get("chunks") or []
    metadata, rejects = generate_metadata(raw_text, chunks)
    metadata_bytes = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) if metadata else 0

    plan_data = _load_plan(doc_id)
    plan_data.setdefault("doc_id", doc_id)
    _ensure_plan_defaults(plan_data)
    calls_used = plan_data.get("metadata_calls", 0)
    if MAX_METADATA_CALLS_PER_DOC >= 0 and calls_used >= MAX_METADATA_CALLS_PER_DOC:
        return {
            "doc_id": doc_id,
            "error": "metadata_budget_exceeded",
            "detail": f"metadata calls {calls_used} exceeded limit {MAX_METADATA_CALLS_PER_DOC}",
            "plan_hash": plan_data.get("plan_hash"),
        }
    if metadata:
        if metadata_bytes > MAX_METADATA_BYTES:
            rejects.append(f"metadata_bytes_exceeded:{metadata_bytes}>{MAX_METADATA_BYTES}")
            metadata = {}
        else:
            plan_data["doc_metadata"] = metadata
            data["doc_metadata"] = metadata

    plan_data["metadata_calls"] = calls_used + 1
    plan_hash = _compute_plan_hash(plan_data)
    plan_data["plan_hash"] = plan_hash
    data["plan_hash"] = plan_hash
    data["model_version"] = plan_data.get("model_version")
    data["prompt_sha"] = plan_data.get("prompt_sha")
    data["metadata_calls"] = plan_data.get("metadata_calls")

    _save_plan(doc_id, plan_data)
    _write_json(chunk_path, data)

    return {
        "doc_id": doc_id,
        "doc_metadata": metadata,
        "rejects": rejects,
        "plan_hash": plan_hash,
        "metadata_bytes": metadata_bytes,
        "metadata_calls": plan_data.get("metadata_calls", 0),
    }


@mcp.tool(name="ingest.assess_quality", title="Ingest: Assess Quality", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def ingest_assess_quality(ctx: Context, doc_id: str, artifact_ref: Optional[str] = None) -> Dict[str, Any]:
    if not doc_id:
        return {"error": "missing_doc_id"}
    if artifact_ref:
        try:
            chunk_path = _validate_artifact_path(artifact_ref)
        except ValueError as exc:
            return {"error": "invalid_artifact_path", "detail": str(exc)}
    else:
        chunk_path = _chunks_artifact_path(doc_id)
    data = _read_json(chunk_path)
    if not data:
        return {"error": "artifact_not_found", "detail": f"no chunk artifact for {doc_id}"}

    chunks = data.get("chunks") or []
    chunk_count = len(chunks)
    table_rows = sum(1 for chunk in chunks if "table_row" in (chunk.get("types") or []))
    total_len = sum(len(chunk.get("text") or "") for chunk in chunks)
    avg_len = total_len / chunk_count if chunk_count else 0.0
    pages_covered: set = set()
    for chunk in chunks:
        for page in chunk.get("pages") or []:
            pages_covered.add(page)

    metadata_available = bool(data.get("doc_metadata"))
    plan_hash = data.get("plan_hash") or _load_plan(doc_id).get("plan_hash")
    model_version = data.get("model_version") or _load_plan(doc_id).get("model_version")
    prompt_sha = data.get("prompt_sha") or _load_plan(doc_id).get("prompt_sha")
    metadata_calls = data.get("metadata_calls") or _load_plan(doc_id).get("metadata_calls")
    try:
        metadata_calls = int(metadata_calls)
    except Exception:
        metadata_calls = 0
    doc_metadata = data.get("doc_metadata")
    metadata_bytes = len(json.dumps(doc_metadata, ensure_ascii=False).encode("utf-8")) if doc_metadata else 0

    warnings: List[str] = []
    if chunk_count == 0:
        warnings.append("no_chunks")
    if not metadata_available:
        warnings.append("no_metadata")

    canary_results = await _run_canaries(doc_id, data)

    return {
        "doc_id": doc_id,
        "chunk_count": chunk_count,
        "table_row_chunks": table_rows,
        "avg_chunk_chars": round(avg_len, 2),
        "page_coverage": len(pages_covered),
        "has_metadata": metadata_available,
        "plan_hash": plan_hash,
        "model_version": model_version,
        "prompt_sha": prompt_sha,
        "metadata_calls": metadata_calls,
        "metadata_bytes": metadata_bytes,
        "warnings": warnings,
        "canaries": canary_results,
    }


@mcp.tool(name="ingest.enhance", title="Ingest: Enhance Artifacts", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def ingest_enhance(ctx: Context, doc_id: str, op: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    supported_ops = {"add_synonyms", "link_crossrefs", "fix_table_pages"}
    if not doc_id:
        return {"error": "missing_doc_id"}
    if op not in supported_ops:
        return {"error": "unsupported_operation", "supported_ops": sorted(supported_ops)}
    args = args or {}
    chunk_path = _chunks_artifact_path(doc_id)
    data = _read_json(chunk_path)
    if not data:
        return {"error": "artifact_not_found", "detail": f"no chunk artifact for {doc_id}"}
    chunks = data.get("chunks") or []
    modified = False

    if op == "add_synonyms":
        additions = args.get("synonyms") if isinstance(args, dict) else None
        if not isinstance(additions, dict):
            return {"error": "invalid_args", "detail": "synonyms dict required"}
        for chunk in chunks:
            meta = chunk.setdefault("synonyms", [])
            if not isinstance(meta, list):
                chunk["synonyms"] = meta = []
            for term, repls in additions.items():
                if not term or not repls:
                    continue
                entry = {"term": term, "replacements": repls}
                if entry not in meta:
                    meta.append(entry)
                    modified = True

    elif op == "link_crossrefs":
        refs = args.get("references") if isinstance(args, dict) else None
        if not isinstance(refs, list):
            return {"error": "invalid_args", "detail": "references list required"}
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            chunk_id = ref.get("chunk_id")
            target = ref.get("target")
            if not chunk_id or not target:
                continue
            for chunk in chunks:
                if chunk.get("chunk_id") == chunk_id:
                    links = chunk.setdefault("crossrefs", [])
                    if target not in links:
                        links.append(target)
                        modified = True
                    break

    elif op == "fix_table_pages":
        fixes = args.get("pages") if isinstance(args, dict) else None
        if not isinstance(fixes, dict):
            return {"error": "invalid_args", "detail": "pages dict required"}
        for chunk in chunks:
            if "table_row" not in (chunk.get("types") or []):
                continue
            element_ids = chunk.get("element_ids") or []
            for element in element_ids:
                page = fixes.get(element)
                if isinstance(page, int):
                    chunk["pages"] = [page]
                    modified = True

    if modified:
        data["chunks"] = chunks
        _write_json(chunk_path, data)

    return {
        "doc_id": doc_id,
        "operation": op,
        "modified": modified,
        "status": "completed",
    }


@mcp.tool(name="ingest.upsert", title="Ingest: Upsert Document", annotations={
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def ingest_upsert_tool(
    ctx: Context,
    doc_id: str,
    collection: Optional[str] = None,
    chunks_artifact: Optional[str] = None,
    metadata_artifact: Optional[str] = None,
    thin_payload: Optional[bool] = None,
    skip_vectors: bool = False,
    update_graph: bool = True,
    update_summary: bool = True,
    fts_rebuild: bool = False,
    client_id: Optional[str] = None,
    client_model: Optional[str] = None,
    client_decisions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Upsert document chunks to Qdrant and FTS database.

    Final ingestion step: embeds chunks, upserts to Qdrant vector store,
    and indexes in SQLite FTS5. Deterministic chunk IDs enable safe re-runs.

    Args:
        ctx: MCP context with session metadata
        doc_id: Document identifier from extraction
        collection: Target Qdrant collection name
        chunks_artifact: Path to chunks JSON (uses default if None)
        metadata_artifact: Optional path to metadata JSON
        thin_payload: Store minimal payload in vectors (saves memory)
        skip_vectors: Skip vector embedding (FTS only)
        update_graph: Update knowledge graph links
        update_summary: Update summary index
        fts_rebuild: Recreate FTS index from scratch
        client_id: Client identifier for provenance
        client_model: Model used for client decisions
        client_decisions: List of decision objects for audit trail

    Returns:
        Dict with status, doc_id, chunk_count, collection, timings
    """
    if not doc_id:
        return {"error": "missing_doc_id"}
    if chunks_artifact:
        try:
            chunk_path = _validate_artifact_path(chunks_artifact)
        except ValueError as exc:
            return {"error": "invalid_artifact_path", "detail": str(exc)}
    else:
        chunk_path = _chunks_artifact_path(doc_id)
    if not chunk_path.exists():
        return {"error": "artifact_not_found", "detail": f"no chunk artifact for {doc_id}"}
    if metadata_artifact:
        try:
            meta_path = _validate_artifact_path(metadata_artifact)
        except ValueError as exc:
            return {"error": "invalid_metadata_path", "detail": str(exc)}
    else:
        meta_path = None
    if meta_path and not meta_path.exists():
        return {"error": "metadata_not_found", "detail": f"no metadata artifact at {meta_path.as_posix()}"}

    plan = _load_plan(doc_id)
    plan_collection = plan.get("collection") if isinstance(plan, dict) else None
    resolved_collection = collection or plan_collection
    if resolved_collection and resolved_collection in SCOPES:
        resolved_collection = SCOPES[resolved_collection].get("collection") or resolved_collection
    if not resolved_collection:
        return {"error": "missing_collection", "detail": "Provide 'collection' or ensure plan includes collection"}

    if client_decisions is not None and not isinstance(client_decisions, list):
        return {"error": "invalid_client_decisions", "detail": "client_decisions must be a list of objects"}

    decisions_payload = []
    if client_decisions:
        for entry in client_decisions:
            if isinstance(entry, dict):
                decisions_payload.append(entry)
            else:
                decisions_payload.append({"detail": str(entry)})

    result = await _perform_upsert(
        doc_id,
        resolved_collection,
        chunk_path,
        metadata_artifact=meta_path,
        thin_payload=thin_payload,
        skip_vectors=skip_vectors,
        update_graph_links=update_graph,
        update_summary_index=update_summary,
        fts_recreate=fts_rebuild,
    )

    _record_client_decisions(
        doc_id,
        decisions=decisions_payload,
        client_meta={"client_id": client_id, "client_model": client_model},
    )

    if result.get("status") == "ok" and isinstance(plan, dict):
        if plan.get("collection") != resolved_collection:
            plan["collection"] = resolved_collection
            _save_plan(doc_id, plan)
        audit_payload = {
            "doc_id": doc_id,
            "collection": resolved_collection,
            "chunks_upserted": result.get("chunks_upserted"),
            "qdrant_points": result.get("qdrant_points"),
            "fts_rows": result.get("fts_rows"),
        }
        if client_id or client_model:
            audit_payload["client"] = {"client_id": client_id, "client_model": client_model}
        _audit("ingest_upsert", audit_payload)
    return result


@mcp.tool(name="ingest.upsert_batch", title="Ingest: Batch Upsert", annotations={
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def ingest_upsert_batch(
    ctx: Context,
    upserts: List[Dict[str, Any]],
    collection: Optional[str] = None,
    parallel: int = 4,
    thin_payload: Optional[bool] = None,
    skip_vectors: bool = False,
    update_graph: bool = True,
    update_summary: bool = True,
    fts_rebuild: bool = False,
    client_id: Optional[str] = None,
    client_model: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(upserts, list) or not upserts:
        return {"error": "empty_specs"}
    max_workers = max(1, int(parallel))
    sem = asyncio.Semaphore(max_workers)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    successes = 0
    start = time.perf_counter()

    async def run_one(spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        async with sem:
            spec_doc_id = spec.get("doc_id")
            if not spec_doc_id:
                return "", {"error": "missing_doc_id"}
            spec_collection = spec.get("collection") or collection
            chunk_path = spec.get("chunks_artifact") or _chunks_artifact_path(spec_doc_id).as_posix()
            meta_path = spec.get("metadata_artifact")
            result = await ingest_upsert_tool(
                ctx,
                spec_doc_id,
                collection=spec_collection,
                chunks_artifact=chunk_path,
                metadata_artifact=meta_path,
                thin_payload=spec.get("thin_payload", thin_payload),
                skip_vectors=spec.get("skip_vectors", skip_vectors),
                update_graph=spec.get("update_graph", update_graph),
                update_summary=spec.get("update_summary", update_summary),
                fts_rebuild=spec.get("fts_rebuild", fts_rebuild),
                client_id=spec.get("client_id", client_id),
                client_model=spec.get("client_model", client_model),
                client_decisions=spec.get("client_decisions"),
            )
            return spec_doc_id, result

    tasks = [asyncio.create_task(run_one(spec)) for spec in upserts]
    for task in asyncio.as_completed(tasks):
        doc_id_val, outcome = await task
        if outcome.get("status") == "ok":
            successes += 1
        else:
            failures.append({"doc_id": doc_id_val, "error": outcome})
        outcome.setdefault("doc_id", doc_id_val)
        results.append(outcome)

    elapsed = time.perf_counter() - start
    return {
        "status": "ok",
        "total_docs": len(upserts),
        "successful": successes,
        "failed": len(failures),
        "failures": failures,
        "results": results,
        "elapsed_seconds": round(elapsed, 3),
    }


@mcp.tool(name="ingest.generate_summary", title="Ingest: Store Summary", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def ingest_generate_summary_tool(
    ctx: Context,
    doc_id: str,
    summary_text: str,
    section_path: Optional[List[str]] = None,
    collection: Optional[str] = None,
    element_ids: Optional[List[str]] = None,
    summary_metadata: Optional[Dict[str, Any]] = None,
    client_id: Optional[str] = None,
    client_model: Optional[str] = None,
) -> Dict[str, Any]:
    if not doc_id:
        return {"error": "missing_doc_id"}
    if not isinstance(summary_text, str) or not summary_text.strip():
        return {"error": "empty_summary"}
    summary_clean = summary_text.strip()
    if len(summary_clean) > 1200:
        return {"error": "summary_too_long", "detail": "Summary must be <= 1200 characters"}

    plan = _load_plan(doc_id)
    try:
        _, collection_name, _ = _resolve_scope(collection)
    except Exception:
        plan_collection = plan.get("collection") if isinstance(plan, dict) else None
        if plan_collection and plan_collection in SCOPES:
            collection_name = SCOPES[plan_collection]["collection"]
        else:
            collection_name = plan_collection
    if not collection_name:
        return {"error": "missing_collection", "detail": "Provide 'collection' or ensure plan includes collection"}

    if isinstance(plan, dict):
        if plan.get("collection") != collection_name:
            plan["collection"] = collection_name
            _save_plan(doc_id, plan)

    section = section_path or []
    if not isinstance(section, list):
        return {"error": "invalid_section_path", "detail": "section_path must be a list of strings"}
    section = [str(part) for part in section]
    elements = element_ids or []
    if element_ids is not None:
        if not isinstance(element_ids, list):
            return {"error": "invalid_element_ids", "detail": "element_ids must be a list"}
        elements = [str(e) for e in element_ids]

    metadata = summary_metadata.copy() if isinstance(summary_metadata, dict) else {}
    metadata.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    if client_id:
        metadata.setdefault("client_id", client_id)
    if client_model:
        metadata.setdefault("client_model", client_model)

    await asyncio.to_thread(
        upsert_summary_entry,
        collection_name,
        doc_id,
        section,
        summary_clean,
        elements,
        metadata,
    )

    decision_entry = {
        "step": "summary_generation",
        "section_path": section,
        "summary_sha": hashlib.sha256(summary_clean.encode("utf-8")).hexdigest(),
    }
    if metadata:
        decision_entry["metadata"] = metadata
    _record_client_decisions(
        doc_id,
        decisions=[decision_entry],
        client_meta={"client_id": client_id, "client_model": client_model},
    )

    audit_payload = {
        "doc_id": doc_id,
        "collection": collection_name,
        "section_path": section,
    }
    if client_id or client_model:
        audit_payload["client"] = {"client_id": client_id, "client_model": client_model}
    _audit("ingest_generate_summary", audit_payload)

    return {
        "status": "ok",
        "doc_id": doc_id,
        "collection": collection_name,
        "section_path": section,
        "element_ids": elements,
        "metadata": metadata,
    }


@mcp.tool(name="ingest.corpus_upsert", title="Ingest: Corpus Upsert", annotations={
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def ingest_corpus_upsert(
    ctx: Context,
    root_path: str,
    collection: Optional[str] = None,
    extractor: str = "auto",
    chunk_profile: str = "auto",
    max_chars: int = 1800,
    overlap_sentences: int = 1,
    dry_run: bool = False,
    thin_payload: Optional[bool] = None,
    skip_vectors: bool = False,
    update_graph: bool = True,
    update_summary: bool = True,
    fts_rebuild: bool = False,
    extensions: Optional[str] = None,
    client_id: Optional[str] = None,
    client_model: Optional[str] = None,
) -> Dict[str, Any]:
    if not root_path:
        return {"error": "missing_root"}
    root = Path(root_path).expanduser()
    if not root.is_dir():
        return {"error": "invalid_root", "detail": f"directory not found: {root.as_posix()}"}

    try:
        _, collection_name, _ = _resolve_scope(collection)
    except Exception as exc:
        return {"error": "invalid_collection", "detail": str(exc)}

    extractor_normalized = (extractor or "auto").lower()
    profile_normalized = (chunk_profile or "auto").lower()
    max_chars = max(200, int(max_chars))
    overlap_sentences = max(0, int(overlap_sentences))

    if extensions:
        ext_set = {ext.strip().lower() for ext in extensions.split(",") if ext.strip()}
    else:
        ext_set = {".pdf", ".docx", ".txt"}

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ext_set and path.suffix.lower() not in ext_set:
            continue
        files.append(path)

    if not files:
        return {"status": "ok", "collection": collection_name, "processed": 0, "progress": [], "dry_run": dry_run}

    progress_log: List[Dict[str, Any]] = []
    processed = 0
    failures = 0
    rebuild_flag = bool(fts_rebuild)
    start = time.perf_counter()

    for file_path in files:
        path_str = file_path.as_posix()
        try:
            analyze = await ingest_analyze(ctx, path_str)
            if analyze.get("error"):
                progress_log.append({
                    "path": path_str,
                    "status": "error",
                    "error": analyze,
                })
                failures += 1
                continue
            doc_id = analyze.get("doc_id")
            if not doc_id:
                progress_log.append({
                    "path": path_str,
                    "status": "error",
                    "error": {"detail": "doc_id missing after analyze"},
                })
                failures += 1
                continue

            plan = _load_plan(doc_id)
            if isinstance(plan, dict):
                plan["collection"] = collection_name
                triage = plan.get("triage")
                if extractor_normalized in {"markitdown", "docling"} and isinstance(triage, dict):
                    for page in triage.get("pages", []):
                        if isinstance(page, dict):
                            page["route"] = extractor_normalized
                _save_plan(doc_id, plan)
            _record_client_decisions(
                doc_id,
                decisions=None,
                client_meta={"client_id": client_id, "client_model": client_model},
            )

            extract = await ingest_extract(ctx, path_str, plan=plan or {})
            if extract.get("error"):
                progress_log.append({
                    "path": path_str,
                    "doc_id": doc_id,
                    "status": "error",
                    "error": extract,
                })
                failures += 1
                continue

            chunk = await ingest_chunk(
                ctx,
                extract.get("artifact"),
                profile=profile_normalized,
                max_chars=max_chars,
                overlap_sentences=overlap_sentences,
            )
            if chunk.get("error"):
                progress_log.append({
                    "path": path_str,
                    "doc_id": doc_id,
                    "status": "error",
                    "error": chunk,
                })
                failures += 1
                continue
            chunk_artifact = chunk.get("artifact")

            metadata_result = await ingest_generate_metadata(ctx, doc_id, chunk_artifact)
            if metadata_result.get("error"):
                progress_log.append({
                    "path": path_str,
                    "doc_id": doc_id,
                    "status": "warning",
                    "warning": metadata_result,
                })

            if dry_run:
                processed += 1
                progress_log.append({
                    "path": path_str,
                    "doc_id": doc_id,
                    "status": "dry_run",
                    "plan_hash": chunk.get("plan_hash"),
                    "chunk_count": chunk.get("chunk_count"),
                })
                continue

            upsert_result = await _perform_upsert(
                doc_id,
                collection_name,
                Path(chunk_artifact),
                thin_payload=thin_payload,
                skip_vectors=skip_vectors,
                update_graph_links=update_graph,
                update_summary_index=update_summary,
                fts_recreate=rebuild_flag,
            )
            rebuild_flag = False  # Only rebuild once if requested
            if upsert_result.get("status") != "ok":
                failures += 1
                progress_log.append({
                    "path": path_str,
                    "doc_id": doc_id,
                    "status": "error",
                    "error": upsert_result,
                })
                continue

            processed += 1
            progress_log.append({
                "path": path_str,
                "doc_id": doc_id,
                "status": "ok",
                "plan_hash": upsert_result.get("plan_hash"),
                "chunks_upserted": upsert_result.get("chunks_upserted"),
                "qdrant_points": upsert_result.get("qdrant_points"),
            })
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("corpus upsert failed for %s", path_str)
            failures += 1
            progress_log.append({
                "path": path_str,
                "status": "error",
                "error": {"detail": str(exc)},
            })

    elapsed = time.perf_counter() - start
    return {
        "status": "ok",
        "collection": collection_name,
        "root": root.as_posix(),
        "total_files": len(files),
        "processed": processed,
        "failed": failures,
        "dry_run": dry_run,
        "progress": progress_log,
        "elapsed_seconds": round(elapsed, 3),
    }


async def _fetch_chunks_by_ids(chunk_ids: List[str], collection_name: str) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    try:
        store = _get_doc_store(collection_name)
        return await asyncio.to_thread(store.get_records_bulk, chunk_ids)
    except Exception as exc:
        logger.debug("fetch_chunk_records failed: %s", exc)
        return {}


async def _hydrate_qdrant_hits(hits, collection_name: str) -> None:
    missing: List[str] = []
    for h in hits or []:
        payload = getattr(h, "payload", None) or {}
        if payload.get("text"):
            continue
        chunk_id = str(getattr(h, "id", None) or payload.get("chunk_id") or "")
        if chunk_id:
            missing.append(chunk_id)
    if not missing:
        return
    fetched = await _fetch_chunks_by_ids(missing, collection_name)
    for h in hits or []:
        payload = getattr(h, "payload", None) or {}
        chunk_id = str(getattr(h, "id", None) or payload.get("chunk_id") or "")
        info = fetched.get(chunk_id)
        if not info:
            continue
        payload.setdefault("text", info.get("text"))
        payload.setdefault("chunk_start", info.get("chunk_start"))
        payload.setdefault("chunk_end", info.get("chunk_end"))
        payload.setdefault("doc_id", info.get("doc_id"))
        payload.setdefault("path", info.get("path"))
        payload.setdefault("filename", info.get("filename"))
        if "page_numbers" not in payload and info.get("page_numbers") is not None:
            payload["page_numbers"] = _parse_page_numbers(info.get("page_numbers"))
        if "mtime" not in payload and info.get("mtime") is not None:
            payload["mtime"] = info.get("mtime")
        for key in ("pages", "section_path", "element_ids", "bboxes", "types", "source_tools", "table_headers", "table_units", "chunk_profile", "plan_hash", "model_version", "prompt_sha", "doc_metadata"):
            if payload.get(key) in (None, "", []):
                value = info.get(key)
                if isinstance(value, list):
                    payload[key] = value
                elif value not in (None, ""):
                    payload[key] = _coerce_json(value)
        setattr(h, "payload", payload)
async def _ensure_row_texts(
    rows: List[Dict[str, Any]],
    collection: Optional[str],
    subjects: List[str],
) -> None:
    missing: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("id") or row.get("chunk_id") or "")
        if chunk_id:
            missing.append(chunk_id)
    fetched = await _fetch_chunks_by_ids(missing, collection or "")
    missing_qdrant = [cid for cid in missing if cid not in fetched]
    if missing_qdrant and collection:
        try:
            records = await asyncio.to_thread(
                qdr.retrieve,
                collection_name=collection,
                ids=missing_qdrant,
                with_vectors=False,
                with_payload=True,
            )
            for rec in records or []:
                cid = str(getattr(rec, "id", ""))
                if not cid or cid in fetched:
                    continue
                payload = rec.payload or {}
                payload.setdefault("chunk_id", cid)
                fetched[cid] = payload
        except Exception as exc:
            logger.debug("qdrant retrieve fallback failed: %s", exc)
    store = _get_doc_store(collection or "")
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("id") or row.get("chunk_id") or "")
        info = fetched.get(chunk_id)
        existing_text = row.pop("text", None)
        doc_id = row.get("doc_id") or (info.get("doc_id") if info else None)
        allowed = store.is_allowed(str(doc_id) if doc_id else None, collection, subjects)
        store.build_row(row, info, allowed, include_text=False)
        source_for_pages = info or row
        _set_page_fields(row, source_for_pages)
        if info:
            for key in ("plan_hash", "model_version", "prompt_sha"):
                if row.get(key) in (None, ""):
                    val = info.get(key)
                    if val not in (None, ""):
                        row[key] = val
        if allowed:
            row.pop("forbidden", None)
            row.pop("reason", None)
            if info:
                text = info.get("text")
                if text is not None:
                    row.setdefault("text", text)
            if existing_text and "text" not in row:
                row["text"] = existing_text
        else:
            row.pop("text", None)
            row.setdefault("reason", "access_denied")
        for key in ("section_path", "element_ids", "bboxes", "types", "source_tools"):
            value = row.get(key)
            if isinstance(value, str) and value:
                try:
                    row[key] = json.loads(value)
                except Exception:
                    pass
        th = row.get("table_headers")
        if isinstance(th, str) and th:
            try:
                row["table_headers"] = json.loads(th)
            except Exception:
                pass
        tu = row.get("table_units")
        if isinstance(tu, str) and tu:
            try:
                row["table_units"] = json.loads(tu)
            except Exception:
                pass
        dm = row.get("doc_metadata")
        if isinstance(dm, str) and dm:
            try:
                row["doc_metadata"] = json.loads(dm)
            except Exception:
                pass


def _normalize_response_profile(profile_str: str) -> ResponseProfile:
    """Safely convert response_profile string to enum, with fallback to SLIM.

    Args:
        profile_str: User-supplied profile string

    Returns:
        ResponseProfile enum value (SLIM if invalid input)
    """
    try:
        return ResponseProfile(profile_str.lower().strip())
    except (ValueError, AttributeError):
        logger.warning(f"Invalid response_profile '{profile_str}', defaulting to slim")
        return ResponseProfile.SLIM


def prune_row(row: Dict[str, Any], profile: ResponseProfile = ResponseProfile.SLIM) -> Dict[str, Any]:
    """Shared response formatter - reduces payload size by dropping heavy metadata.

    Args:
        row: Full chunk row with all metadata
        profile: Response profile (SLIM, FULL, or DIAGNOSTIC)

    Returns:
        Pruned row dict based on profile

    Profile behaviors:
        SLIM (default): Keep essential fields including chunk_id, text, doc_id, path,
                       section_path, page_numbers, chunk_start/end, score, route.
                       Preserves error/ACL markers. ~85% token reduction.
        FULL: Keep structural metadata (tables, bboxes, element_ids) too.
        DIAGNOSTIC: Return everything including provenance and score breakdowns.
    """
    # Preserve error and ACL denial markers regardless of profile
    if row.get("error") or row.get("forbidden") or row.get("abstain") or row.get("note"):
        return row

    if profile == ResponseProfile.DIAGNOSTIC:
        # Return everything unchanged
        return row

    if profile == ResponseProfile.SLIM:
        # Essential fields for citations + context + neighbor sorting
        return {
            "chunk_id": row.get("chunk_id"),
            "text": row.get("text"),
            "doc_id": row.get("doc_id"),
            "path": row.get("path"),
            "section_path": row.get("section_path", []),
            "page_numbers": row.get("page_numbers", []),
            "chunk_start": row.get("chunk_start"),  # Required for neighbor sorting
            "chunk_end": row.get("chunk_end"),      # Required for neighbor sorting
            "score": row.get("final_score") or row.get("score", 0.0),
            "route": row.get("route"),
        }

    if profile == ResponseProfile.FULL:
        # Keep structural metadata, drop provenance
        keep_fields = {
            "chunk_id", "text", "doc_id", "path",
            "chunk_start", "chunk_end", "section_path",
            "pages", "page_numbers", "filename",
            "score", "final_score", "route",
            "element_ids", "bboxes", "types",
            "table_headers", "table_units", "source_tools"
        }
        return {k: v for k, v in row.items() if k in keep_fields}

    # Fallback to full
    return row


def _extract_score(row: Dict[str, Any]) -> float:
    for key in ("final_score", "score", "rerank_score", "rrf_score", "dense_score"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _best_score(rows: List[Dict[str, Any]]) -> float:
    best = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "error" in row:
            continue
        if row.get("note"):
            continue
        if row.get("forbidden"):
            continue
        best = max(best, _extract_score(row))
    return best


def _normalize_score(value: float, stats: Optional[Tuple[float, float]]) -> float:
    if not stats:
        return 0.0
    min_val, max_val = stats
    if max_val - min_val < 1e-9:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def _combined_score(
    bm25: Optional[float],
    dense: Optional[float],
    rerank: Optional[float],
    bm_stats: Optional[Tuple[float, float]],
    dense_stats: Optional[Tuple[float, float]],
    rerank_stats: Optional[Tuple[float, float]],
    decay: float,
) -> float:
    weight_sum = max(MIX_W_BM25 + MIX_W_DENSE + MIX_W_RERANK, 1e-6)
    bm_norm = _normalize_score(bm25 or 0.0, bm_stats) if bm25 is not None else 0.0
    dense_norm = _normalize_score(dense or 0.0, dense_stats) if dense is not None else 0.0
    rerank_norm = _normalize_score(rerank or 0.0, rerank_stats) if rerank is not None else 0.0
    combined = (
        MIX_W_BM25 * bm_norm
        + MIX_W_DENSE * dense_norm
        + MIX_W_RERANK * rerank_norm
    ) / weight_sum
    return combined * decay


async def plan_route(query: str) -> Dict[str, Any]:
    """Choose a retrieval route based on cheap heuristics."""
    tokens = query.split()
    n_tokens = len(tokens)
    has_caps = any(any(c.isupper() for c in tok) for tok in tokens)
    if n_tokens <= 3 and "?" not in query:
        if HAVE_SPARSE_SPLADE:
            return {"route": "sparse_splade", "k": 24}
        return {"route": "sparse", "k": 24}
    if n_tokens <= 4 and has_caps:
        return {"route": "sparse_splade" if HAVE_SPARSE_SPLADE else "hybrid", "k": 32}
    if "?" in query or n_tokens > 10:
        if HAVE_COLBERT:
            return {"route": "colbert", "k": 32}
        return {"route": "rerank", "k": 24}
    if HAVE_COLBERT and n_tokens >= 6:
        return {"route": "colbert", "k": 24}
    return {"route": "hybrid", "k": 24}


def _should_retry_sparse(query: str, rows: List[Dict[str, Any]]) -> bool:
    if not rows:
        return True
    best = _best_score(rows)
    if best >= max(ANSWERABILITY_THRESHOLD, 0.35):
        return False
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9]{4,}", query)]
    if not tokens:
        return False
    top = next((row for row in rows if row.get("text")), None)
    if not top:
        return True
    text = (top.get("text") or "").lower()
    coverage = sum(1 for tok in tokens if tok in text) / max(len(tokens), 1)
    if coverage < 0.25:
        return True
    types = top.get("types") or []
    if not _is_table_row(types) and coverage < 0.4 and best < 0.2:
        return True
    return False


async def _run_semantic(
    collection: str,
    query: str,
    query_vec: List[float],
    top_k: int,
    return_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    limit = max(top_k, return_k)
    try:
        start = time.perf_counter()
        hits = await asyncio.to_thread(
            qdr.search,
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        timings["qdrant_ms"] = timings.get("qdrant_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return [{"error": "qdrant_search_failed", "detail": str(exc)}]

    await _hydrate_qdrant_hits(hits, collection)
    rows: List[Dict[str, Any]] = []
    for h in hits[:return_k]:
        pl = h.payload or {}
        dense_score = getattr(h, "score", 0.0)
        decay = _decay_factor(pl.get("mtime"))
        prior_mult = _prior_multiplier(pl.get("doc_id"), subjects, boosts)
        final_score = dense_score * prior_mult
        row = {
            "score": final_score,
            "final_score": final_score,
            "dense_score": dense_score,
            "decay_factor": decay,
            "id": str(getattr(h, "id", "")),
            "chunk_id": str(getattr(h, "id", "")),
            "doc_id": pl.get("doc_id"),
            "path": pl.get("path"),
            "chunk_start": pl.get("chunk_start"),
            "chunk_end": pl.get("chunk_end"),
            "text": pl.get("text"),
            "section_path": pl.get("section_path"),
            "element_ids": pl.get("element_ids"),
            "bboxes": pl.get("bboxes"),
            "types": pl.get("types"),
            "source_tools": pl.get("source_tools"),
            "table_headers": pl.get("table_headers"),
            "table_units": pl.get("table_units"),
            "chunk_profile": pl.get("chunk_profile"),
            "plan_hash": pl.get("plan_hash"),
            "model_version": pl.get("model_version"),
            "prompt_sha": pl.get("prompt_sha"),
            "doc_metadata": pl.get("doc_metadata"),
            "scores": _score_breakdown(
                bm25=None,
                dense=dense_score,
                rrf=None,
                rerank=None,
                decay=decay,
                final=final_score,
            ),
        }
        _set_page_fields(row, pl)
        _ensure_score_bucket(row)["prior"] = prior_mult
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    _annotate_rows(rows, query)
    return rows


async def _run_sparse(
    collection: str,
    query: str,
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    fts_db_path = _get_fts_db_path(collection)
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    start = time.perf_counter()
    lexical_hits = await asyncio.to_thread(_fts_search, query, limit, fts_db_path)
    timings["fts_ms"] = timings.get("fts_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    bm_values = [row.get("bm25") for row in lexical_hits if row.get("bm25") is not None]
    bm_stats = (min(bm_values), max(bm_values)) if bm_values else None
    rows: List[Dict[str, Any]] = []
    for item in lexical_hits[:return_k]:
        bm = item.get("bm25")
        decay = _decay_factor(item.get("mtime"))
        score = _combined_score(
            bm25=bm,
            dense=None,
            rerank=None,
            bm_stats=bm_stats,
            dense_stats=None,
            rerank_stats=None,
            decay=decay,
        )
        prior_mult = _prior_multiplier(item.get("doc_id"), subjects, boosts)
        score *= prior_mult
        rows.append({
            "score": score,
            "final_score": score,
            "bm25_score": bm,
            "decay_factor": decay,
            "id": item.get("chunk_id"),
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "path": item.get("path"),
            "chunk_start": item.get("chunk_start"),
            "chunk_end": item.get("chunk_end"),
            "text": item.get("text"),
            "section_path": item.get("section_path"),
            "element_ids": item.get("element_ids"),
            "bboxes": item.get("bboxes"),
            "types": item.get("types"),
            "source_tools": item.get("source_tools"),
            "table_headers": item.get("table_headers"),
            "table_units": item.get("table_units"),
            "chunk_profile": item.get("chunk_profile"),
            "plan_hash": item.get("plan_hash"),
            "model_version": item.get("model_version"),
            "prompt_sha": item.get("prompt_sha"),
            "doc_metadata": item.get("doc_metadata"),
            "scores": _score_breakdown(
                bm25=bm,
                dense=None,
                rrf=None,
                rerank=None,
                decay=decay,
                final=score,
            ),
        })
        _set_page_fields(rows[-1], item)
        _ensure_score_bucket(rows[-1])["prior"] = prior_mult
    await _ensure_row_texts(rows, collection, subjects)
    _annotate_rows(rows, query)
    return rows


async def _run_sparse_splade(
    collection: str,
    query: str,
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not HAVE_SPARSE_SPLADE:
        return [{"error": "sparse_expander_disabled"}]
    fts_db_path = _get_fts_db_path(collection)
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    start = time.perf_counter()
    query_weights = _encode_sparse_query(query)
    timings["sparse_expand_ms"] = timings.get("sparse_expand_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    if not query_weights:
        return []
    start = time.perf_counter()
    sparse_hits = await asyncio.to_thread(_sparse_terms_search, query_weights, limit, fts_db_path)
    timings["sparse_terms_ms"] = timings.get("sparse_terms_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    if not sparse_hits:
        return []
    scores = [float(row.get("sparse_score", 0.0)) for row in sparse_hits if row.get("sparse_score") is not None]
    score_stats = (min(scores), max(scores)) if scores else None
    rows: List[Dict[str, Any]] = []
    for item in sparse_hits[:return_k]:
        sparse_score = float(item.get("sparse_score", 0.0) or 0.0)
        decay = _decay_factor(item.get("mtime"))
        score = _combined_score(
            bm25=sparse_score,
            dense=None,
            rerank=None,
            bm_stats=score_stats,
            dense_stats=None,
            rerank_stats=None,
            decay=decay,
        )
        prior_mult = _prior_multiplier(item.get("doc_id"), subjects, boosts)
        score *= prior_mult
        rows.append({
            "score": score,
            "final_score": score,
            "bm25_score": sparse_score,
            "decay_factor": decay,
            "id": item.get("chunk_id"),
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "path": item.get("path"),
            "chunk_start": item.get("chunk_start"),
            "chunk_end": item.get("chunk_end"),
            "text": item.get("text"),
            "section_path": item.get("section_path"),
            "element_ids": item.get("element_ids"),
            "bboxes": item.get("bboxes"),
            "types": item.get("types"),
            "source_tools": item.get("source_tools"),
            "table_headers": item.get("table_headers"),
            "table_units": item.get("table_units"),
            "chunk_profile": item.get("chunk_profile"),
            "plan_hash": item.get("plan_hash"),
            "model_version": item.get("model_version"),
            "prompt_sha": item.get("prompt_sha"),
            "doc_metadata": item.get("doc_metadata"),
            "scores": _score_breakdown(
                bm25=sparse_score,
                dense=None,
                rrf=None,
                rerank=None,
                decay=decay,
                final=score,
            ),
        })
        _set_page_fields(rows[-1], item)
        _ensure_score_bucket(rows[-1])["prior"] = prior_mult
    await _ensure_row_texts(rows, collection, subjects)
    _annotate_rows(rows, query)
    return rows


async def _run_rerank(
    collection: str,
    query: str,
    query_vec: List[float],
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    try:
        start = time.perf_counter()
        hits = await asyncio.to_thread(
            qdr.search,
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        timings["qdrant_ms"] = timings.get("qdrant_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return [{"error": "qdrant_search_failed", "detail": str(exc)}]

    await _hydrate_qdrant_hits(hits, collection)
    texts = [str((h.payload or {}).get("text") or "")[:RERANK_MAX_CHARS] for h in hits]
    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=90.0) as client:
            rr = await client.post(
                f"{TEI_RERANK_URL}/rerank",
                json={"query": query, "texts": texts, "raw_scores": False},
            )
            rr.raise_for_status()
            data = rr.json()
        timings["rerank_ms"] = timings.get("rerank_ms", 0.0) + (time.perf_counter() - start) * 1000.0
        if isinstance(data, list):
            order = sorted(data, key=lambda r: r.get("score", 0.0), reverse=True)
        else:
            results = data.get("results", [])
            order = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        rerank_scores = {int(o.get("index", 0)): float(o.get("score", 0.0) or 0.0) for o in order}
    except Exception as exc:
        logger.warning("Rerank failed, falling back to semantic results: %s", exc)
        fallback_rows: List[Dict[str, Any]] = []
        for h in hits[: top_k]:
            pl = h.payload or {}
            dense_val = getattr(h, "score", 0.0)
            decay = _decay_factor(pl.get("mtime"))
            fallback_rows.append({
                "score": dense_val,
                "final_score": dense_val,
                "dense_score": dense_val,
                "decay_factor": decay,
                "id": str(getattr(h, "id", "")),
                "chunk_id": str(getattr(h, "id", "")),
                "doc_id": pl.get("doc_id"),
                "path": pl.get("path"),
                "chunk_start": pl.get("chunk_start"),
                "chunk_end": pl.get("chunk_end"),
                "text": pl.get("text"),
                "section_path": pl.get("section_path"),
                "element_ids": pl.get("element_ids"),
                "bboxes": pl.get("bboxes"),
                "types": pl.get("types"),
                "source_tools": pl.get("source_tools"),
                "table_headers": pl.get("table_headers"),
                "table_units": pl.get("table_units"),
                "chunk_profile": pl.get("chunk_profile"),
                "plan_hash": pl.get("plan_hash"),
                "model_version": pl.get("model_version"),
                "prompt_sha": pl.get("prompt_sha"),
                "doc_metadata": pl.get("doc_metadata"),
                "scores": _score_breakdown(
                    bm25=None,
                    dense=dense_val,
                    rrf=None,
                    rerank=None,
                    decay=decay,
                    final=dense_val,
                ),
            })
            _set_page_fields(fallback_rows[-1], pl)
        await _ensure_row_texts(fallback_rows, collection, subjects)
        _annotate_rows(fallback_rows, query)
        return fallback_rows

    dense_values = [getattr(h, "score", 0.0) for h in hits]
    dense_stats = (min(dense_values), max(dense_values)) if dense_values else None
    rerank_values = list(rerank_scores.values())
    rerank_stats = (min(rerank_values), max(rerank_values)) if rerank_values else None
    scored = []
    decay_map: Dict[int, float] = {}
    prior_map: Dict[int, float] = {}
    for o in order:
        idx = o.get("index", 0)
        base = float(o.get("score", 0.0) or 0.0)
        payload = hits[idx].payload or {}
        decay = _decay_factor(payload.get("mtime"))
        final = _combined_score(
            bm25=None,
            dense=getattr(hits[idx], "score", 0.0),
            rerank=base,
            bm_stats=None,
            dense_stats=dense_stats,
            rerank_stats=rerank_stats,
            decay=decay,
        )
        prior_mult = _prior_multiplier(payload.get("doc_id"), subjects, boosts)
        final *= prior_mult
        scored.append((final, idx))
        decay_map[int(idx)] = decay
        prior_map[int(idx)] = prior_mult
    scored.sort(key=lambda t: t[0], reverse=True)
    final_scores = {idx: score for score, idx in scored}

    rows: List[Dict[str, Any]] = []
    for _, idx in scored[:return_k]:
        hit = hits[idx]
        payload = hit.payload or {}
        dense_val = getattr(hit, "score", 0.0)
        decay = decay_map.get(int(idx))
        prior_mult = prior_map.get(int(idx), 1.0)
        row = {
            "score": final_scores.get(idx, 0.0),
            "final_score": final_scores.get(idx, 0.0),
            "rerank_score": rerank_scores.get(idx, 0.0),
            "dense_score": dense_val,
            "decay_factor": decay,
            "id": str(getattr(hit, "id", "")),
            "chunk_id": str(getattr(hit, "id", "")),
            "doc_id": payload.get("doc_id"),
            "path": payload.get("path"),
            "chunk_start": payload.get("chunk_start"),
            "chunk_end": payload.get("chunk_end"),
            "text": payload.get("text"),
            "section_path": payload.get("section_path"),
            "element_ids": payload.get("element_ids"),
            "bboxes": payload.get("bboxes"),
            "types": payload.get("types"),
            "source_tools": payload.get("source_tools"),
            "table_headers": payload.get("table_headers"),
            "table_units": payload.get("table_units"),
            "chunk_profile": payload.get("chunk_profile"),
            "plan_hash": payload.get("plan_hash"),
            "model_version": payload.get("model_version"),
            "prompt_sha": payload.get("prompt_sha"),
            "doc_metadata": payload.get("doc_metadata"),
            "scores": _score_breakdown(
                bm25=None,
                dense=dense_val,
                rrf=None,
                rerank=rerank_scores.get(idx, 0.0),
                decay=decay,
                final=final_scores.get(idx, 0.0),
            ),
        }
        _set_page_fields(row, payload)
        _ensure_score_bucket(row)["prior"] = prior_mult
        if NEIGHBOR_CHUNKS > 0 and row["doc_id"] and row["chunk_start"] is not None:
            fts_db_path = _get_fts_db_path(collection)
            neigh = _fts_neighbors(row["doc_id"], int(row["chunk_start"]), NEIGHBOR_CHUNKS, fts_db_path=fts_db_path)
            if neigh:
                ordered = sorted(neigh, key=lambda r: int(r.get("chunk_start", 0) or 0))
                txt = "\n".join([str(r.get("text") or "") for r in ordered]).strip()
                if txt:
                    row["text"] = txt
                row["chunk_start"] = int(ordered[0].get("chunk_start", row["chunk_start"]))
                row["chunk_end"] = int(ordered[-1].get("chunk_end", row["chunk_end"]))
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    _annotate_rows(rows, query)
    return rows


async def _run_hybrid(
    collection: str,
    query: str,
    query_vec: List[float],
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    try:
        start = time.perf_counter()
        dense_hits = await asyncio.to_thread(
            qdr.search,
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        timings["qdrant_ms"] = timings.get("qdrant_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return [{"error": "qdrant_search_failed", "detail": str(exc)}]

    await _hydrate_qdrant_hits(dense_hits, collection)
    fts_db_path = _get_fts_db_path(collection)
    start = time.perf_counter()
    lexical_hits = await asyncio.to_thread(_fts_search, query, limit, fts_db_path)
    timings["fts_ms"] = timings.get("fts_ms", 0.0) + (time.perf_counter() - start) * 1000.0

    dense_list: List[Dict[str, Any]] = []
    for h in dense_hits:
        payload = h.payload or {}
        prior_mult = _prior_multiplier(payload.get("doc_id"), subjects, boosts)
        dense_list.append({
            "chunk_id": str(getattr(h, "id", "")),
            "doc_id": payload.get("doc_id"),
            "path": payload.get("path"),
            "filename": payload.get("filename"),
            "chunk_start": payload.get("chunk_start"),
            "chunk_end": payload.get("chunk_end"),
            "mtime": payload.get("mtime"),
            "text": payload.get("text"),
            "dense_score": getattr(h, "score", None),
            "section_path": payload.get("section_path"),
            "element_ids": payload.get("element_ids"),
            "bboxes": payload.get("bboxes"),
            "types": payload.get("types"),
            "source_tools": payload.get("source_tools"),
            "table_headers": payload.get("table_headers"),
            "table_units": payload.get("table_units"),
            "_prior_mult": prior_mult,
        })
        _set_page_fields(dense_list[-1], payload)

    fts_list: List[Dict[str, Any]] = list(lexical_hits)
    fused = _rrf_fuse([dense_list, fts_list], id_getter=lambda x: x.get("chunk_id"))
    fused = fused[:limit]

    texts = [str(x.get("text") or "")[:RERANK_MAX_CHARS] for x in fused]
    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=90.0) as client:
            rr = await client.post(
                f"{TEI_RERANK_URL}/rerank",
                json={"query": query, "texts": texts, "raw_scores": False},
            )
            rr.raise_for_status()
            data = rr.json()
        timings["rerank_ms"] = timings.get("rerank_ms", 0.0) + (time.perf_counter() - start) * 1000.0
        if isinstance(data, list):
            order = sorted(data, key=lambda r: r.get("score", 0.0), reverse=True)
        else:
            results = data.get("results", [])
            order = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        rerank_scores = {int(o.get("index", 0)): float(o.get("score", 0.0) or 0.0) for o in order}
    except Exception as exc:
        logger.warning("Rerank failed, falling back to fused order: %s", exc)
        fallback_rows: List[Dict[str, Any]] = []
        for x in fused[: top_k]:
            decay = _decay_factor(x.get("mtime"))
            prior_mult = x.get("_prior_mult")
            if prior_mult is None:
                prior_mult = _prior_multiplier(x.get("doc_id"), subjects, boosts)
            base_rrf = x.get("_rrf_score") or 0.0
            final = base_rrf * prior_mult
            fallback_rows.append({
                "score": final,
                "final_score": final,
                "rrf_score": x.get("_rrf_score"),
                "bm25_score": x.get("bm25"),
                "dense_score": x.get("dense_score"),
                "decay_factor": decay,
                "id": x.get("chunk_id"),
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "path": x.get("path"),
                "chunk_start": x.get("chunk_start"),
                "chunk_end": x.get("chunk_end"),
                "text": x.get("text"),
                "section_path": x.get("section_path"),
                "element_ids": x.get("element_ids"),
                "bboxes": x.get("bboxes"),
                "types": x.get("types"),
                "source_tools": x.get("source_tools"),
                "table_headers": x.get("table_headers"),
                "table_units": x.get("table_units"),
                "chunk_profile": x.get("chunk_profile"),
                "plan_hash": x.get("plan_hash"),
                "model_version": x.get("model_version"),
                "prompt_sha": x.get("prompt_sha"),
                "doc_metadata": x.get("doc_metadata"),
                "scores": _score_breakdown(
                    bm25=x.get("bm25"),
                    dense=x.get("dense_score"),
                    rrf=x.get("_rrf_score"),
                    rerank=None,
                    decay=decay,
                    final=final,
                ),
            })
            _set_page_fields(fallback_rows[-1], x)
            _ensure_score_bucket(fallback_rows[-1])["prior"] = prior_mult
        await _ensure_row_texts(fallback_rows, collection, subjects)
        _annotate_rows(fallback_rows, query)
        return fallback_rows

    bm_values = [x.get("bm25") for x in fused if x.get("bm25") is not None]
    bm_stats = (min(bm_values), max(bm_values)) if bm_values else None
    dense_values = [x.get("dense_score") for x in fused if x.get("dense_score") is not None]
    dense_stats = (min(dense_values), max(dense_values)) if dense_values else None
    rerank_values = list(rerank_scores.values())
    rerank_stats = (min(rerank_values), max(rerank_values)) if rerank_values else None
    scored = []
    decay_map: Dict[int, float] = {}
    prior_map: Dict[int, float] = {}
    for o in order:
        idx = o.get("index", 0)
        base = float(o.get("score", 0.0) or 0.0)
        x = fused[idx]
        decay = _decay_factor(x.get("mtime"))
        final = _combined_score(
            bm25=x.get("bm25"),
            dense=x.get("dense_score"),
            rerank=base,
            bm_stats=bm_stats,
            dense_stats=dense_stats,
            rerank_stats=rerank_stats,
            decay=decay,
        )
        prior_mult = x.get("_prior_mult")
        if prior_mult is None:
            prior_mult = _prior_multiplier(x.get("doc_id"), subjects, boosts)
        final *= prior_mult
        scored.append((final, idx))
        decay_map[int(idx)] = decay
        prior_map[int(idx)] = prior_mult
    scored.sort(key=lambda t: t[0], reverse=True)
    final_scores = {idx: score for score, idx in scored}

    rows: List[Dict[str, Any]] = []
    for _, idx in scored[:return_k]:
        x = fused[idx]
        decay = decay_map.get(int(idx))
        prior_mult = prior_map.get(int(idx), 1.0)
        row = {
            "score": final_scores.get(idx, 0.0),
            "final_score": final_scores.get(idx, 0.0),
            "rerank_score": rerank_scores.get(idx, 0.0),
            "rrf_score": x.get("_rrf_score"),
            "bm25_score": x.get("bm25"),
            "dense_score": x.get("dense_score"),
            "decay_factor": decay,
            "id": x.get("chunk_id"),
            "chunk_id": x.get("chunk_id"),
            "doc_id": x.get("doc_id"),
            "path": x.get("path"),
            "chunk_start": x.get("chunk_start"),
            "chunk_end": x.get("chunk_end"),
            "text": x.get("text"),
            "section_path": x.get("section_path"),
            "element_ids": x.get("element_ids"),
            "bboxes": x.get("bboxes"),
            "types": x.get("types"),
            "source_tools": x.get("source_tools"),
            "table_headers": x.get("table_headers"),
            "table_units": x.get("table_units"),
            "chunk_profile": x.get("chunk_profile"),
            "plan_hash": x.get("plan_hash"),
            "model_version": x.get("model_version"),
            "prompt_sha": x.get("prompt_sha"),
            "doc_metadata": x.get("doc_metadata"),
            "scores": _score_breakdown(
                bm25=x.get("bm25"),
                dense=x.get("dense_score"),
                rrf=x.get("_rrf_score"),
                rerank=rerank_scores.get(idx, 0.0),
                decay=decay,
                final=final_scores.get(idx, 0.0),
            ),
        }
        _set_page_fields(row, x)
        _ensure_score_bucket(row)["prior"] = prior_mult
        if NEIGHBOR_CHUNKS > 0 and row["doc_id"] and row["chunk_start"] is not None:
            fts_db_path = _get_fts_db_path(collection)
            neigh = _fts_neighbors(row["doc_id"], int(row["chunk_start"]), NEIGHBOR_CHUNKS, fts_db_path=fts_db_path)
            if neigh:
                ordered = sorted(neigh, key=lambda r: int(r.get("chunk_start", 0) or 0))
                txt = "\n".join([str(r.get("text") or "") for r in ordered]).strip()
                if txt:
                    row["text"] = txt
                row["chunk_start"] = int(ordered[0].get("chunk_start", row["chunk_start"]))
                row["chunk_end"] = int(ordered[-1].get("chunk_end", row["chunk_end"]))
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    _annotate_rows(rows, query)
    return rows


async def _execute_search(
    route: str,
    collection: str,
    query: str,
    query_vec: List[float],
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
    boosts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if route == "sparse":
        return await _run_sparse(collection, query, retrieve_k, return_k, top_k, subjects, timings, boosts)
    if route == "sparse_splade":
        return await _run_sparse_splade(collection, query, retrieve_k, return_k, top_k, subjects, timings, boosts)
    if route == "semantic":
        return await _run_semantic(collection, query, query_vec, top_k, return_k, subjects, timings, boosts)
    if route == "rerank":
        return await _run_rerank(collection, query, query_vec, retrieve_k, return_k, top_k, subjects, timings, boosts)
    if route == "hybrid":
        return await _run_hybrid(collection, query, query_vec, retrieve_k, return_k, top_k, subjects, timings, boosts)
    if route == "colbert":
        return await _run_colbert(collection, query, retrieve_k, return_k, top_k, subjects, timings, boosts)
    return [{"error": f"invalid route '{route}'"}]


# ---- Register search tools per scope ---------------------------------------
def _resolve_scope(collection: Optional[str]) -> Tuple[str, str, str]:
    if not SCOPES:
        raise ValueError("NOMIC_KB_SCOPES is empty; configure at least one collection")
    if collection:
        key = collection.strip()
        if not key:
            raise ValueError("collection argument was empty")
        if key in SCOPES:
            cfg = SCOPES[key]
            return key, cfg["collection"], cfg.get("title") or key
        lowered = key.lower()
        for slug, cfg in SCOPES.items():
            title = (cfg.get("title") or slug or "").lower()
            if cfg.get("collection") == key or title == lowered:
                return slug, cfg["collection"], cfg.get("title") or slug
        raise ValueError(f"Unknown collection '{collection}'. Known entries: {sorted(SCOPES)}")
    slug, cfg = next(iter(SCOPES.items()))
    return slug, cfg["collection"], cfg.get("title") or slug


def _get_fts_db_path(collection_name: str) -> str:
    """
    Resolve FTS database path for a given Qdrant collection name.

    First checks if an explicit fts_db path is configured in SCOPES,
    otherwise derives it from collection name: data/{collection_name}_fts.db
    Falls back to global FTS_DB_PATH if no collection name provided.
    """
    if not collection_name:
        return FTS_DB_PATH

    # Check if any scope explicitly configures fts_db for this collection
    for slug, cfg in SCOPES.items():
        if cfg.get("collection") == collection_name:
            if "fts_db" in cfg:
                return cfg["fts_db"]
            # Convention: data/{collection_name}_fts.db
            return f"data/{collection_name}_fts.db"

    # Fallback: derive from collection name
    return f"data/{collection_name}_fts.db"


# ---- Unified retrieval tools (collection parameter required) --------------

@mcp.tool(name="kb.sparse", title="KB: Sparse Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_sparse(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """Full-text search using FTS5 BM25 ranking.

    Performs lexical search across the knowledge base using SQLite FTS5 with
    BM25 scoring. Best for exact term matching, acronyms, and technical terms.

    Args:
        ctx: MCP context with session metadata
        query: Search query text (will be tokenized)
        collection: Target collection slug or name (uses default if None)
        retrieve_k: Number of candidates to retrieve (1-256, default 24)
        return_k: Number of results to return (1-256, default 8)
        scope: Optional dict with doc_id boosts {"boosts": {"doc_id": weight}}
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with:
            - rows: List of matching chunks with scores
            - count: Number of results returned
            - retrieve_k/return_k: Pagination parameters used
            - has_more: Whether more results exist beyond return_k
            - best_score: Highest BM25 score in results
            - timings: Performance metrics (fts_ms)
            - abstain: True if best_score below answerability threshold
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    slug, collection_name, _ = _resolve_scope(collection)
    retrieve_k = max(1, min(int(retrieve_k), 256))
    return_k = max(1, min(int(return_k), retrieve_k))
    subjects = get_subjects_from_context(ctx)
    timings: Dict[str, float] = {}
    boosts = _parse_scope_boosts(scope)
    rows = await _run_sparse(
        collection_name,
        query,
        retrieve_k,
        return_k,
        retrieve_k,
        subjects,
        timings,
        boosts,
    )
    best = _best_score(rows)
    profile = _normalize_response_profile(response_profile)
    pruned_rows = [prune_row(row, profile) for row in rows]
    result = {
        "collection": collection_name,
        "slug": slug,
        "rows": pruned_rows,
        "count": len(pruned_rows),
        "retrieve_k": retrieve_k,
        "return_k": return_k,
        "has_more": len(rows) >= return_k,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "route": "sparse",
        "best_score": best,
    }
    if boosts:
        result["boosts"] = boosts
    if best < ANSWERABILITY_THRESHOLD:
        result.update({
            "abstain": True,
            "reason": "best_score_below_threshold",
            "threshold": ANSWERABILITY_THRESHOLD,
        })
    return result

@mcp.tool(name="kb.sparse_splade", title="KB: SPLADE Sparse Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_sparse_splade(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """SPLADE-enhanced sparse search with learned term expansion.

    Uses SPLADE neural model to expand query terms before FTS5 search,
    improving recall for semantic variants. Requires SPLADE service running.

    Args:
        ctx: MCP context with session metadata
        query: Search query text
        collection: Target collection slug or name (uses default if None)
        retrieve_k: Number of candidates to retrieve (1-256, default 24)
        return_k: Number of results to return (1-256, default 8)
        scope: Optional dict with doc_id boosts {"boosts": {"doc_id": weight}}
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with rows, count, pagination metadata, best_score, timings
        Error dict if SPLADE service unavailable or query empty
    """
    if not HAVE_SPARSE_SPLADE:
        return {"error": "sparse_expander_disabled"}
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    slug, collection_name, _ = _resolve_scope(collection)
    retrieve_k = max(1, min(int(retrieve_k), 256))
    return_k = max(1, min(int(return_k), retrieve_k))
    subjects = get_subjects_from_context(ctx)
    timings: Dict[str, float] = {}
    boosts = _parse_scope_boosts(scope)
    rows = await _run_sparse_splade(
        collection_name,
        query,
        retrieve_k,
        return_k,
        retrieve_k,
        subjects,
        timings,
        boosts,
    )
    best = _best_score(rows)
    profile = _normalize_response_profile(response_profile)
    pruned_rows = [prune_row(row, profile) for row in rows]
    result = {
        "collection": collection_name,
        "slug": slug,
        "rows": pruned_rows,
        "count": len(pruned_rows),
        "retrieve_k": retrieve_k,
        "return_k": return_k,
        "has_more": len(rows) >= return_k,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "route": "sparse_splade",
        "best_score": best,
    }
    if boosts:
        result["boosts"] = boosts
    if best < ANSWERABILITY_THRESHOLD:
        result.update({
            "abstain": True,
            "reason": "best_score_below_threshold",
            "threshold": ANSWERABILITY_THRESHOLD,
        })
    return result

@mcp.tool(name="kb.dense", title="KB: Dense Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_dense(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """Pure vector similarity search using embeddings.

    Performs semantic search by embedding the query and finding nearest
    neighbors in vector space. Best for conceptual/semantic matching.

    Args:
        ctx: MCP context with session metadata
        query: Search query text (will be embedded)
        collection: Target collection slug or name (uses default if None)
        retrieve_k: Number of candidates to retrieve (1-256, default 24)
        return_k: Number of results to return (1-256, default 8)
        scope: Optional dict with doc_id boosts {"boosts": {"doc_id": weight}}
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with rows, count, pagination metadata, best_score, timings
        Includes embed_ms timing for embedding latency
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    slug, collection_name, _ = _resolve_scope(collection)
    retrieve_k = max(1, min(int(retrieve_k), 256))
    return_k = max(1, min(int(return_k), retrieve_k))
    subjects = get_subjects_from_context(ctx)
    timings: Dict[str, float] = {}
    query_vec = await embed_query(query)
    boosts = _parse_scope_boosts(scope)
    rows = await _run_semantic(
        collection_name,
        query,
        query_vec,
        retrieve_k,
        return_k,
        subjects,
        timings,
        boosts,
    )
    best = _best_score(rows)
    profile = _normalize_response_profile(response_profile)
    pruned_rows = [prune_row(row, profile) for row in rows]
    result = {
        "collection": collection_name,
        "slug": slug,
        "rows": pruned_rows,
        "count": len(pruned_rows),
        "retrieve_k": retrieve_k,
        "return_k": return_k,
        "has_more": len(rows) >= return_k,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "route": "semantic",
        "best_score": best,
    }
    if boosts:
        result["boosts"] = boosts
    if best < ANSWERABILITY_THRESHOLD:
        result.update({
            "abstain": True,
            "reason": "best_score_below_threshold",
            "threshold": ANSWERABILITY_THRESHOLD,
        })
    return result

@mcp.tool(name="kb.hybrid", title="KB: Hybrid Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_hybrid(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    top_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """Hybrid search combining dense vectors and sparse FTS5.

    Retrieves candidates using both semantic embeddings and BM25 lexical
    matching, then merges results using reciprocal rank fusion (RRF).

    Args:
        ctx: MCP context with session metadata
        query: Search query text
        collection: Target collection slug or name (uses default if None)
        retrieve_k: Candidates per route before merge (1-256, default 24)
        return_k: Final results to return (1-256, default 8)
        top_k: Top-k for RRF merge stage (default 8)
        scope: Optional dict with doc_id boosts {"boosts": {"doc_id": weight}}
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with rows, count, pagination metadata, best_score, timings
        Timings include embed_ms, vector_ms, fts_ms, merge_ms
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    slug, collection_name, _ = _resolve_scope(collection)
    retrieve_k = max(1, min(int(retrieve_k), 256))
    return_k = max(1, min(int(return_k), retrieve_k))
    top_k = max(return_k, min(int(top_k), retrieve_k))
    subjects = get_subjects_from_context(ctx)
    timings: Dict[str, float] = {}
    query_vec = await embed_query(query)
    boosts = _parse_scope_boosts(scope)
    rows = await _run_hybrid(
        collection_name,
        query,
        query_vec,
        retrieve_k,
        return_k,
        top_k,
        subjects,
        timings,
        boosts,
    )
    best = _best_score(rows)
    profile = _normalize_response_profile(response_profile)
    pruned_rows = [prune_row(row, profile) for row in rows]
    result = {
        "collection": collection_name,
        "slug": slug,
        "rows": pruned_rows,
        "count": len(pruned_rows),
        "retrieve_k": retrieve_k,
        "return_k": return_k,
        "top_k": top_k,
        "has_more": len(rows) >= return_k,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "route": "hybrid",
        "best_score": best,
    }
    if boosts:
        result["boosts"] = boosts
    if best < ANSWERABILITY_THRESHOLD:
        result.update({
            "abstain": True,
            "reason": "best_score_below_threshold",
            "threshold": ANSWERABILITY_THRESHOLD,
        })
    return result

@mcp.tool(name="kb.rerank", title="KB: Rerank Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_rerank(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    top_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """Two-stage search: vector retrieval then cross-encoder reranking.

    First retrieves candidates via dense vector search, then applies a
    cross-encoder reranker (TEI) for precise relevance scoring. Recommended
    default for most queries - balances recall and precision.

    Args:
        ctx: MCP context with session metadata
        query: Search query text
        collection: Target collection slug or name (uses default if None)
        retrieve_k: Dense retrieval candidates (1-128, default 24)
        return_k: Final results after reranking (1-256, default 8)
        top_k: Candidates sent to reranker (default 8)
        scope: Optional dict with doc_id boosts {"boosts": {"doc_id": weight}}
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with rows, count, pagination metadata, best_score, timings
        Timings include embed_ms, vector_ms, rerank_ms
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    slug, collection_name, _ = _resolve_scope(collection)
    retrieve_k = max(1, min(int(retrieve_k), RERANK_MAX_ITEMS))
    return_k = max(1, min(int(return_k), retrieve_k))
    top_k = max(return_k, min(int(top_k), retrieve_k))
    subjects = get_subjects_from_context(ctx)
    timings: Dict[str, float] = {}
    query_vec = await embed_query(query)
    boosts = _parse_scope_boosts(scope)
    rows = await _run_rerank(
        collection_name,
        query,
        query_vec,
        retrieve_k,
        return_k,
        top_k,
        subjects,
        timings,
        boosts,
    )
    best = _best_score(rows)
    profile = _normalize_response_profile(response_profile)
    rows = [prune_row(row, profile) for row in rows]
    result = {
        "collection": collection_name,
        "slug": slug,
        "rows": rows,
        "count": len(rows),
        "retrieve_k": retrieve_k,
        "return_k": return_k,
        "top_k": top_k,
        "has_more": len(rows) >= return_k,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "route": "rerank",
        "best_score": best,
    }
    if boosts:
        result["boosts"] = boosts
    if best < ANSWERABILITY_THRESHOLD:
        result.update({
            "abstain": True,
            "reason": "best_score_below_threshold",
            "threshold": ANSWERABILITY_THRESHOLD,
        })
    return result

@mcp.tool(name="kb.colbert", title="KB: ColBERT Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_colbert(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """Late-interaction search using ColBERT token-level matching.

    Uses ColBERT's MaxSim scoring for fine-grained token-level relevance.
    Requires ColBERT service (RAGatouille) running. Best for complex queries
    where token-level matching matters.

    Args:
        ctx: MCP context with session metadata
        query: Search query text
        collection: Target collection slug or name (uses default if None)
        retrieve_k: Number of candidates to retrieve (1-256, default 24)
        return_k: Number of results to return (1-256, default 8)
        scope: Optional dict with doc_id boosts {"boosts": {"doc_id": weight}}
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with rows, count, pagination metadata, best_score, timings
        Error dict if ColBERT service unavailable or query empty
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    if not HAVE_COLBERT:
        return {"error": "colbert_service_unavailable"}
    slug, collection_name, _ = _resolve_scope(collection)
    retrieve_k = max(1, min(int(retrieve_k), 256))
    return_k = max(1, min(int(return_k), retrieve_k))
    subjects = get_subjects_from_context(ctx)
    timings: Dict[str, float] = {}
    boosts = _parse_scope_boosts(scope)
    rows = await _run_colbert(
        collection_name,
        query,
        retrieve_k,
        return_k,
        retrieve_k,
        subjects,
        timings,
        boosts,
    )
    best = _best_score(rows)
    profile = _normalize_response_profile(response_profile)
    rows = [prune_row(row, profile) for row in rows]
    result = {
        "collection": collection_name,
        "slug": slug,
        "rows": rows,
        "count": len(rows),
        "retrieve_k": retrieve_k,
        "return_k": return_k,
        "has_more": len(rows) >= return_k,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "route": "colbert",
        "best_score": best,
    }
    if boosts:
        result["boosts"] = boosts
    if best < ANSWERABILITY_THRESHOLD:
        result.update({
            "abstain": True,
            "reason": "best_score_below_threshold",
            "threshold": ANSWERABILITY_THRESHOLD,
        })
    return result

@mcp.tool(name="kb.batch", title="KB: Batch Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_batch(
    ctx: Context,
    queries: List[str],
    routes: Optional[List[str]] = None,
    collection: Optional[str] = None,
    collections: Optional[List[str]] = None,
    retrieve_k: int = 24,
    return_k: int = 8,
    scope: Optional[Dict[str, Any]] = None,
    scopes: Optional[List[Dict[str, Any]]] = None,
    response_profile: str = "slim",
) -> Dict[str, Any]:
    """Execute multiple search queries in a single call.

    Processes multiple queries sequentially with per-query route and
    collection overrides. Useful for query rephrasing or multi-aspect search.

    Args:
        ctx: MCP context with session metadata
        queries: List of search query strings
        routes: Optional per-query routes (sparse/semantic/rerank/hybrid/colbert/auto)
        collection: Default collection for all queries
        collections: Optional per-query collection overrides
        retrieve_k: Candidates per query (1-256, default 24)
        return_k: Results per query (1-256, default 8)
        scope: Default scope/boosts for all queries
        scopes: Optional per-query scope overrides
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        Dict with results list, each containing:
            - index, query, route, collection, rows, count, best_score, timings
            - has_more, retrieve_k, return_k pagination metadata
    """
    if not isinstance(queries, list) or not queries:
        return {"error": "missing_queries"}
    retrieve_k = max(1, min(int(retrieve_k), 256))
    return_k = max(1, min(int(return_k), retrieve_k))
    if routes and len(routes) != len(queries):
        return {"error": "routes_length_mismatch"}
    use_routes = routes or ["auto"] * len(queries)
    subjects = get_subjects_from_context(ctx)
    results: List[Dict[str, Any]] = []
    for idx, (query, route) in enumerate(zip(queries, use_routes)):
        q = (query or "").strip()
        if not q:
            results.append({"index": idx, "query": query, "route": route, "error": "empty_query"})
            continue
        route_norm = route.lower()
        valid_routes = {"sparse", "sparse_splade", "semantic", "rerank", "hybrid", "colbert", "auto"}
        if route_norm not in valid_routes:
            results.append({"index": idx, "query": query, "route": route, "error": "invalid_route"})
            continue
        coll_name = collections[idx] if collections and idx < len(collections) else collection
        try:
            slug, collection_name, _ = _resolve_scope(coll_name)
        except Exception as exc:
            results.append({"index": idx, "query": query, "route": route_norm, "error": str(exc)})
            continue
        timings: Dict[str, float] = {}
        boosts = None
        if scopes and idx < len(scopes):
            boosts = _parse_scope_boosts(scopes[idx])
        elif scope:
            boosts = _parse_scope_boosts(scope)
        try:
            route_exec = route_norm
            route_retrieve = retrieve_k
            if route_norm == "auto":
                try:
                    planned = await plan_route(q)
                except Exception as exc:
                    logger.debug("plan_route failed in batch: %s", exc)
                    planned = {}
                route_exec = str((planned or {}).get("route") or "rerank")
                if route_exec not in valid_routes - {"auto"}:
                    route_exec = "rerank"
                planned_k = (planned or {}).get("k")
                if isinstance(planned_k, int) and 1 <= planned_k <= 256:
                    route_retrieve = planned_k
            route_retrieve = max(return_k, min(route_retrieve, 256))
            query_vec: List[float] = []
            if route_exec in {"semantic", "rerank", "hybrid"}:
                query_vec = await embed_query(q)
            rows = await _execute_search(
                route_exec,
                collection_name,
                q,
                query_vec,
                route_retrieve,
                return_k,
                max(return_k, route_retrieve),
                subjects,
                timings,
                boosts,
            )
            best = _best_score(rows)
            profile = _normalize_response_profile(response_profile)
            rows = [prune_row(row, profile) for row in rows]
            entry = {
                "index": idx,
                "query": q,
                "requested_route": route_norm,
                "route": route_exec,
                "collection": collection_name,
                "slug": slug,
                "rows": rows,
                "count": len(rows),
                "retrieve_k": route_retrieve,
                "return_k": return_k,
                "has_more": len(rows) >= return_k,
                "timings": {k: round(v, 3) for k, v in timings.items()},
                "best_score": best,
            }
            if boosts:
                entry["boosts"] = boosts
            if best < ANSWERABILITY_THRESHOLD:
                entry.update({
                    "abstain": True,
                    "reason": "best_score_below_threshold",
                    "threshold": ANSWERABILITY_THRESHOLD,
                })
            results.append(entry)
        except Exception as exc:
            results.append({"index": idx, "query": q, "route": route_norm, "error": str(exc)})
    return {"results": results}

@mcp.tool(name="kb.quality", title="KB: Inspect Hits", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_quality(
    ctx: Context,
    hits: List[Dict[str, Any]],
    collection: Optional[str] = None,
    rules: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    rules = rules or {}
    slug, collection_name, _ = _resolve_scope(collection)
    min_score = float(rules.get("min_score", ANSWERABILITY_THRESHOLD))
    require_metadata = bool(rules.get("require_metadata", False))
    require_plan_hash = bool(rules.get("require_plan_hash", False))
    require_table = bool(rules.get("require_table_hit", False))
    valid_hits = [hit for hit in hits if isinstance(hit, dict)] if hits else []
    query_tokens = _tokenize_quality(query)
    score_values: List[float] = []
    coverage_values: List[float] = []
    doc_counts: Dict[str, int] = {}
    table_hits = 0
    failures: List[str] = []
    for idx, hit in enumerate(valid_hits):
        score = hit.get("final_score") if hit.get("final_score") is not None else hit.get("score")
        try:
            score_val = float(score)
        except Exception:
            score_val = 0.0
        if score_val < min_score:
            failures.append(f"hit_{idx}_score_below_min")
        score_values.append(score_val)
        if require_metadata and not hit.get("doc_metadata"):
            failures.append(f"hit_{idx}_missing_metadata")
        if require_plan_hash and not hit.get("plan_hash"):
            failures.append(f"hit_{idx}_missing_plan_hash")
        if query_tokens:
            coverage = _coverage_ratio(query_tokens, hit.get("text"))
            coverage_values.append(coverage)
        doc_id = str(hit.get("doc_id") or "")
        if doc_id:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        if _is_table_row(hit.get("types")):
            table_hits += 1
    if require_table and table_hits == 0:
        failures.append("missing_table_hit")
    warnings: List[str] = []
    if query_tokens and coverage_values and max(coverage_values) < 0.3:
        warnings.append("low_query_coverage")
    duplicate_docs = [doc for doc, count in doc_counts.items() if count > 1]
    if duplicate_docs:
        warnings.append("duplicate_doc_hits")
    analysis = {
        "score_summary": _summarize(score_values),
        "coverage_summary": _summarize(coverage_values) if coverage_values else None,
        "query_tokens": len(query_tokens),
        "table_hits": table_hits,
        "doc_counts": sorted(
            [{"doc_id": doc, "hits": count} for doc, count in doc_counts.items()],
            key=lambda x: x["hits"],
            reverse=True,
        ),
        "warnings": warnings,
    }
    return {
        "collection": collection_name,
        "slug": slug,
        "pass": not failures,
        "failures": failures,
        "evaluated": len(valid_hits),
        "rules": {
            "min_score": min_score,
            "require_metadata": require_metadata,
            "require_plan_hash": require_plan_hash,
            "require_table_hit": require_table,
        },
        "analysis": analysis,
    }

@mcp.tool(name="kb.hint", title="KB: Sparse Hints", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_hint(
    ctx: Context,
    term: Optional[str] = None,
    terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    seeds: List[str] = []
    if term:
        seeds.append(str(term))
    if isinstance(terms, list):
        seeds.extend(str(t) for t in terms if t)
    seeds = [s for s in map(str.strip, seeds) if s]
    if not seeds:
        return {"error": "missing_terms"}
    expansions: Dict[str, List[str]] = {}
    for seed in seeds:
        upper = seed.upper()
        matches: List[str] = []
        alias_list = ALIASES.get(upper)
        if alias_list:
            matches.extend(alias_list)
        for key, synonyms in ALIASES.items():
            bucket = [key] + synonyms
            if any(seed.lower() == entry.lower() for entry in bucket):
                matches.extend(bucket)
        normalized: List[str] = []
        for candidate in matches:
            if candidate.lower() != seed.lower() and candidate not in normalized:
                normalized.append(candidate)
        expansions[seed] = normalized
    _audit("hint", {
        "terms": seeds,
        "subjects": _audit_subjects(get_subjects_from_context(ctx)),
    })
    return {
        "terms": seeds,
        "expansions": expansions,
        "alias_count": len(ALIASES),
    }

def _expand_table_query(query: str) -> str:
    """
    Expand table query from AND (all terms) to OR (any term) for better recall.
    Falls back to original if expansion returns nothing.
    """
    # Tokenize and remove stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being"}
    tokens = re.findall(r'\w+', query.lower())
    meaningful_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    if not meaningful_tokens:
        return query  # No meaningful tokens, use original

    # Build OR query
    expanded_query = " OR ".join(meaningful_tokens)
    return expanded_query


@mcp.tool(name="kb.table", title="KB: Table Lookup", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_table(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    limit: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search for table rows matching a query.

    Specialized search that filters to chunks with type='table_row'.
    Best for structured data extraction from tables in documents.

    Args:
        ctx: MCP context with session metadata
        query: Search terms (expanded to OR query internally)
        collection: Target collection (uses default if None)
        doc_id: Limit search to specific document
        limit: Max rows to return (1-50, default 10)
        where: Key-value filter dict - both key and value must appear in text

    Returns:
        Dict with rows (table_row chunks), collection, query, limit
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "empty_query"}
    slug, collection_name, _ = _resolve_scope(collection)
    fts_db_path = _get_fts_db_path(collection_name)
    limit = max(1, min(int(limit), 50))
    fetch_limit = max(limit * 3, limit)

    # Try expanded OR query first for better table row recall
    expanded_query = _expand_table_query(query)
    raw_hits = await asyncio.to_thread(_fts_search, expanded_query, fetch_limit, fts_db_path)

    # Fallback to original query if OR returned nothing
    if not raw_hits:
        raw_hits = await asyncio.to_thread(_fts_search, query, fetch_limit, fts_db_path)
    filtered: List[Dict[str, Any]] = []
    target_doc = str(doc_id) if doc_id else None
    where_map = where or {}
    for item in raw_hits:
        if target_doc and str(item.get("doc_id")) != target_doc:
            continue
        if not _is_table_row(item.get("types")):
            continue
        text = (item.get("text") or "").lower()
        ok = True
        for key, val in where_map.items():
            if key and str(key).lower() not in text:
                ok = False
                break
            if val and str(val).lower() not in text:
                ok = False
                break
        if not ok:
            continue
        filtered.append(item)
        if len(filtered) >= limit:
            break
    await _ensure_row_texts(filtered, collection_name, get_subjects_from_context(ctx))
    filtered = [row for row in filtered if _is_allowed_path(row.get("path"))]
    _audit("table_lookup", {
        "slug": slug,
        "collection": collection_name,
        "query": query,
        "doc_id": doc_id,
        "count": len(filtered),
        "subjects": _audit_subjects(get_subjects_from_context(ctx)),
    })
    return {
        "collection": collection_name,
        "slug": slug,
        "query": query,
        "doc_id": doc_id,
        "rows": filtered,
        "limit": limit,
    }

@mcp.tool(name="kb.open", title="KB: Open Chunk", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_open(
    ctx: Context,
    chunk_id: Optional[str] = None,
    element_id: Optional[str] = None,
    collection: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve a specific chunk by ID with optional text slicing.

    Fetches full chunk data including text, metadata, and document info.
    Can look up by chunk_id UUID or element_id from ingestion.

    Args:
        ctx: MCP context with session metadata
        chunk_id: Chunk UUID to retrieve (preferred)
        element_id: Alternative: element ID from ingestion
        collection: Target collection (uses default if None)
        start: Optional start offset for text slicing
        end: Optional end offset for text slicing

    Returns:
        Dict with chunk_id, text, doc_id, path, section_path, metadata
        Error dict if chunk not found or access denied
    """
    slug, collection_name, _ = _resolve_scope(collection)
    fts_db_path = _get_fts_db_path(collection_name)
    target = chunk_id or _lookup_chunk_by_element(element_id or "", fts_db_path=fts_db_path)
    if not target:
        return {"error": "missing_target", "detail": "Provide chunk_id or element_id"}
    row = {"id": target}
    subjects = get_subjects_from_context(ctx)
    await _ensure_row_texts([row], collection_name, subjects)
    if not row.get("text"):
        return {"error": "not_found"}
    if not _is_allowed_path(row.get("path")):
        _audit("open_denied", {
            "slug": slug,
            "collection": collection_name,
            "doc_id": row.get("doc_id"),
            "chunk_id": target,
            "reason": "path_not_allowed",
            "path": row.get("path"),
            "subjects": _audit_subjects(subjects),
        })
        return {"error": "forbidden", "detail": "path not allowed"}
    text = row.get("text", "")
    if start is not None or end is not None:
        s = max(0, int(start or 0))
        e = int(end) if end is not None else len(text)
        row["text"] = text[s:e]
        row["slice"] = {"start": s, "end": e}
    row["chunk_id"] = target
    _audit("open", {
        "slug": slug,
        "collection": collection_name,
        "doc_id": row.get("doc_id"),
        "chunk_id": target,
        "subjects": _audit_subjects(subjects),
    })
    return row

@mcp.tool(name="kb.neighbors", title="KB: Neighbor Chunks", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_neighbors(
    ctx: Context,
    chunk_id: str,
    collection: Optional[str] = None,
    n: int = 1,
    response_profile: str = "slim",
) -> List[Dict[str, Any]]:
    """Retrieve neighboring chunks for context expansion.

    Given a reference chunk_id, returns n chunks before and n chunks after
    from the same document, sorted by chunk_start position. Essential for
    reconstructing context around search results.

    CRITICAL: With 700-char chunks, use n=10 (21 chunks total) to capture
    complete tables, procedures, or multi-paragraph context. Never rely on
    a single chunk alone.

    Args:
        ctx: MCP context with session metadata
        chunk_id: Reference chunk UUID to anchor the window
        collection: Target collection (auto-discovers if None)
        n: Neighbor radius - returns n before + reference + n after (default 1)
        response_profile: Detail level - slim/full/diagnostic (default slim)

    Returns:
        List of chunks sorted by chunk_start position, including:
            - chunk_id, text, doc_id, path, section_path
            - chunk_start, chunk_end, page_numbers
            - is_reference: True for the anchor chunk
    """
    if not chunk_id:
        return [{"error": "missing_chunk_id"}]

    subjects = get_subjects_from_context(ctx)

    # If collection specified, use it (current behavior)
    if collection:
        slug, collection_name, _ = _resolve_scope(collection)
        fts_db_path = _get_fts_db_path(collection_name)
        seed = {"id": chunk_id}
        await _ensure_row_texts([seed], collection_name, subjects)
        doc_id = seed.get("doc_id")
        chunk_start = seed.get("chunk_start")
        if doc_id is None or chunk_start is None:
            return [{"error": "not_found"}]
    else:
        # Auto-discover: search all collections for the chunk_id
        collection_name = None
        slug = None
        fts_db_path = None
        seed = None

        for scope_slug, scope_cfg in SCOPES.items():
            coll = scope_cfg.get("collection")
            if not coll:
                continue

            test_seed = {"id": chunk_id}
            await _ensure_row_texts([test_seed], coll, subjects)

            if test_seed.get("doc_id") is not None and test_seed.get("chunk_start") is not None:
                # Found it!
                collection_name = coll
                slug = scope_slug
                fts_db_path = _get_fts_db_path(collection_name)
                seed = test_seed
                break

        if collection_name is None or seed is None:
            return [{"error": "not_found_in_any_collection"}]

        doc_id = seed.get("doc_id")
        chunk_start = seed.get("chunk_start")
    neighbor_rows_raw = _fts_neighbors(str(doc_id), int(chunk_start), max(1, int(n)), fts_db_path=fts_db_path)
    rows: List[Dict[str, Any]] = []
    for raw in neighbor_rows_raw:
        rows.append({
            "id": raw.get("chunk_id"),
            "chunk_id": raw.get("chunk_id"),
            "doc_id": raw.get("doc_id"),
            "path": raw.get("path"),
            "chunk_start": raw.get("chunk_start"),
            "chunk_end": raw.get("chunk_end"),
            "text": raw.get("text"),
            "pages": raw.get("pages"),
            "section_path": raw.get("section_path"),
            "element_ids": raw.get("element_ids"),
        })
    await _ensure_row_texts(rows, collection_name, subjects)
    rows = [row for row in rows if _is_allowed_path(row.get("path"))]
    profile = _normalize_response_profile(response_profile)
    rows = [prune_row(row, profile) for row in rows]
    _audit("neighbors", {
        "slug": slug,
        "collection": collection_name,
        "chunk_id": chunk_id,
        "count": len(rows),
        "subjects": _audit_subjects(subjects),
    })
    return rows

@mcp.tool(name="kb.summary", title="KB: Section Summary", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_summary(
    ctx: Context,
    topic: str,
    collection: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    topic = (topic or "").strip()
    if not topic:
        return [{"error": "missing_topic"}]
    slug, collection_name, _ = _resolve_scope(collection)
    results = await asyncio.to_thread(
        query_summaries,
        collection_name,
        topic,
        max(1, int(limit)),
        SUMMARY_DB_PATH,
    )
    if not results:
        return [{"info": "no_matches"}]
    return results

@mcp.tool(name="kb.outline", title="KB: Outline", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_outline(ctx: Context, doc_id: str, collection: Optional[str] = None) -> Dict[str, Any]:
    if not doc_id:
        return {"error": "missing_doc_id"}
    slug, collection_name, _ = _resolve_scope(collection)
    fts_db_path = _get_fts_db_path(collection_name)
    conn = sqlite3.connect(fts_db_path)
    try:
        cur = conn.execute(
            "SELECT text, section_path, pages, chunk_start, chunk_end FROM fts_chunks WHERE doc_id = ? AND types LIKE '%heading%' ORDER BY chunk_start",
            (doc_id,),
        )
        outline: List[Dict[str, Any]] = []
        for text, section_path, pages, chunk_start, chunk_end in cur.fetchall():
            try:
                section = json.loads(section_path) if isinstance(section_path, str) else section_path
            except Exception:
                section = section_path
            pages_list = _parse_page_numbers(pages)
            outline.append({
                "heading": text,
                "section_path": section or [],
                "pages": pages_list,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            })
    finally:
        conn.close()
    _audit("outline", {
        "slug": slug,
        "collection": collection_name,
        "doc_id": doc_id,
        "count": len(outline),
        "subjects": _audit_subjects(get_subjects_from_context(ctx)),
    })
    return {"doc_id": doc_id, "outline": outline}

@mcp.tool(name="kb.entities", title="KB: Graph Entities", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_entities(
    ctx: Context,
    collection: Optional[str] = None,
    types: Optional[List[str]] = None,
    match: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    slug, collection_name, _ = _resolve_scope(collection)
    type_filter: List[str] = []
    if isinstance(types, list):
        type_filter = [str(t).strip() for t in types if t]
    data = await asyncio.to_thread(
        graph_list_entities,
        collection_name,
        type_filter or None,
        (match or "").strip() or None,
        max(1, int(limit)),
        GRAPH_DB_PATH,
    )
    return {
        "collection": collection_name,
        "slug": slug,
        "types": type_filter,
        "match": (match or "").strip() or None,
        "entities": data,
    }

@mcp.tool(name="kb.linkouts", title="KB: Entity Mentions", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_linkouts(
    ctx: Context,
    entity_id: str,
    limit: int = 25,
) -> Dict[str, Any]:
    entity_id = (entity_id or "").strip()
    if not entity_id:
        return {"error": "missing_entity_id"}
    data = await asyncio.to_thread(
        graph_entity_linkouts,
        entity_id,
        max(1, int(limit)),
        GRAPH_DB_PATH,
    )
    if not data.get("entity"):
        return {"error": "not_found"}
    return data

@mcp.tool(name="kb.graph", title="KB: Graph Neighbors", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_graph(
    ctx: Context,
    node_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    node_id = (node_id or "").strip()
    if not node_id:
        return {"error": "missing_node_id"}
    data = await asyncio.to_thread(graph_neighbors, node_id, max(1, int(limit)), GRAPH_DB_PATH)
    if not data.get("node"):
        return {"error": "not_found"}
    return data

@mcp.tool(name="kb.promote", title="KB: Promote Doc", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
})
async def kb_promote(ctx: Context, doc_id: str, weight: float = 0.2) -> Dict[str, Any]:
    if not doc_id:
        return {"error": "missing_doc_id"}
    subjects = get_subjects_from_context(ctx)
    multiplier = _update_session_prior(subjects, doc_id, abs(weight))
    key = _session_key(subjects)
    delta = SESSION_PRIORS.get(key, {}).get(str(doc_id), 0.0)
    _audit("promote", {
        "doc_id": doc_id,
        "delta": delta,
        "subjects": _audit_subjects(subjects),
    })
    return {
        "doc_id": doc_id,
        "prior_delta": delta,
        "multiplier": multiplier,
    }

@mcp.tool(name="kb.demote", title="KB: Demote Doc", annotations={
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
})
async def kb_demote(ctx: Context, doc_id: str, weight: float = 0.2) -> Dict[str, Any]:
    if not doc_id:
        return {"error": "missing_doc_id"}
    subjects = get_subjects_from_context(ctx)
    multiplier = _update_session_prior(subjects, doc_id, -abs(weight))
    key = _session_key(subjects)
    delta = SESSION_PRIORS.get(key, {}).get(str(doc_id), 0.0)
    _audit("demote", {
        "doc_id": doc_id,
        "delta": delta,
        "subjects": _audit_subjects(subjects),
    })
    return {
        "doc_id": doc_id,
        "prior_delta": delta,
        "multiplier": multiplier,
    }

@mcp.tool(name="kb.collections", title="KB: List Collections", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
async def kb_collections(ctx: Context) -> Dict[str, Any]:
    """Return the configured collection slugs and their metadata."""
    entries = []
    for slug, cfg in SCOPES.items():
        entries.append({
            "slug": slug,
            "collection": cfg.get("collection"),
            "title": cfg.get("title"),
        })
    return {"collections": entries}

@mcp.tool(name="kb.search", title="KB: Search", annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
})
async def kb_search(ctx: Context, params: KBSearchInput) -> List[Dict[str, Any]]:
    """Vector search (semantic), rerank, hybrid, sparse, or auto planner.

    This tool performs semantic, lexical, or hybrid search based on the specified mode.
    It automatically routes queries to the optimal search strategy when mode="auto".

    Args:
        ctx: MCP context with session metadata
        params: Search parameters (KBSearchInput) including:
            - query (str): The search query text
            - collection (Optional[str]): Target collection slug or name
            - mode (str): Search mode (auto/semantic/rerank/hybrid/sparse/sparse_splade/colbert)
            - retrieve_k (int): Number of candidates to retrieve (1-256)
            - return_k (int): Number of results to return (1-256)
            - top_k (int): Top-k for reranking stage (1-256)
            - scope (Optional[Dict]): Filters and doc_id boosts
            - response_profile (str): Detail level (slim/full/diagnostic)

    Returns:
        List[Dict[str, Any]]: Search results with scores, metadata, and timings
    """
    # Extract validated parameters from Pydantic model
    query = params.query
    collection = params.collection
    mode = params.mode
    top_k = params.top_k
    retrieve_k = params.retrieve_k
    return_k = params.return_k
    scope = params.scope
    response_profile = params.response_profile

    if not query.strip():
        return [{"error": "empty_query"}]
    slug, collection_name, _ = _resolve_scope(collection)
    start_time = time.perf_counter()
    timings: Dict[str, float] = {}
    route_retrieve_val = retrieve_k
    boosts = _parse_scope_boosts(scope)
    subjects = get_subjects_from_context(ctx)

    def finalize(rows: List[Dict[str, Any]], route_name: str) -> List[Dict[str, Any]]:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        clean_rows = [r for r in rows if isinstance(r, dict) and not r.get("note") and not r.get("abstain")]
        best_score = _best_score(rows)
        abstain_flag = ANSWERABILITY_THRESHOLD > 0.0 and best_score < ANSWERABILITY_THRESHOLD
        payload = {
            "slug": slug,
            "collection": collection_name,
            "mode": mode,
            "route": route_name,
            "retrieve_k": route_retrieve_val,
            "return_k": return_k,
            "top_k": top_k,
            "result_count": len(clean_rows),
            "best_score": best_score,
            "duration_ms": round(duration_ms, 3),
            "thin_payload": THIN_PAYLOAD_ENABLED,
            "abstain": abstain_flag,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "subjects": [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in subjects if s],
            "results": [
                {
                    "doc_id": row.get("doc_id"),
                    "element_ids": row.get("element_ids"),
                    "chunk_id": row.get("chunk_id"),
                    "score": row.get("score"),
                }
                for row in clean_rows[:return_k]
            ],
        }
        if boosts:
            payload["boosts"] = boosts
        if abstain_flag:
            payload.setdefault("reason", "best_score_below_threshold")
            payload["threshold"] = ANSWERABILITY_THRESHOLD
        try:
            _audit("search", payload)
        except Exception:
            logger.debug("audit_log_failed", exc_info=True)
        try:
            logger.info("search_metrics %s", json.dumps(payload))
        except Exception:
            logger.debug("failed to log search metrics", exc_info=True)
        return rows

    valid_modes = {"semantic", "rerank", "hybrid", "sparse", "sparse_splade", "colbert", "auto"}
    if mode not in valid_modes:
        return finalize([{"error": f"invalid mode '{mode}', expected one of {sorted(valid_modes)}"}], mode)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        return finalize([{"error": "invalid top_k", "detail": "top_k must be int between 1 and 100"}], mode)
    if not isinstance(retrieve_k, int) or retrieve_k < 1 or retrieve_k > 256:
        return finalize([{"error": "invalid retrieve_k", "detail": "retrieve_k must be int between 1 and 256"}], mode)
    if not isinstance(return_k, int) or return_k < 1 or return_k > retrieve_k:
        return finalize([{"error": "invalid return_k", "detail": "return_k must be int between 1 and retrieve_k"}], mode)

    route = mode
    route_retrieve = retrieve_k
    if mode == "auto":
        try:
            planned_route = await plan_route(query)
        except Exception as exc:
            logger.debug("plan_route failed, falling back to rerank: %s", exc)
            planned_route = {}
        route = str((planned_route or {}).get("route") or "rerank")
        if route not in {"semantic", "rerank", "hybrid", "sparse", "sparse_splade", "colbert"}:
            route = "rerank"
        planned_k = (planned_route or {}).get("k")
        if isinstance(planned_k, int) and 1 <= planned_k <= 256:
            route_retrieve = planned_k
    route_retrieve = max(return_k, min(route_retrieve, 256))
    route_retrieve_val = route_retrieve

    query_vec: List[float] = []
    if route in {"semantic", "rerank", "hybrid"}:
        try:
            embed_start = time.perf_counter()
            query_vec = await embed_query(query, normalize=True)
            timings["embed_ms"] = (time.perf_counter() - embed_start) * 1000.0
        except Exception as exc:
            return finalize([{ "error": "embedding_failed", "detail": str(exc)}], route)

    rows = await _execute_search(
        route=route,
        collection=collection_name,
        query=query,
        query_vec=query_vec,
        retrieve_k=route_retrieve,
        return_k=return_k,
        top_k=top_k,
        subjects=subjects,
        timings=timings,
        boosts=boosts,
    )

    if rows and isinstance(rows[0], dict) and rows[0].get("error"):
        return finalize(rows, route)

    for row in rows:
        if isinstance(row, dict) and "route" not in row:
            row["route"] = route

    best = _best_score(rows)
    if ANSWERABILITY_THRESHOLD > 0.0 and best < ANSWERABILITY_THRESHOLD:
        return finalize([
            {
                "abstain": True,
                "reason": "low_answerability",
                "top_score": best,
                "threshold": ANSWERABILITY_THRESHOLD,
            }
        ], route)

    if route not in {"sparse", "sparse_splade"} and _should_retry_sparse(query, rows):
        sparse_rows = await _execute_search(
            route="sparse",
            collection=collection_name,
            query=query,
            query_vec=query_vec,
            retrieve_k=max(route_retrieve, 32),
            return_k=return_k,
            top_k=top_k,
            subjects=subjects,
            timings=timings,
            boosts=boosts,
        )
        if sparse_rows and not sparse_rows[0].get("error"):
            for row in sparse_rows:
                if isinstance(row, dict) and "route" not in row:
                    row["route"] = "sparse"
            rows = sparse_rows
            route = "sparse"

    profile = _normalize_response_profile(response_profile)
    rows = [prune_row(row, profile) for row in rows]
    return finalize(rows, route)

if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        logger.exception("MCP server crashed")
        raise
