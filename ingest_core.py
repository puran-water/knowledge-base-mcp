import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
from tqdm import tqdm

from lexical_index import FTSWriter
from graph_builder import update_graph
from summary_index import upsert_summaries

try:  # optional dependency during tests
    from sparse_expansion import SparseExpander  # type: ignore
except Exception:  # pragma: no cover - optional
    SparseExpander = None  # type: ignore

try:
    from qdrant_client import QdrantClient, models
except ImportError as exc:  # pragma: no cover - qdrant is a runtime dependency
    raise RuntimeError("qdrant-client is required for ingestion") from exc


_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a shared requests.Session with connection pooling."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=10)
        _SESSION.mount("http://", adapter)
        _SESSION.mount("https://", adapter)
    return _SESSION


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "snowflake-arctic-embed:xs")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_QDRANT_METRIC = os.getenv("QDRANT_METRIC", "cosine")
DEFAULT_FTS_DB = os.getenv("FTS_DB_PATH", "data/fts.db")
INGEST_MODEL_VERSION = os.getenv("INGEST_MODEL_VERSION", "structured_ingest_v1")
INGEST_PROMPT_SHA = os.getenv("INGEST_PROMPT_SHA", "sha_deterministic_chunking_v1")
MAX_METADATA_BYTES = int(os.getenv("MAX_METADATA_BYTES", "8192"))

_QDRANT_CLIENTS: Dict[Tuple[str, Optional[str]], QdrantClient] = {}


def l2_normalize(mat: Sequence[Sequence[float]]) -> List[List[float]]:
    import math

    out: List[List[float]] = []
    for row in mat:
        norm = math.sqrt(sum(v * v for v in row)) or 1.0
        out.append([v / norm for v in row])
    return out


def embed_texts(
    ollama_url: str,
    model: str,
    texts: List[str],
    batch_size: int = 128,
    timeout: int = 120,
    normalize: bool = True,
    parallel: int = 1,
    num_threads: int = 8,
    keep_alive: str = "1h",
    force_per_item: bool = False,
) -> List[List[float]]:
    headers = {"content-type": "application/json"}
    embeddings: List[List[float]] = []
    total = len(texts)
    if total == 0:
        return embeddings
    if total <= 1 or force_per_item:
        for text in texts:
            r = _get_session().post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text, "keep_alive": keep_alive, "options": {"num_thread": num_threads}},
                timeout=timeout,
                headers=headers,
            )
            r.raise_for_status()
            vec = r.json().get("embedding")
            if vec is None:
                raise RuntimeError("Missing 'embedding' in /api/embeddings response")
            embeddings.append(vec)
        return l2_normalize(embeddings) if normalize else embeddings

    # Calculate total batches for progress reporting
    total_batches = (total + batch_size - 1) // batch_size

    index = 0
    with tqdm(total=total_batches, desc="Embedding batches", leave=False, unit="batch") as pbar:
        while index < total:
            batch = texts[index:index + batch_size]
            index += batch_size
            payload = {
                "model": model,
                "input": batch,
                "keep_alive": keep_alive,
                "options": {"parallel": parallel, "num_thread": num_threads},
            }
            for attempt in range(3):
                try:
                    r = _get_session().post(
                        f"{ollama_url}/api/embed",
                        json=payload,
                        timeout=timeout,
                        headers=headers,
                    )
                    if r.status_code == 404:
                        raise RuntimeError("/api/embed endpoint unavailable")
                    r.raise_for_status()
                    batch_emb = r.json().get("embeddings")
                    if not isinstance(batch_emb, list):
                        raise RuntimeError("Unexpected /api/embed response structure")
                    if normalize:
                        batch_emb = l2_normalize(batch_emb)
                    embeddings.extend(batch_emb)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            pbar.update(1)
    return embeddings


def embed_texts_robust(
    ollama_url: str,
    model: str,
    texts: List[str],
    timeout: int = 120,
    normalize: bool = True,
    num_threads: int = 8,
    keep_alive: str = "1h",
    max_retries: int = 2,
) -> List[Optional[List[float]]]:
    headers = {"content-type": "application/json"}
    out: List[Optional[List[float]]] = [None] * len(texts)
    for i, text in enumerate(texts):
        for attempt in range(max_retries + 1):
            try:
                r = _get_session().post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": model, "prompt": text, "keep_alive": keep_alive, "options": {"num_thread": num_threads}},
                    timeout=timeout,
                    headers=headers,
                )
                r.raise_for_status()
                vec = r.json().get("embedding")
                if vec is None:
                    raise RuntimeError("Missing 'embedding' in /api/embeddings response")
                out[i] = l2_normalize([vec])[0] if normalize else vec
                break
            except Exception:
                if attempt == max_retries:
                    out[i] = None
                else:
                    time.sleep(1.0 * (attempt + 1))
    return out


def _qdrant_distance(metric: str):
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }
    if metric not in mapping:
        raise ValueError(f"Unsupported Qdrant metric '{metric}'")
    return mapping[metric]


def get_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None, timeout: Optional[int] = None) -> QdrantClient:
    key = (url or DEFAULT_QDRANT_URL, api_key)
    if key not in _QDRANT_CLIENTS:
        client_kwargs: Dict[str, Any] = {"url": key[0], "api_key": key[1]}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        _QDRANT_CLIENTS[key] = QdrantClient(**client_kwargs)
    return _QDRANT_CLIENTS[key]


def ensure_qdrant_collection(
    client: QdrantClient,
    name: str,
    size: int,
    metric: str,
) -> None:
    dist = _qdrant_distance(metric)
    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=dist),
        )
        return
    info = client.get_collection(collection_name=name)
    cfg = info.config.params.vectors
    current_size = getattr(cfg, "size", None)
    current_dist = getattr(cfg, "distance", None)
    if current_size != size or current_dist != dist:
        raise RuntimeError(
            f"Existing collection '{name}' size={current_size} distance={current_dist}; expected size={size} distance={dist}"
        )


def upsert_qdrant(
    client: QdrantClient,
    collection: str,
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[Dict[str, Any]],
    ids: Sequence[str],
) -> None:
    """
    Upsert vectors to Qdrant with batching for large collections.
    Batches large upserts into 512-vector chunks with progress reporting.
    """
    BATCH_SIZE = 512
    total_vectors = len(vectors)

    if total_vectors <= BATCH_SIZE:
        # Fast path for small batches - no progress reporting needed
        points = [
            models.PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        client.upsert(collection_name=collection, points=points)
    else:
        # Progress reporting for large batches
        for i in range(0, total_vectors, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_vectors)
            batch_points = [
                models.PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j])
                for j in range(i, batch_end)
            ]
            client.upsert(collection_name=collection, points=batch_points)
            tqdm.write(f"    Upserted {batch_end}/{total_vectors} vectors")


def qdrant_any_by_filter(
    client: QdrantClient,
    collection: str,
    must: List[Dict[str, Any]],
) -> bool:
    conds = [
        models.FieldCondition(
            key=item["key"],
            match=models.MatchValue(value=item["value"]),
        )
        for item in must
    ]
    flt = models.Filter(must=conds)
    points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=1, with_payload=False, with_vectors=False)
    return bool(points)


def qdrant_delete_by_doc_id(client: QdrantClient, collection: str, doc_id: str) -> None:
    flt = models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ]
    )
    client.delete(collection_name=collection, points_selector=models.FilterSelector(filter=flt))


def qdrant_batch_doc_metadata(
    client: QdrantClient,
    collection: str,
    doc_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Batch fetch mtime_ns + file_size for doc_ids from Qdrant.

    Returns {doc_id: {"mtime_ns": ..., "file_size": ..., "content_hash": ...}}.
    Only fetches one point per doc_id to minimize data transfer.
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not doc_ids:
        return result
    # Process in batches to avoid very large scroll requests
    BATCH = 100
    for i in range(0, len(doc_ids), BATCH):
        batch_ids = doc_ids[i : i + BATCH]
        for doc_id in batch_ids:
            try:
                flt = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
                points, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=flt,
                    limit=1,
                    with_payload=["mtime_ns", "file_size", "content_hash"],
                    with_vectors=False,
                )
                if points:
                    pl = points[0].payload or {}
                    result[doc_id] = {
                        "mtime_ns": pl.get("mtime_ns"),
                        "file_size": pl.get("file_size"),
                        "content_hash": pl.get("content_hash"),
                    }
            except Exception:
                continue
    return result


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"artifact not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _normalize_pages(pages: Any) -> List[int]:
    if isinstance(pages, list):
        return [int(x) for x in pages if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit())]
    if isinstance(pages, str):
        import re

        return [int(d) for d in re.findall(r"\d+", pages)]
    if isinstance(pages, (int, float)):
        return [int(pages)]
    return []


def _uuid_from_doc(doc_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(doc_id)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_URL, doc_id)


def upsert_document_artifacts(
    doc_id: str,
    collection: str,
    chunk_artifact: Union[str, Path],
    *,
    metadata_artifact: Optional[Union[str, Path]] = None,
    ollama_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    embed_batch_size: int = 32,
    embed_timeout: int = 120,
    embed_parallel: int = 1,
    embed_threads: int = 8,
    embed_keepalive: str = "1h",
    embed_force_per_item: bool = False,
    embed_robust: bool = False,
    qdrant_client: Optional[QdrantClient] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_metric: str = DEFAULT_QDRANT_METRIC,
    fts_db_path: Optional[str] = None,
    fts_recreate: bool = False,
    thin_payload: Optional[bool] = None,
    update_graph_links: bool = True,
    update_summary_index: bool = True,
    skip_vectors: bool = False,
    sparse_expander: Optional[Any] = None,
) -> Dict[str, Any]:
    chunk_data = _load_json(chunk_artifact)
    if metadata_artifact:
        meta_data = _load_json(metadata_artifact)
        if isinstance(meta_data.get("doc_metadata"), dict):
            chunk_data["doc_metadata"] = meta_data["doc_metadata"]
        if isinstance(meta_data.get("plan_hash"), str):
            chunk_data["plan_hash"] = meta_data["plan_hash"]

    artifact_doc_id = chunk_data.get("doc_id")
    if artifact_doc_id and artifact_doc_id != doc_id:
        raise ValueError(f"doc_id mismatch: artifact has {artifact_doc_id}, expected {doc_id}")

    chunks = chunk_data.get("chunks") or []
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunk artifact contains no chunks")

    path_str = chunk_data.get("path") or ""
    plan_hash = chunk_data.get("plan_hash")
    doc_metadata = chunk_data.get("doc_metadata") or {}
    chunk_profile = chunk_data.get("chunk_profile") or ""
    content_hash = chunk_data.get("content_hash")
    plan_model_version = chunk_data.get("model_version", INGEST_MODEL_VERSION)
    plan_prompt_sha = chunk_data.get("prompt_sha", INGEST_PROMPT_SHA)
    metadata_calls = chunk_data.get("metadata_calls", 0)

    if metadata_artifact:
        if metadata_bytes := len(json.dumps(doc_metadata, ensure_ascii=False).encode("utf-8")):
            if metadata_bytes > MAX_METADATA_BYTES:
                raise ValueError(f"metadata exceeds MAX_METADATA_BYTES ({metadata_bytes}>{MAX_METADATA_BYTES})")

    doc_uuid = _uuid_from_doc(doc_id)
    ollama_url = ollama_url or DEFAULT_OLLAMA_URL
    ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL
    fts_db_path = fts_db_path or DEFAULT_FTS_DB

    client = qdrant_client or get_qdrant_client(qdrant_url, qdrant_api_key)
    thin_payload_flag = thin_payload
    if thin_payload_flag is None:
        thin_payload_flag = bool(chunk_data.get("thin_payload"))

    filename = ""
    mtime = int(time.time())
    file_size = 0
    mtime_ns = 0
    source_path_str = chunk_data.get("source_path") or ""
    if path_str:
        path_obj = Path(path_str)
        filename = path_obj.name
        try:
            st = path_obj.stat()
            mtime = int(st.st_mtime)
            file_size = st.st_size
            mtime_ns = st.st_mtime_ns
        except Exception:
            pass

    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []
    ids: List[str] = []
    for chunk in chunks:
        chunk_start = int(chunk.get("chunk_start", 0) or 0)
        chunk_end = int(chunk.get("chunk_end", 0) or 0)
        chunk_uuid = uuid.uuid5(doc_uuid, f"{chunk_start}-{chunk_end}")
        text = chunk.get("text", "") or ""
        texts.append(text)
        pages = _normalize_pages(chunk.get("pages"))
        section_path = chunk.get("section_path") or []
        element_ids = chunk.get("element_ids") or []
        bboxes = chunk.get("bboxes") or []
        types = chunk.get("types") or []
        source_tools = chunk.get("source_tools") or []
        table_headers = chunk.get("headers") or chunk.get("table_headers") or []
        table_units = chunk.get("units") or chunk.get("table_units") or []
        sparse_terms = chunk.get("sparse_terms")

        payload: Dict[str, Any] = {
            "doc_id": doc_id,
            "path": path_str,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "filename": filename,
            "source_path": source_path_str or None,
            "mtime": mtime,
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            "content_hash": content_hash,
            "page_numbers": pages,
            "section_path": section_path,
            "element_ids": element_ids,
            "bboxes": bboxes,
            "types": types,
            "source_tools": source_tools,
            "table_headers": table_headers,
            "table_units": table_units,
            "chunk_profile": chunk.get("profile") or chunk_profile,
            "plan_hash": plan_hash,
            "model_version": plan_model_version,
            "prompt_sha": plan_prompt_sha,
            "doc_metadata": doc_metadata,
        }
        if sparse_terms:
            payload["sparse_terms"] = sparse_terms
        if not thin_payload_flag:
            payload["text"] = text
        else:
            payload["thin_payload"] = True
        payloads.append(payload)
        ids.append(str(chunk_uuid))
        chunk.setdefault("doc_metadata", doc_metadata)

    vectors: List[List[float]] = []
    vector_count = 0
    if not skip_vectors:
        if embed_robust:
            vecs = embed_texts_robust(
                ollama_url,
                ollama_model,
                texts,
                timeout=embed_timeout,
                normalize=(qdrant_metric == "cosine"),
                num_threads=embed_threads,
                keep_alive=embed_keepalive,
            )
            filtered_ids = []
            filtered_payloads = []
            filtered_vecs = []
            filtered_texts = []
            for cid, payload, vec, text in zip(ids, payloads, vecs, texts):
                if vec is None:
                    continue
                filtered_ids.append(cid)
                filtered_payloads.append(payload)
                filtered_vecs.append(vec)
                filtered_texts.append(text)
            ids = filtered_ids
            payloads = filtered_payloads
            texts = filtered_texts
            vectors = filtered_vecs
        else:
            vectors = embed_texts(
                ollama_url,
                ollama_model,
                texts,
                batch_size=embed_batch_size,
                timeout=embed_timeout,
                normalize=(qdrant_metric == "cosine"),
                parallel=embed_parallel,
                num_threads=embed_threads,
                keep_alive=embed_keepalive,
                force_per_item=embed_force_per_item,
            )
        if not vectors:
            raise RuntimeError("Failed to obtain embeddings for chunks")
        ensure_qdrant_collection(client, collection, len(vectors[0]), qdrant_metric)
        upsert_qdrant(client, collection, vectors, payloads, ids)
        vector_count = len(vectors)

    fts_rows = 0
    if fts_db_path:
        writer = FTSWriter(fts_db_path, recreate=fts_recreate)
        try:
            writer.delete_doc(doc_id)
            rows = []
            for cid, chunk, text, payload in zip(ids, chunks, texts, payloads):
                pages = _normalize_pages(chunk.get("pages"))
                section_path = chunk.get("section_path") or []
                element_ids = chunk.get("element_ids") or []
                bboxes = chunk.get("bboxes") or []
                types = chunk.get("types") or []
                source_tools = chunk.get("source_tools") or []
                table_headers = chunk.get("headers") or chunk.get("table_headers") or []
                table_units = chunk.get("units") or chunk.get("table_units") or []
                doc_metadata_payload = chunk.get("doc_metadata") or doc_metadata or {}
                rows.append({
                    "text": text,
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "path": path_str,
                    "filename": filename,
                    "source_path": source_path_str or "",
                    "chunk_start": payload.get("chunk_start"),
                    "chunk_end": payload.get("chunk_end"),
                    "mtime": mtime,
                    "page_numbers": ",".join(str(n) for n in pages) if pages else "",
                    "pages": json.dumps(pages, ensure_ascii=False),
                    "section_path": json.dumps(section_path, ensure_ascii=False) if section_path else "",
                    "element_ids": json.dumps(element_ids, ensure_ascii=False) if element_ids else "",
                    "bboxes": json.dumps(bboxes, ensure_ascii=False) if bboxes else "",
                    "types": json.dumps(types, ensure_ascii=False) if types else "",
                    "source_tools": json.dumps(source_tools, ensure_ascii=False) if source_tools else "",
                    "table_headers": json.dumps(table_headers, ensure_ascii=False) if table_headers else "",
                    "table_units": json.dumps(table_units, ensure_ascii=False) if table_units else "",
                    "chunk_profile": payload.get("chunk_profile", ""),
                    "plan_hash": plan_hash or "",
                    "model_version": plan_model_version or "",
                    "prompt_sha": plan_prompt_sha or "",
                    "doc_metadata": json.dumps(doc_metadata_payload, ensure_ascii=False) if doc_metadata_payload else "",
                    "sparse_terms": chunk.get("sparse_terms", []),
                })
            writer.upsert_many(rows)
            fts_rows = len(rows)
        finally:
            writer.close()

    if update_graph_links:
        try:
            update_graph(collection, doc_id, path_str, chunks)
        except Exception:
            pass

    if update_summary_index:
        try:
            upsert_summaries(collection, doc_id, chunks)
        except Exception:
            pass

    return {
        "status": "ok",
        "doc_id": doc_id,
        "collection": collection,
        "plan_hash": plan_hash,
        "chunks_upserted": len(ids),
        "qdrant_points": vector_count,
        "fts_rows": fts_rows,
        "thin_payload": thin_payload_flag,
        "model_version": plan_model_version,
        "prompt_sha": plan_prompt_sha,
        "metadata_calls": metadata_calls,
    }


__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_QDRANT_URL",
    "DEFAULT_QDRANT_API_KEY",
    "DEFAULT_QDRANT_METRIC",
    "DEFAULT_FTS_DB",
    "INGEST_MODEL_VERSION",
    "INGEST_PROMPT_SHA",
    "MAX_METADATA_BYTES",
    "SparseExpander",
    "embed_texts",
    "embed_texts_robust",
    "l2_normalize",
    "get_qdrant_client",
    "ensure_qdrant_collection",
    "upsert_qdrant",
    "qdrant_any_by_filter",
    "qdrant_delete_by_doc_id",
    "qdrant_batch_doc_metadata",
    "upsert_document_artifacts",
]
