"""Per-document extraction+chunking worker for multiprocessing.

This module isolates Docling-heavy extraction into a standalone function
that can be safely called from a ProcessPoolExecutor with 'spawn' start method.
Workers do NOT embed — they return chunk texts for centralized embedding on
the main thread (avoids Ollama contention and large IPC overhead).
"""

import hashlib
import json
import os
import pathlib
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def file_uri(p: pathlib.Path) -> str:
    return p.resolve().as_uri()


def translate_path(posix_path: str, source_prefix: str, display_prefix: str) -> Optional[str]:
    """Convert '/mnt/c/Users/hvksh/docs/file.pdf' → 'C:\\Users\\hvksh\\docs\\file.pdf'"""
    if not source_prefix or not display_prefix:
        return None
    if posix_path.startswith(source_prefix):
        remainder = posix_path[len(source_prefix):]
        return display_prefix.rstrip("\\") + remainder.replace("/", "\\")
    return None


def _alpha_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text))


def _select_chunk_profile(blocks: list) -> str:
    if not blocks:
        return "fixed_window"
    table_rows = sum(1 for b in blocks if getattr(b, "type", "") == "table_row")
    if table_rows:
        return "table_row"
    step_like = 0
    for b in blocks:
        btype = getattr(b, "type", "")
        text = getattr(b, "text", "") or ""
        text = text.strip()
        if btype == "list":
            step_like += 1
        elif re.match(r"^\d+(\.\d+)*\s", text):
            step_like += 1
    if step_like and step_like >= max(3, len(blocks) // 4):
        return "procedure_block"
    return "heading_based"


def process_single_document(
    file_path: str,
    max_chars: int,
    overlap: int,
    min_words: int,
    source_prefix: str = "",
    display_prefix: str = "",
    plan_dir: str = "data/ingest_plans",
) -> Dict[str, Any]:
    """Extract and chunk a single document. Returns data for centralized embedding.

    NOTE: Does NOT embed — returns texts for the main thread to embed in batches.
    This function is designed to be called from a ProcessPoolExecutor with 'spawn'.

    Returns dict with keys:
        doc_id, path_str, source_path, filename, content_hash, mtime, mtime_ns,
        file_size, chunks, texts, plan_payload, chunk_profile, error
    """
    result: Dict[str, Any] = {"error": None}
    try:
        p = pathlib.Path(file_path)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_uri(p)))
        path_str = p.resolve().as_posix()
        stat_info = p.stat()

        result["doc_id"] = doc_id
        result["path_str"] = path_str
        result["filename"] = p.name
        result["mtime"] = int(stat_info.st_mtime)
        result["mtime_ns"] = stat_info.st_mtime_ns
        result["file_size"] = stat_info.st_size
        result["source_path"] = translate_path(path_str, source_prefix, display_prefix)

        # Import heavy deps inside worker (spawn-safe)
        from ingest_blocks import extract_document_blocks, chunk_blocks
        from metadata_schema import generate_metadata

        # Load existing plan if available
        plan_path = pathlib.Path(plan_dir) / f"{doc_id}.json"
        existing_plan = None
        try:
            if plan_path.exists():
                existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            pass

        plan_override = existing_plan.get("triage") if isinstance(existing_plan, dict) else None

        # Extract
        triage_blocks, triage_info = extract_document_blocks(p, doc_id, plan_override=plan_override)
        if not triage_blocks:
            result["error"] = "no_blocks"
            return result

        text = "\n\n".join(b.text for b in triage_blocks if b.text)
        if not text.strip():
            result["error"] = "empty_text"
            return result

        if min_words and _alpha_word_count(text) < min_words:
            result["error"] = "below_min_words"
            return result

        # Chunk
        overlap_sentences = max(1, overlap // 80 if overlap else 1)
        if plan_override and isinstance(plan_override, dict):
            params = existing_plan.get("chunk_params") if isinstance(existing_plan, dict) else None
            if isinstance(params, dict):
                overlap_sentences = int(params.get("overlap_sentences", overlap_sentences) or overlap_sentences)

        chunk_profile = (
            existing_plan.get("chunk_profile")
            if isinstance(existing_plan, dict) and existing_plan.get("chunk_profile")
            else _select_chunk_profile(triage_blocks)
        )
        chunks, raw_text = chunk_blocks(
            triage_blocks,
            max_chars,
            overlap_sentences=overlap_sentences,
            profile=chunk_profile,
        )

        if not raw_text.strip():
            raw_text = text
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        # Metadata
        doc_metadata, metadata_rejects = generate_metadata(raw_text, chunks)

        # Prepare texts for centralized embedding
        texts = [c.get("text", "") for c in chunks]

        result["chunks"] = chunks
        result["texts"] = texts
        result["content_hash"] = content_hash
        result["chunk_profile"] = chunk_profile
        result["doc_metadata"] = doc_metadata
        result["metadata_rejects"] = metadata_rejects
        result["triage_info"] = {
            "tier": triage_info.get("tier", "UNKNOWN") if isinstance(triage_info, dict) else "UNKNOWN",
        }
        result["existing_plan"] = existing_plan

    except Exception as ex:
        result["error"] = str(ex)

    return result
