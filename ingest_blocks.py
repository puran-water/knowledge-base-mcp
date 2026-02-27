import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import warnings

warnings.filterwarnings("ignore", message=r".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*swigvarlink.*", category=DeprecationWarning)

os.environ.setdefault("HF_HOME", str(Path(".cache") / "hf"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

from structured_chunk import detect_blocks, _expand_table_block


# Docling-only extraction - no routing or triage needed


@dataclass
class Block:
    doc_id: str
    path: str
    page: int
    type: str
    text: str
    section_path: List[str]
    element_id: str
    bbox: Optional[List[float]]
    headers: Optional[List[str]]
    units: Optional[Any]
    span: Optional[List[int]]
    source_tool: str


def _init_counters() -> Dict[str, int]:
    return defaultdict(int)


DOC_CONVERTER: Optional[Any] = None  # Type: DocumentConverter when loaded
CONVERTER_CACHE: Dict[Tuple[bool, int], Any] = {}  # Cache: (needs_ocr, timeout) -> DocumentConverter


def _check_pdf_needs_ocr(pdf_path: Path) -> bool:
    """Check if PDF has extractable text or needs OCR by sampling across document."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        # Sample first, middle, last pages (not just first 3)
        pages_to_check = [0]  # Always check first page
        if total_pages > 1:
            pages_to_check.append(total_pages - 1)  # Last page
        if total_pages > 2:
            pages_to_check.append(total_pages // 2)  # Middle page
        if total_pages > 10:
            # For larger docs, also check quarter points
            pages_to_check.append(total_pages // 4)
            pages_to_check.append(3 * total_pages // 4)

        # Remove duplicates and sort
        pages_to_check = sorted(set(pages_to_check))

        text_chars = 0
        for page_idx in pages_to_check:
            text = reader.pages[page_idx].extract_text() or ""
            text_chars += len(text.strip())

        # If average < 100 chars per page, likely needs OCR
        avg_chars = text_chars / len(pages_to_check) if pages_to_check else 0
        needs_ocr = avg_chars < 100

        print(f"  OCR detection: {avg_chars:.0f} chars/page avg (sampled {len(pages_to_check)}/{total_pages} pages) -> {'needs OCR' if needs_ocr else 'has text'}")
        return needs_ocr
    except Exception as e:
        print(f"  OCR detection failed ({e}), assuming needs OCR")
        return True


def _get_docling_converter(needs_ocr: bool = True, file_size_mb: float = 0):
    """
    Get cached Docling converter with 2-tier configuration.

    Tier 1 (HIGH_FIDELITY): OCR OFF, ACCURATE tables, pictures ON
    Tier 2 (OCR_LEAN): OCR ON, FAST tables, pictures OFF

    Args:
        needs_ocr: Whether PDF needs OCR (detected via text sniff)
        file_size_mb: File size in MB for timeout scaling
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )

    # Determine timeout and tier
    # No timeout - let documents process to completion regardless of size
    timeout_seconds = None
    if needs_ocr:
        tier_name = "OCR_LEAN"
    else:
        tier_name = "HIGH_FIDELITY"

    # Check cache
    cache_key = (needs_ocr, timeout_seconds)
    if cache_key in CONVERTER_CACHE:
        print(f"  Pipeline: {tier_name} (cached)")
        return CONVERTER_CACHE[cache_key]

    # Log tier configuration
    if needs_ocr:
        print(f"  Pipeline: {tier_name} (Tables=FAST, Pictures=OFF, OCR=ON, timeout=None)")
        pdf_opts = PdfPipelineOptions(
            do_ocr=True,
            do_picture_description=False,
            do_picture_classification=False,
            table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST),
            document_timeout=timeout_seconds,
        )
    else:
        print(f"  Pipeline: {tier_name} (Tables=FAST, Pictures=OFF, OCR=OFF, timeout=None)")
        pdf_opts = PdfPipelineOptions(
            do_ocr=False,
            do_picture_description=False,
            do_picture_classification=False,
            table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST),
            document_timeout=timeout_seconds,
        )

    # Create and cache converter
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    CONVERTER_CACHE[cache_key] = converter

    return converter


def _next_element_id(counters: Dict[str, int], prefix: str, page: int) -> str:
    counters[prefix] += 1
    return f"{prefix}_{page}_{counters[prefix]}"


def _update_heading_path(
    heading_path: List[str],
    text: str,
    level: int,
) -> List[str]:
    level = max(1, min(level, 6))
    while len(heading_path) >= level:
        heading_path.pop()
    heading_path.append(text.strip())
    return heading_path


def _infer_heading_level(text: str) -> int:
    match = re.match(r"^(?P<num>(?:\d+\.)*\d+)\s+", text.strip())
    if match:
        return min(6, match.group("num").count(".") + 1)
    if text.isupper():
        return 2
    return 1


# Common engineering units for trailing detection
KNOWN_UNITS = {
    # SI base and derived
    "m", "mm", "cm", "km", "g", "kg", "mg", "L", "mL", "s", "min", "h", "hr", "A", "V", "W", "kW", "MW",
    "Pa", "kPa", "MPa", "bar", "psi", "psig", "N", "kN", "J", "kJ", "MJ",
    # Temperature
    "°C", "°F", "K", "degC", "degF",
    # Flow and volume
    "m³", "m3", "gal", "ft³", "ft3", "GPM", "gpm", "LPM", "lpm", "m³/h", "m3/h", "L/s", "m³/s", "m3/s",
    # Concentration
    "mg/L", "g/L", "ppm", "ppb", "%", "wt%", "vol%",
    # Velocity
    "m/s", "ft/s", "mph", "km/h",
    # Other
    "rpm", "Hz", "kHz", "MHz", "Ω", "ohm", "mol", "mmol", "μm", "nm"
}


def _extract_unit_from_header(header: str) -> Tuple[str, Optional[str]]:
    """
    Extract parameter name and unit from table header.
    Handles: (unit), [unit], comma-separated, and trailing known units.
    Returns: (param_name, unit) or (header, None) if no unit found
    """
    header = header.strip()

    # 1. Parentheses: "Parameter (unit)"
    match = re.match(r"^(?P<name>.+?)\s*\((?P<unit>[^)]+)\)$", header)
    if match:
        return match.group("name").strip(), match.group("unit").strip()

    # 2. Square brackets: "Parameter [unit]"
    match = re.match(r"^(?P<name>.+?)\s*\[(?P<unit>[^\]]+)\]$", header)
    if match:
        return match.group("name").strip(), match.group("unit").strip()

    # 3. Comma-separated: "Temperature, °C"
    match = re.match(r"^(?P<name>.+?)\s*,\s*(?P<unit>.+)$", header)
    if match:
        unit_candidate = match.group("unit").strip()
        # Validate it looks like a unit (short, no long words)
        if len(unit_candidate) <= 20 and not re.search(r"\s+[a-z]{5,}", unit_candidate):
            return match.group("name").strip(), unit_candidate

    # 4. Trailing known units: "Flow Rate m³/h"
    words = header.rsplit(None, 1)  # Split on last whitespace
    if len(words) == 2:
        name, potential_unit = words
        if potential_unit in KNOWN_UNITS or potential_unit.replace("³", "3") in KNOWN_UNITS:
            return name.strip(), potential_unit

    # No unit found
    return header, None


def blocks_from_doc_dict(
    doc_dict: dict,
    path: Path,
    doc_id: str,
) -> Tuple[List[Block], Dict[str, Any]]:
    """Convert a Docling doc_dict into Block objects.

    Enables remote extraction (Modal) to return doc_dict while
    local code handles block construction and chunking.
    """
    if not doc_dict:
        return [], {"error": "Empty doc_dict", "total_pages": 0}

    # Group all content by page for organized processing
    page_items: Dict[int, Dict[str, List]] = defaultdict(lambda: {"tables": [], "texts": [], "pictures": [], "captions": []})

    # Organize tables by page
    for table in doc_dict.get("tables", []):
        prov = table.get("prov") or []
        if prov:
            page_no = prov[0].get("page_no")
            if page_no:
                page_items[page_no]["tables"].append(table)
                page_items[page_no]["captions"].extend(table.get("captions", []))

    # Organize text items by page
    for item in doc_dict.get("texts", []):
        prov = item.get("prov") or []
        if prov:
            page_no = prov[0].get("page_no")
            if page_no:
                page_items[page_no]["texts"].append(item)

    # Organize pictures by page
    for picture in doc_dict.get("pictures", []):
        prov = picture.get("prov") or []
        if prov:
            page_no = prov[0].get("page_no")
            if page_no:
                page_items[page_no]["pictures"].append(picture)

    # Now process all pages in order
    blocks: List[Block] = []
    heading_state: List[str] = []
    counters = _init_counters()
    total_pages = max(page_items.keys()) if page_items else 0

    for page_num in sorted(page_items.keys()):
        page_data = page_items[page_num]

        # Process tables first
        for table in page_data["tables"]:
            table_id = _next_element_id(counters, "table", page_num)
            cells = table.get("data", {}).get("table_cells", [])
            rows_map: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
            headers_map: Dict[int, str] = {}

            for cell in cells:
                start_row = cell.get("start_row_offset_idx")
                start_col = cell.get("start_col_offset_idx")
                text = (cell.get("text") or "").strip()
                if text:
                    rows_map[start_row][start_col] = cell
                if start_row == 0:
                    headers_map[start_col] = text

            headers = [headers_map[idx] for idx in sorted(headers_map)]

            for row_idx, col_map in sorted(rows_map.items()):
                if row_idx == 0:
                    continue
                parts = []
                units_map = {}
                for col_idx, cell in sorted(col_map.items()):
                    header = headers_map.get(col_idx, f"col_{col_idx}")
                    value = cell.get("text", "").strip()
                    if not value:
                        continue
                    parts.append(f"{header}: {value}")

                    # Enhanced unit extraction: parentheses, brackets, comma-separated, trailing units
                    param_name, unit = _extract_unit_from_header(header)
                    if unit:
                        units_map[param_name] = unit

                if not parts:
                    continue

                bbox = None
                bboxes = [cell.get("bbox") for cell in col_map.values() if cell.get("bbox")]
                if bboxes:
                    left = min(b["l"] for b in bboxes)
                    top = min(b["t"] for b in bboxes)
                    right = max(b["r"] for b in bboxes)
                    bottom = max(b["b"] for b in bboxes)
                    bbox = [left, top, right, bottom]

                element_id = _next_element_id(counters, "table_row", page_num)
                blocks.append(
                    Block(
                        doc_id=doc_id,
                        path=str(path),
                        page=page_num,
                        type="table_row",
                        text="; ".join(parts),
                        section_path=heading_state.copy(),
                        element_id=element_id,
                        bbox=bbox,
                        headers=headers,
                        units=units_map or None,
                        span=None,
                        source_tool="docling",
                    )
                )

        # Process text items for this page
        for item in page_data["texts"]:
            prov = item.get("prov") or []
            if not prov:
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            bbox_dict = prov[0].get("bbox") or {}
            bbox = [bbox_dict.get(k) for k in ("l", "t", "r", "b")] if bbox_dict else None
            span = prov[0].get("charspan")
            label = item.get("label", "text")

            if label in {"section_header", "title", "heading"}:
                level = _infer_heading_level(text)
                heading_state = _update_heading_path(heading_state, text, level)
                element_id = _next_element_id(counters, "heading", page_num)
                blocks.append(
                    Block(
                        doc_id=doc_id,
                        path=str(path),
                        page=page_num,
                        type="heading",
                        text=text,
                        section_path=heading_state.copy(),
                        element_id=element_id,
                        bbox=bbox,
                        headers=None,
                        units=None,
                        span=span,
                        source_tool="docling",
                    )
                )
                continue

            block_type = "para"
            if "list" in label:
                block_type = "list"
            element_prefix = block_type if block_type != "para" else "para"
            element_id = _next_element_id(counters, element_prefix, page_num)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page_num,
                    type=block_type,
                    text=text,
                    section_path=heading_state.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=None,
                    units=None,
                    span=span,
                    source_tool="docling",
                )
            )

        # Process captions for this page
        for caption in page_data["captions"]:
            cap_text = (caption.get("text") or "").strip()
            if not cap_text:
                continue
            prov = caption.get("prov") or []
            bbox = None
            if prov:
                b = prov[0].get("bbox") or {}
                bbox = [b.get("l"), b.get("t"), b.get("r"), b.get("b")] if b else None
            element_id = _next_element_id(counters, "caption", page_num)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page_num,
                    type="caption",
                    text=cap_text,
                    section_path=heading_state.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=None,
                    units=None,
                    span=None,
                    source_tool="docling",
                )
            )

        # Process pictures for this page
        for picture in page_data["pictures"]:
            prov = picture.get("prov") or []
            if not prov:
                continue
            bbox_dict = prov[0].get("bbox") or {}
            bbox = [bbox_dict.get("l"), bbox_dict.get("t"), bbox_dict.get("r"), bbox_dict.get("b")] if bbox_dict else None
            element_id = _next_element_id(counters, "figure", page_num)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page_num,
                    type="figure",
                    text="",
                    section_path=heading_state.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=None,
                    units=None,
                    span=None,
                    source_tool="docling",
                )
            )

    metadata = {
        "total_pages": total_pages,
        "total_blocks": len(blocks),
        "extractor": "docling",
    }

    return blocks, metadata


def _docling_full_document_to_blocks(
    path: Path,
    doc_id: str,
) -> Tuple[List[Block], Dict[str, Any]]:
    """
    Process entire PDF with Docling in a single call.
    Returns all blocks and metadata for the document.
    """
    # Detect if OCR is needed and get file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    needs_ocr = _check_pdf_needs_ocr(path)

    converter = _get_docling_converter(needs_ocr=needs_ocr, file_size_mb=file_size_mb)

    print(f"Converting full document with Docling: {path} ({file_size_mb:.1f}MB)")
    try:
        result = converter.convert(str(path))
        doc_dict = result.document.export_to_dict()
    except Exception as exc:
        print(f"ERROR: Docling failed for {path}: {exc}")
        return [], {"error": str(exc), "total_pages": 0}

    return blocks_from_doc_dict(doc_dict, path, doc_id)


def _serialize_blocks(blocks: Iterable[Block]) -> List[Dict[str, Any]]:
    serialised = []
    for b in blocks:
        serialised.append(
            {
                "doc_id": b.doc_id,
                "path": b.path,
                "page": b.page,
                "type": b.type,
                "text": b.text,
                "section_path": b.section_path,
                "element_id": b.element_id,
                "bbox": b.bbox,
                "headers": b.headers,
                "units": b.units,
                "span": b.span,
                "source_tool": b.source_tool,
            }
        )
    return serialised


def _deserialize_blocks(items: Iterable[Dict[str, Any]]) -> List[Block]:
    blocks: List[Block] = []
    for it in items:
        blocks.append(
            Block(
                doc_id=it["doc_id"],
                path=it["path"],
                page=it["page"],
                type=it["type"],
                text=it["text"],
                section_path=it.get("section_path") or [],
                element_id=it.get("element_id", ""),
                bbox=it.get("bbox"),
                headers=it.get("headers"),
                units=it.get("units"),
                span=it.get("span"),
                source_tool=it.get("source_tool", "unknown"),
            )
        )
    return blocks


def extract_document_blocks(
    path: Path,
    doc_id: str,
    plan_override: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Block], Dict[str, Any]]:
    """
    Extract blocks from PDF using Docling only - no routing, no triage.
    Breaking change: always uses Docling for full document processing.
    """
    return _docling_full_document_to_blocks(path, doc_id)


def chunk_blocks(
    blocks: List[Block],
    max_chars: int,
    overlap_sentences: int = 1,
    profile: str = "heading_based",
) -> Tuple[List[Dict[str, Any]], str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    profile = (profile or "heading_based").lower()
    respect_headings = profile != "fixed_window"
    sentence_overlap = overlap_sentences if respect_headings else 0

    chunks: List[Dict[str, Any]] = []
    buffer: List[Block] = []
    chunk_cursor = 0
    raw_text_parts: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer, chunk_cursor
        if not buffer:
            return
        text = "\n\n".join(b.text for b in buffer if b.text)
        if not text.strip():
            buffer = []
            return
        pages = sorted({b.page for b in buffer})
        section_path = buffer[-1].section_path if buffer else []
        element_ids = [b.element_id for b in buffer if b.element_id]
        bboxes = [b.bbox for b in buffer]
        types = [b.type for b in buffer]
        source_tools = list({b.source_tool for b in buffer})
        headers: List[str] = []
        units: List[str] = []
        for b in buffer:
            if b.headers:
                headers.extend([str(h).strip() for h in b.headers if h])
            if b.units:
                if isinstance(b.units, dict):
                    units.extend(str(v).strip() for v in b.units.values() if v)
                elif isinstance(b.units, list):
                    units.extend(str(u).strip() for u in b.units if u)
                else:
                    units.append(str(b.units).strip())
        chunk = {
            "text": text,
            "pages": pages,
            "section_path": section_path,
            "element_ids": element_ids,
            "bboxes": bboxes,
            "types": types,
            "source_tools": source_tools,
            "headers": headers,
            "table_headers": headers,
            "units": units,
            "table_units": units,
            "chunk_start": chunk_cursor,
            "chunk_end": chunk_cursor + len(text),
            "doc_id": buffer[0].doc_id,
            "path": buffer[0].path,
            "profile": profile,
        }
        chunks.append(chunk)
        chunk_cursor += len(text)
        if sentence_overlap > 0:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            tail = " ".join(sentences[-sentence_overlap:]).strip()
            buffer = [
                Block(
                    doc_id=buffer[-1].doc_id,
                    path=buffer[-1].path,
                    page=buffer[-1].page,
                    type="para",
                    text=tail,
                    section_path=section_path,
                    element_id="",
                    bbox=None,
                    headers=None,
                    units=None,
                    span=None,
                    source_tool=buffer[-1].source_tool,
                )
            ] if tail else []
        else:
            buffer = []

    for block in blocks:
        raw_text_parts.append(block.text)
        if profile == "table_row" and block.type == "table_row":
            flush_buffer()
            text = block.text or ""
            if not text.strip():
                continue
            row_headers = [str(h).strip() for h in (block.headers or []) if h]
            row_units: List[str] = []
            if block.units:
                if isinstance(block.units, dict):
                    row_units.extend(str(v).strip() for v in block.units.values() if v)
                elif isinstance(block.units, list):
                    row_units.extend(str(u).strip() for u in block.units if u)
                else:
                    row_units.append(str(block.units).strip())
            chunk = {
                "text": text,
                "pages": [block.page],
                "section_path": block.section_path,
                "element_ids": [block.element_id] if block.element_id else [],
                "bboxes": [block.bbox] if block.bbox else [],
                "types": [block.type],
                "source_tools": [block.source_tool] if block.source_tool else [],
                "headers": row_headers,
                "table_headers": row_headers,
                "units": row_units,
                "table_units": row_units,
                "chunk_start": chunk_cursor,
                "chunk_end": chunk_cursor + len(text),
                "doc_id": block.doc_id,
                "path": block.path,
                "profile": profile,
            }
            chunks.append(chunk)
            chunk_cursor += len(text)
            continue

        if respect_headings and block.type == "heading":
            flush_buffer()
            buffer = [block]
            flush_buffer()
            buffer = []
            continue

        buffer.append(block)
        total_chars = sum(len(b.text) for b in buffer)
        if total_chars >= max_chars:
            flush_buffer()

    flush_buffer()
    raw_text = "\n".join(raw_text_parts)
    return chunks, raw_text
