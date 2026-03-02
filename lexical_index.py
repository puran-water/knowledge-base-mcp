import os
import sqlite3
import time
from collections import defaultdict
from typing import Iterable, Dict, Any, List, Optional, Tuple

ALIASES = {
    "MBR": ["membrane bioreactor"],
    "MLSS": ["mixed liquor suspended solids"],
    "SBR": ["sequencing batch reactor"],
    "BOD": ["biochemical oxygen demand"],
    "COD": ["chemical oxygen demand"],
    "P&ID": ["piping and instrumentation diagram", "PID"],
    "PLC": ["programmable logic controller"],
    "DAF": ["dissolved air flotation"],
    "RO": ["reverse osmosis"],
}

FTS_EXPECTED_UNINDEXED = {
    "chunk_id": "UNINDEXED",
    "doc_id": "UNINDEXED",
    "path": "UNINDEXED",
    "filename": "UNINDEXED",
    "source_path": "UNINDEXED",
    "chunk_start": "UNINDEXED",
    "chunk_end": "UNINDEXED",
    "mtime": "UNINDEXED",
    "page_numbers": "UNINDEXED",
    "pages": "UNINDEXED",
    "section_path": "UNINDEXED",
    "element_ids": "UNINDEXED",
    "bboxes": "UNINDEXED",
    "types": "UNINDEXED",
    "source_tools": "UNINDEXED",
    "table_headers": "UNINDEXED",
    "table_units": "UNINDEXED",
    "chunk_profile": "UNINDEXED",
    "plan_hash": "UNINDEXED",
    "model_version": "UNINDEXED",
    "prompt_sha": "UNINDEXED",
    "doc_metadata": "UNINDEXED",
}


def expand_query(query: str) -> str:
    expanded = [query]
    lower_q = query.lower()
    for key, synonyms in ALIASES.items():
        key_lower = key.lower()
        if key_lower in lower_q or any(syn.lower() in lower_q for syn in synonyms):
            terms = [key] + synonyms
            clause = " OR ".join(f"\"{term}\"" for term in terms)
            expanded.append(f"({clause})")
    if len(expanded) == 1:
        return query
    return " OR ".join(expanded)


FTS_CREATE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    text,
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    path UNINDEXED,
    filename UNINDEXED,
    source_path UNINDEXED,
    chunk_start UNINDEXED,
    chunk_end UNINDEXED,
    mtime UNINDEXED,
    page_numbers UNINDEXED,
    pages UNINDEXED,
    section_path UNINDEXED,
    element_ids UNINDEXED,
    bboxes UNINDEXED,
    types UNINDEXED,
    source_tools UNINDEXED,
    table_headers UNINDEXED,
    table_units UNINDEXED,
    chunk_profile UNINDEXED,
    plan_hash UNINDEXED,
    model_version UNINDEXED,
    prompt_sha UNINDEXED,
    doc_metadata UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

SPARSE_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS fts_chunks_sparse (
    term TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    doc_id TEXT,
    weight REAL NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
"""

SPARSE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_sparse_terms_term ON fts_chunks_sparse(term);"
SPARSE_CHUNK_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_sparse_terms_chunk ON fts_chunks_sparse(chunk_id);"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks'")
    exists = cur.fetchone() is not None
    if not exists:
        conn.execute(FTS_CREATE_SQL)
        conn.execute(SPARSE_CREATE_SQL)
        conn.execute(SPARSE_INDEX_SQL)
        conn.execute(SPARSE_CHUNK_INDEX_SQL)
        conn.commit()
        return
    # Existing table: check for missing columns.
    # NOTE: FTS5 virtual tables do NOT support ALTER TABLE ADD COLUMN.
    # If columns are missing, warn and suggest --fts-rebuild for full re-ingestion.
    current_columns = set()
    for row in conn.execute("PRAGMA table_xinfo('fts_chunks')"):
        name = row[1]
        if name:
            current_columns.add(name)
    missing = [col for col in FTS_EXPECTED_UNINDEXED if col not in current_columns]
    if missing:
        import sys
        print(
            f"WARN: FTS table missing columns: {missing}. "
            "FTS5 virtual tables cannot be altered. Use --fts-rebuild to recreate with full schema.",
            file=sys.stderr,
        )
    conn.execute(SPARSE_CREATE_SQL)
    conn.execute(SPARSE_INDEX_SQL)
    conn.execute(SPARSE_CHUNK_INDEX_SQL)
    conn.commit()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def ensure_fts(db_path: str) -> None:
    _ensure_parent_dir(db_path)
    last_err = None
    conn = None
    for attempt in range(3):
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            _ensure_schema(conn)
            return
        except sqlite3.OperationalError as ex:
            last_err = ex
            if "unable to open database file" in str(ex).lower():
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
        finally:
            if conn is not None:
                conn.close()
    # If we reach here, retries exhausted
    raise last_err


class FTSWriter:
    """Single-connection FTS writer for reliable, fast bulk upserts.

    Use as a context manager:

        with FTSWriter(path, recreate=True) as w:
            w.upsert_many(rows)
    """

    def __init__(self, db_path: str, recreate: bool = False):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        _ensure_parent_dir(db_path)
        # Open with retries
        last_err = None
        for attempt in range(3):
            try:
                self.conn = sqlite3.connect(db_path)
                self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA busy_timeout=5000;")
                if recreate:
                    cur = self.conn.cursor()
                    cur.execute("DROP TABLE IF EXISTS fts_chunks")
                    self.conn.commit()
                _ensure_schema(self.conn)
                break
            except sqlite3.OperationalError as ex:
                last_err = ex
                if "unable to open database file" in str(ex).lower():
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        if self.conn is None:
            raise last_err

        cur = self.conn.cursor()
        if recreate:
            cur.execute("DROP TABLE IF EXISTS fts_chunks")
            cur.execute("DROP TABLE IF EXISTS fts_chunks_sparse")
        cur.execute(FTS_CREATE_SQL)
        cur.execute(SPARSE_CREATE_SQL)
        cur.execute(SPARSE_INDEX_SQL)
        cur.execute(SPARSE_CHUNK_INDEX_SQL)
        self.conn.commit()

    def _get_fts_columns(self) -> list:
        """Return the actual columns in the fts_chunks table (cached)."""
        if not hasattr(self, "_fts_columns"):
            cols = []
            for row in self.conn.execute("PRAGMA table_xinfo('fts_chunks')"):
                name = row[1]
                if name:
                    cols.append(name)
            self._fts_columns = cols
        return self._fts_columns

    def upsert_many(self, rows: Iterable[Dict[str, Any]], batch_size: int = 2000, show_progress: bool = False) -> int:
        """
        Upsert chunks with batched commits to prevent WAL bloat.

        Args:
            rows: Chunk rows to upsert
            batch_size: Number of rows per commit (default 2000, recommended for large docs)
            show_progress: Show progress bar (useful for large ingestions)

        Returns:
            Total number of rows upserted
        """
        rows = [r for r in rows if r.get("text")]
        if not rows:
            return 0

        # Determine actual columns in FTS table for graceful fallback on old schemas
        actual_cols = self._get_fts_columns()
        full_cols = [
            "text", "chunk_id", "doc_id", "path", "filename", "source_path",
            "chunk_start", "chunk_end", "mtime", "page_numbers",
            "pages", "section_path", "element_ids", "bboxes",
            "types", "source_tools", "table_headers", "table_units",
            "chunk_profile", "plan_hash", "model_version", "prompt_sha", "doc_metadata",
        ]
        # Only insert columns that exist in the table (handles pre-rebuild DBs)
        insert_cols = [c for c in full_cols if c in actual_cols]
        placeholders = ", ".join("?" for _ in insert_cols)
        col_list = ", ".join(insert_cols)
        insert_sql = f"INSERT INTO fts_chunks ({col_list}) VALUES ({placeholders})"

        cur = self.conn.cursor()
        total_inserted = 0

        # Process in batches with intermediate commits
        import sys
        for batch_start in range(0, len(rows), batch_size):
            batch_end = min(batch_start + batch_size, len(rows))
            batch = rows[batch_start:batch_end]

            # Delete existing by chunk_id — use batched IN clause instead of
            # executemany to avoid O(N * table_size) full table scans.
            # FTS5 virtual tables have no B-tree index on non-rowid columns,
            # so each individual DELETE scans the entire content table.
            # A single IN clause does ONE scan checking against a hash set.
            chunk_ids_to_delete = [str(r.get("chunk_id")) for r in batch]
            _IN_CHUNK = 500  # stay under SQLITE_MAX_VARIABLE_NUMBER
            for _di in range(0, len(chunk_ids_to_delete), _IN_CHUNK):
                _sub = chunk_ids_to_delete[_di : _di + _IN_CHUNK]
                _ph = ",".join("?" * len(_sub))
                cur.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({_ph})", _sub)

            # Insert batch — extract values for only the columns that exist
            def _row_val(r, col):
                if col in ("chunk_start", "chunk_end", "mtime"):
                    return int(r.get(col, 0) or 0)
                return str(r.get(col, "") or "")

            cur.executemany(
                insert_sql,
                [tuple(_row_val(r, c) for c in insert_cols) for r in batch],
            )

            # Process sparse terms for this batch
            sparse_delete = set()
            sparse_rows: List[Tuple[str, str, str, float]] = []
            for r in batch:
                chunk_id = str(r.get("chunk_id"))
                doc_id = str(r.get("doc_id"))
                terms = r.get("sparse_terms") or []
                if not terms:
                    continue
                sparse_delete.add(chunk_id)
                for entry in terms:
                    if isinstance(entry, dict):
                        term = entry.get("term")
                        weight = entry.get("weight")
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        term, weight = entry[0], entry[1]
                    else:
                        continue
                    if not term:
                        continue
                    try:
                        w = float(weight)
                    except Exception:
                        continue
                    term_norm = str(term).strip().lower()
                    if len(term_norm) < 3:
                        continue
                    sparse_rows.append((term_norm, chunk_id, doc_id, w))

            if sparse_delete:
                cur.executemany("DELETE FROM fts_chunks_sparse WHERE chunk_id = ?", [(cid,) for cid in sparse_delete])
            if sparse_rows:
                cur.executemany(
                    "INSERT OR REPLACE INTO fts_chunks_sparse(term, chunk_id, doc_id, weight) VALUES(?,?,?,?)",
                    sparse_rows,
                )

            # Commit this batch to bound WAL growth
            self.conn.commit()
            total_inserted += len(batch)

            # Progress feedback
            if show_progress:
                print(f"  FTS upserted {total_inserted}/{len(rows)} chunks", file=sys.stderr, flush=True)

        return total_inserted

    def delete_doc(self, doc_id: str, commit: bool = True) -> None:
        if not doc_id:
            return
        cur = self.conn.cursor()
        cur.execute("DELETE FROM fts_chunks WHERE doc_id = ?", (str(doc_id),))
        cur.execute("DELETE FROM fts_chunks_sparse WHERE doc_id = ?", (str(doc_id),))
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        """Explicit commit for batched operations."""
        if self.conn is not None:
            self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def __enter__(self) -> "FTSWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def upsert_chunks(db_path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Upsert chunk rows into FTS, identified by chunk_id. Returns inserted count.

    Expected row keys: text, chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime
    """
    # Ensure DB exists and open with simple retries to avoid transient
    # "unable to open database file" issues on some systems.
    ensure_fts(db_path)
    last_err = None
    conn = None
    for attempt in range(3):
        try:
            conn = sqlite3.connect(db_path)
            # Improve resilience on busy filesystems
            conn.execute("PRAGMA busy_timeout=5000;")
            break
        except sqlite3.OperationalError as ex:
            last_err = ex
            # Retry only for open failures
            if "unable to open database file" in str(ex).lower():
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    if conn is None:
        # Re-raise the last error if we failed all attempts
        raise last_err
    try:
        cur = conn.cursor()
        count = 0
        cur.execute("BEGIN")
        sparse_delete = set()
        sparse_rows: List[Tuple[str, str, str, float]] = []
        for r in rows:
            if not r.get("text"):
                continue
            chunk_id = str(r.get("chunk_id"))
            doc_id = str(r.get("doc_id"))
            # Delete any existing entry for this chunk_id, then insert
            cur.execute("DELETE FROM fts_chunks WHERE chunk_id = ?", (chunk_id,))
            cur.execute(
                """
                INSERT INTO fts_chunks (
                    text, chunk_id, doc_id, path, filename, source_path,
                    chunk_start, chunk_end, mtime, page_numbers,
                    pages, section_path, element_ids, bboxes,
                    types, source_tools, table_headers, table_units,
                    chunk_profile, plan_hash, model_version, prompt_sha, doc_metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("text", ""),
                    chunk_id,
                    doc_id,
                    str(r.get("path")),
                    str(r.get("filename")),
                    str(r.get("source_path", "") or ""),
                    int(r.get("chunk_start", 0) or 0),
                    int(r.get("chunk_end", 0) or 0),
                    int(r.get("mtime", 0) or 0),
                    str(r.get("page_numbers", "") or ""),
                    str(r.get("pages", "") or ""),
                    str(r.get("section_path", "") or ""),
                    str(r.get("element_ids", "") or ""),
                    str(r.get("bboxes", "") or ""),
                    str(r.get("types", "") or ""),
                    str(r.get("source_tools", "") or ""),
                    str(r.get("table_headers", "") or ""),
                    str(r.get("table_units", "") or ""),
                    str(r.get("chunk_profile", "") or ""),
                    str(r.get("plan_hash", "") or ""),
                    str(r.get("model_version", "") or ""),
                    str(r.get("prompt_sha", "") or ""),
                    str(r.get("doc_metadata", "") or ""),
                ),
            )
            terms = r.get("sparse_terms") or []
            if terms:
                sparse_delete.add(chunk_id)
                for entry in terms:
                    if isinstance(entry, dict):
                        term = entry.get("term")
                        weight = entry.get("weight")
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        term, weight = entry[0], entry[1]
                    else:
                        continue
                    if not term:
                        continue
                    try:
                        w = float(weight)
                    except Exception:
                        continue
                    term_norm = str(term).strip().lower()
                    if len(term_norm) < 3:
                        continue
                    sparse_rows.append((term_norm, chunk_id, doc_id, w))
            count += 1
        if sparse_delete:
            cur.executemany("DELETE FROM fts_chunks_sparse WHERE chunk_id = ?", [(cid,) for cid in sparse_delete])
        if sparse_rows:
            cur.executemany(
                "INSERT OR REPLACE INTO fts_chunks_sparse(term, chunk_id, doc_id, weight) VALUES(?,?,?,?)",
                sparse_rows,
            )
        conn.commit()
        return count
    finally:
        conn.close()


def delete_doc(db_path: str, doc_id: str) -> None:
    """Remove all FTS rows for a given doc_id."""
    if not doc_id:
        return
    ensure_fts(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM fts_chunks WHERE doc_id = ?", (str(doc_id),))
        cur.execute("DELETE FROM fts_chunks_sparse WHERE doc_id = ?", (str(doc_id),))
        conn.commit()
    finally:
        conn.close()


def search(db_path: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search the FTS index with BM25 ranking. Returns list of rows."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        fts_query = expand_query(query)
        # Order by bm25() ASC for best first
        cur.execute(
            """
            SELECT chunk_id, doc_id, path, filename, source_path, chunk_start, chunk_end, mtime, page_numbers,
                   pages, section_path, element_ids, bboxes, types, source_tools,
                   table_headers, table_units, chunk_profile, plan_hash, model_version, prompt_sha, doc_metadata,
                   text,
                   bm25(fts_chunks) AS bm25
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
            ORDER BY bm25 LIMIT ?
            """,
            (fts_query, int(limit)),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def sparse_search(db_path: str, query_terms: Dict[str, float], limit: int = 20) -> List[Dict[str, Any]]:
    if not query_terms or not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        terms = list({term.lower(): weight for term, weight in query_terms.items()}.items())
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        cur.execute(
            f"SELECT term, chunk_id, doc_id, weight FROM fts_chunks_sparse WHERE term IN ({placeholders})",
            [t[0] for t in terms],
        )
        q_weights = {term: weight for term, weight in terms}
        scores: Dict[str, float] = defaultdict(float)
        doc_ids: Dict[str, str] = {}
        for row in cur.fetchall():
            term = row["term"]
            chunk_id = row["chunk_id"]
            doc_id = row["doc_id"]
            weight = row["weight"]
            q_weight = q_weights.get(term)
            if q_weight is None:
                continue
            scores[chunk_id] += float(weight) * float(q_weight)
            if doc_id:
                doc_ids.setdefault(chunk_id, doc_id)
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: int(limit)]
        chunk_map = fetch_texts_by_chunk_ids(db_path, [cid for cid, _ in top])
        results: List[Dict[str, Any]] = []
        for chunk_id, score in top:
            row = chunk_map.get(chunk_id, {"chunk_id": chunk_id, "doc_id": doc_ids.get(chunk_id)})
            data = dict(row)
            data["sparse_score"] = score
            results.append(data)
        return results
    finally:
        conn.close()


def fetch_texts_by_chunk_ids(db_path: str, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch text and metadata for chunk_ids from FTS."""
    ids = [str(cid) for cid in chunk_ids if cid]
    if not ids or not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in ids)
        cur.execute(
            f"""
            SELECT chunk_id, text, doc_id, path, filename, source_path, chunk_start, chunk_end, page_numbers, mtime,
                   pages, section_path, element_ids, bboxes, types, source_tools,
                   table_headers, table_units, chunk_profile, plan_hash, model_version, prompt_sha, doc_metadata
            FROM fts_chunks
            WHERE chunk_id IN ({placeholders})
            """,
            ids,
        )
        rows = cur.fetchall()
        return {str(r["chunk_id"]): dict(r) for r in rows}
    finally:
        conn.close()
