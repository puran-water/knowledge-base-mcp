#!/usr/bin/env python3
"""
Graph Bloat Cleanup Script v2 - Improved based on Codex review

Critical fixes:
1. Accurate doc count (uses type='doc' instead of DISTINCT doc_id)
2. Single-pass statistics using temp tables (80k+ queries → 1 query)
3. Batched deletions to avoid SQLite parameter limit (999)
4. Better TF-IDF scoring (tracks both max and mean)
5. Improved filtering logic with structural signals

New optional cleanup passes:
6. Stopword entity removal (--remove-stopwords)
7. Measurement deduplication (--dedupe-measurements)

Usage:
    # Basic cleanup (co-occurrence + TF-IDF filtering)
    python scripts/cleanup_graph_bloat_v2.py [--dry-run] [--backup] [--collection COLLECTION]

    # With additional stopword and measurement cleanup
    python scripts/cleanup_graph_bloat_v2.py --remove-stopwords --dedupe-measurements [--collection COLLECTION]

    # Dry-run to preview changes
    python scripts/cleanup_graph_bloat_v2.py --dry-run --remove-stopwords --dedupe-measurements
"""

import argparse
import sqlite3
import shutil
import sys
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from datetime import datetime


# Low-value entity patterns
JUNK_PATTERNS = [
    r"^isbn[:\s-]*\d",
    r"^\d{3,}$",
    r"^doi:",
    r"^copyright\s+\d{4}",
    r"^all rights reserved",
    r"^printed in",
    r"^published by",
    r"remains neutral",
    r"including drinking",
    r"^page \d+$",
    r"^chapter \d+$",
    r"^section \d+",
    r"^figure \d+",
    r"^table \d+",
    r"^appendix [a-z]$",
]

# SQLite parameter limit
MAX_SQL_PARAMS = 900


def get_db_stats(conn: sqlite3.Connection) -> Dict:
    """Get current database statistics."""
    c = conn.cursor()
    stats = {}
    stats['total_nodes'] = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    stats['total_edges'] = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    node_types = c.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall()
    stats['nodes_by_type'] = {t: cnt for t, cnt in node_types}
    edge_types = c.execute("SELECT type, COUNT(*) FROM edges GROUP BY type").fetchall()
    stats['edges_by_type'] = {t: cnt for t, cnt in edge_types}
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    stats['file_size_mb'] = Path(db_path).stat().st_size / (1024 * 1024)
    return stats


def print_stats(stats: Dict, label: str = "Current Statistics"):
    """Pretty print database statistics."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"File size: {stats['file_size_mb']:.1f} MB")
    print(f"Total nodes: {stats['total_nodes']:,}")
    print(f"Total edges: {stats['total_edges']:,}")
    print("\nNodes by type:")
    for node_type, count in sorted(stats['nodes_by_type'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
        print(f"  {node_type:20s}: {count:7,} ({pct:5.1f}%)")
    print("\nEdges by type:")
    for edge_type, count in sorted(stats['edges_by_type'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_edges'] if stats['total_edges'] > 0 else 0
        print(f"  {edge_type:20s}: {count:7,} ({pct:5.1f}%)")
    print()


def is_junk_entity(label: str) -> bool:
    """Check if entity matches junk patterns."""
    label_lower = label.lower().strip()
    if len(label_lower) < 2:
        return True

    # Check hardcoded junk patterns
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, label_lower):
            return True

    # Statistical heuristics (Codex recommendation)
    # High digit ratio (unless it's a measurement-related entity)
    digit_count = sum(c.isdigit() for c in label)
    alpha_count = sum(c.isalpha() for c in label)
    total_chars = len(label.replace(' ', ''))

    if total_chars > 0:
        digit_ratio = digit_count / total_chars
        alpha_ratio = alpha_count / total_chars

        # Very high digit ratio (>90%) or very low alpha ratio (<40%)
        if digit_ratio > 0.9 or alpha_ratio < 0.4:
            return True

    return False


def remove_cooccurrence_edges(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Remove all co-occurrence edges (N² bloat source)."""
    c = conn.cursor()
    cooccur_count = c.execute("SELECT COUNT(*) FROM edges WHERE type='co_occurs'").fetchone()[0]
    if cooccur_count == 0:
        print("No co-occurrence edges found.")
        return 0
    print(f"\n[1/5] Removing {cooccur_count:,} co-occurrence edges...")
    if not dry_run:
        c.execute("DELETE FROM edges WHERE type='co_occurs'")
        conn.commit()
        print(f"  ✓ Deleted {cooccur_count:,} co-occurrence edges")
    else:
        print(f"  [DRY RUN] Would delete {cooccur_count:,} edges")
    return cooccur_count


def deduplicate_mention_edges(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """
    Deduplicate mention edges caused by element_id loop bug.
    Only processes duplicates (Codex recommendation).
    """
    c = conn.cursor()
    print("\n[2/5] Deduplicating mention edges (element_id loop bug fix)...")

    # Find duplicates
    duplicates = c.execute("""
        SELECT src, dst, type, collection, doc_id, COUNT(*) as cnt
        FROM edges
        WHERE type = 'mentions'
        GROUP BY src, dst, type, collection, doc_id
        HAVING cnt > 1
    """).fetchall()

    if not duplicates:
        print("  No duplicate mention edges found.")
        return 0

    total_duplicate_edges = sum(cnt - 1 for *_, cnt in duplicates)
    print(f"  Found {len(duplicates):,} unique edges with {total_duplicate_edges:,} duplicates")

    if not dry_run:
        # Keep only first occurrence of each unique edge
        # Delete by rowid to avoid reprocessing entire edge table
        c.execute("""
            DELETE FROM edges
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM edges
                WHERE type = 'mentions'
                GROUP BY src, dst, type, collection, doc_id
            )
            AND type = 'mentions'
        """)
        conn.commit()
        print(f"  ✓ Removed {total_duplicate_edges:,} duplicate mention edges")
    else:
        print(f"  [DRY RUN] Would remove {total_duplicate_edges:,} duplicates")

    return total_duplicate_edges


def compute_entity_stats_fast(conn: sqlite3.Connection, collection: str = None) -> Tuple[Dict, int]:
    """
    Compute TF-IDF statistics using single-pass aggregation (Codex fix).
    Uses temp table instead of 80k+ individual queries.
    """
    c = conn.cursor()
    print("\n[3/5] Computing entity statistics (optimized single-pass)...")

    # Count total documents CORRECTLY (Codex fix)
    # Use type='doc' instead of DISTINCT doc_id
    if collection:
        total_docs = c.execute("SELECT COUNT(*) FROM nodes WHERE type='doc' AND collection=?", (collection,)).fetchone()[0]
    else:
        total_docs = c.execute("SELECT COUNT(*) FROM nodes WHERE type='doc'").fetchone()[0]

    print(f"  Total documents: {total_docs:,}")

    # Get entity labels
    if collection:
        entities = {eid: label for eid, label in c.execute(
            "SELECT id, label FROM nodes WHERE type='entity' AND collection=?", (collection,)
        ).fetchall()}
    else:
        entities = {eid: label for eid, label in c.execute(
            "SELECT id, label FROM nodes WHERE type='entity'"
        ).fetchall()}

    print(f"  Analyzing {len(entities):,} entities...")

    # Single-pass aggregation query (Codex recommendation)
    # Use temp table for entity IDs to avoid IN clause parameter limit (Codex fix #2)
    print("  Building statistics temp table...")

    # Create temp table with entity IDs
    c.execute("CREATE TEMP TABLE entity_ids_filter(id TEXT PRIMARY KEY)")
    entity_list = list(entities.keys())
    for i in range(0, len(entity_list), MAX_SQL_PARAMS):
        batch = entity_list[i:i + MAX_SQL_PARAMS]
        c.execute(f"INSERT INTO entity_ids_filter VALUES {','.join('(?)' for _ in batch)}", batch)

    # Build mention stats using JOIN instead of IN
    collection_filter = "AND e.collection=?" if collection else ""
    params = [collection] if collection else []

    c.execute(f"""
        CREATE TEMP TABLE entity_mention_stats AS
        SELECT
            e.dst AS entity_id,
            e.doc_id,
            COUNT(*) AS tf
        FROM edges e
        JOIN entity_ids_filter f ON e.dst = f.id
        WHERE e.type='mentions' {collection_filter}
        GROUP BY e.dst, e.doc_id
    """, params)

    c.execute("DROP TABLE entity_ids_filter")

    # Aggregate per-entity statistics
    print("  Aggregating entity statistics...")
    entity_stats_raw = c.execute("""
        SELECT
            entity_id,
            COUNT(DISTINCT doc_id) AS doc_count,
            SUM(tf) AS total_mentions,
            MAX(tf) AS max_tf,
            AVG(tf) AS mean_tf
        FROM entity_mention_stats
        GROUP BY entity_id
    """).fetchall()

    # Compute TF-IDF scores (track both max and mean as Codex recommended)
    entity_stats = {}
    for entity_id, doc_count, total_mentions, max_tf, mean_tf in entity_stats_raw:
        if entity_id not in entities:
            continue

        # IDF using correct doc count
        idf = math.log((total_docs + 1) / (doc_count + 1))

        # Compute both max and mean TF-IDF (Codex recommendation)
        tfidf_max = math.log1p(max_tf) * idf
        tfidf_mean = math.log1p(mean_tf) * idf

        # Blended score: 70% max, 30% mean
        tfidf = 0.7 * tfidf_max + 0.3 * tfidf_mean

        entity_stats[entity_id] = {
            'label': entities[entity_id],
            'doc_count': doc_count,
            'total_mentions': total_mentions,
            'max_tf': max_tf,
            'mean_tf': mean_tf,
            'tfidf_max': tfidf_max,
            'tfidf_mean': tfidf_mean,
            'tfidf': tfidf,
        }

    # Clean up temp table
    c.execute("DROP TABLE entity_mention_stats")

    print(f"  ✓ Computed statistics for {len(entity_stats):,} entities")
    return entity_stats, total_docs


def get_structural_signals(conn: sqlite3.Connection, entity_ids: Set[str]) -> Dict[str, Dict]:
    """
    Get structural signals for entities (participation in special relations).
    Codex recommendation: boost quality for entities with structural importance.

    CRITICAL FIX: Check dst not src (entities are TARGETS of these relations).
    """
    if not entity_ids:
        return {}

    c = conn.cursor()
    print("  Analyzing structural signals...")

    # Create temp table to avoid IN clause parameter limit (Codex fix)
    c.execute("CREATE TEMP TABLE entity_targets(id TEXT PRIMARY KEY)")
    entity_list = list(entity_ids)
    for i in range(0, len(entity_list), MAX_SQL_PARAMS):
        batch = entity_list[i:i + MAX_SQL_PARAMS]
        placeholders = ','.join('?' * len(batch))
        c.execute(f"INSERT INTO entity_targets VALUES {','.join('(?)' for _ in batch)}", batch)

    structural_flags = {eid: {'has_measurement': False, 'has_parameter': False} for eid in entity_ids}

    # CRITICAL FIX: Check dst not src (Codex caught this bug)
    # Entities are TARGETS of has_measurement and row_has_parameter relations
    measurement_entities = c.execute("""
        SELECT DISTINCT dst FROM edges
        WHERE type IN ('has_measurement', 'measures')
          AND dst IN (SELECT id FROM entity_targets)
    """).fetchall()

    for (eid,) in measurement_entities:
        structural_flags[eid]['has_measurement'] = True

    parameter_entities = c.execute("""
        SELECT DISTINCT dst FROM edges
        WHERE type = 'row_has_parameter'
          AND dst IN (SELECT id FROM entity_targets)
    """).fetchall()

    for (eid,) in parameter_entities:
        structural_flags[eid]['has_parameter'] = True

    c.execute("DROP TABLE entity_targets")

    return structural_flags


def filter_entities_improved(
    entity_stats: Dict,
    total_docs: int,
    structural_signals: Dict,
    min_doc_freq: int = 2,
    min_tfidf: float = 0.3,
    keep_top_pct: float = 90,
) -> Set[str]:
    """
    Improved filtering logic (Codex recommendations):
    - Hard delete for junk patterns
    - Require BOTH low frequency AND low TF-IDF for deletion
    - Protect entities with structural importance
    - Apply percentile cut per df bucket
    """
    print("\n[4/5] Filtering low-value entities (improved logic)...")

    to_delete = set()
    protected = set()

    # Step 1: Hard delete junk patterns
    junk_count = 0
    for entity_id, stats in entity_stats.items():
        if is_junk_entity(stats['label']):
            to_delete.add(entity_id)
            junk_count += 1

    print(f"  [1/4] Pattern-based junk: {junk_count:,} entities")

    # Step 2: Protect structurally important entities
    for entity_id, flags in structural_signals.items():
        if flags['has_measurement'] or flags['has_parameter']:
            protected.add(entity_id)

    print(f"  [2/4] Protected (structural signals): {len(protected):,} entities")

    # Step 3: Filter by BOTH low frequency AND low TF-IDF (Codex: AND not OR)
    remaining = {eid: stats for eid, stats in entity_stats.items()
                 if eid not in to_delete and eid not in protected}

    low_quality = {eid for eid, stats in remaining.items()
                   if stats['doc_count'] < min_doc_freq and stats['tfidf'] < min_tfidf}
    to_delete.update(low_quality)

    print(f"  [3/4] Low quality (df < {min_doc_freq} AND tfidf < {min_tfidf:.2f}): {len(low_quality):,} entities")

    # Step 4: Percentile cut per df bucket (Codex recommendation)
    remaining = {eid: stats for eid, stats in remaining.items() if eid not in to_delete}

    # Group by df buckets
    df_buckets = {
        'df_1': [],
        'df_2_5': [],
        'df_6_plus': []
    }

    for eid, stats in remaining.items():
        df = stats['doc_count']
        if df == 1:
            df_buckets['df_1'].append((eid, stats))
        elif 2 <= df <= 5:
            df_buckets['df_2_5'].append((eid, stats))
        else:
            df_buckets['df_6_plus'].append((eid, stats))

    # Apply percentile cut within each bucket
    bucket_deletions = 0
    for bucket_name, bucket_items in df_buckets.items():
        if not bucket_items:
            continue

        sorted_bucket = sorted(bucket_items, key=lambda x: x[1]['tfidf'], reverse=True)
        keep_count = int(len(sorted_bucket) * keep_top_pct / 100)
        bottom_entities = {eid for eid, _ in sorted_bucket[keep_count:]}
        to_delete.update(bottom_entities)
        bucket_deletions += len(bottom_entities)

    print(f"  [4/4] Bottom {100-keep_top_pct}% by TF-IDF (per df bucket): {bucket_deletions:,} entities")

    print(f"\n  Total to delete: {len(to_delete):,} / {len(entity_stats):,} ({100*len(to_delete)/len(entity_stats):.1f}%)")
    print(f"  Total to keep: {len(entity_stats) - len(to_delete):,} ({100*(1-len(to_delete)/len(entity_stats)):.1f}%)")

    return to_delete


def delete_orphan_entities(conn: sqlite3.Connection, collection: str = None, dry_run: bool = False) -> int:
    """
    Delete entities with no mention edges (Codex recommendation).
    These 2k+ entities contribute nothing to retrieval but add storage.
    """
    c = conn.cursor()
    print("\n  Deleting orphan entities (no mention edges)...")

    # Find entities with no mentions
    if collection:
        orphan_entities = c.execute("""
            SELECT id FROM nodes
            WHERE type = 'entity'
              AND collection = ?
              AND id NOT IN (SELECT DISTINCT dst FROM edges WHERE type = 'mentions')
        """, (collection,)).fetchall()
    else:
        orphan_entities = c.execute("""
            SELECT id FROM nodes
            WHERE type = 'entity'
              AND id NOT IN (SELECT DISTINCT dst FROM edges WHERE type = 'mentions')
        """).fetchall()

    orphan_ids = [oid for (oid,) in orphan_entities]

    if not orphan_ids:
        print("    No orphan entities found.")
        return 0

    print(f"    Found {len(orphan_ids):,} orphan entities")

    if not dry_run:
        # Delete in batches
        for i in range(0, len(orphan_ids), MAX_SQL_PARAMS):
            batch = orphan_ids[i:i + MAX_SQL_PARAMS]
            placeholders = ','.join('?' * len(batch))
            c.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", batch)
        conn.commit()
        print(f"    ✓ Deleted {len(orphan_ids):,} orphan entities")
    else:
        print(f"    [DRY RUN] Would delete {len(orphan_ids):,} orphan entities")

    return len(orphan_ids)


def delete_entities_and_edges_batched(conn: sqlite3.Connection, entity_ids: Set[str], dry_run: bool = False) -> Tuple[int, int]:
    """
    Delete entities and edges in batches (Codex fix for SQLite parameter limit).
    Uses temp table for accurate edge counting (Codex fix #3).
    """
    if not entity_ids:
        return 0, 0

    c = conn.cursor()

    # Use temp table for accurate edge counting (Codex fix)
    c.execute("CREATE TEMP TABLE entities_to_delete(id TEXT PRIMARY KEY)")
    entity_list = list(entity_ids)
    for i in range(0, len(entity_list), MAX_SQL_PARAMS):
        batch = entity_list[i:i + MAX_SQL_PARAMS]
        c.execute(f"INSERT INTO entities_to_delete VALUES {','.join('(?)' for _ in batch)}", batch)

    # Count edges once using temp table
    edge_count = c.execute("""
        SELECT COUNT(*) FROM edges
        WHERE src IN (SELECT id FROM entities_to_delete)
           OR dst IN (SELECT id FROM entities_to_delete)
    """).fetchone()[0]

    print(f"\n  Deleting {len(entity_ids):,} entity nodes and {edge_count:,} associated edges...")

    if not dry_run:
        # Delete edges
        c.execute("""
            DELETE FROM edges
            WHERE src IN (SELECT id FROM entities_to_delete)
               OR dst IN (SELECT id FROM entities_to_delete)
        """)

        # Delete nodes
        c.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM entities_to_delete)")

        conn.commit()
        print(f"  ✓ Deleted {len(entity_ids):,} entities and {edge_count:,} edges")
    else:
        print(f"  [DRY RUN] Would delete {len(entity_ids):,} entities and {edge_count:,} edges")

    c.execute("DROP TABLE entities_to_delete")

    return len(entity_ids), edge_count


def prune_orphaned_nodes(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Remove nodes with no incoming or outgoing edges."""
    c = conn.cursor()
    print("\n[5/5] Pruning orphaned nodes...")

    orphans = c.execute("""
        SELECT id FROM nodes
        WHERE id NOT IN (SELECT src FROM edges)
          AND id NOT IN (SELECT dst FROM edges)
          AND type != 'entity'
    """).fetchall()

    orphan_ids = [oid for (oid,) in orphans]

    if not orphan_ids:
        print("  No orphaned nodes found.")
        return 0

    print(f"  Found {len(orphan_ids):,} orphaned nodes")

    if not dry_run:
        # Delete in batches
        for i in range(0, len(orphan_ids), MAX_SQL_PARAMS):
            batch = orphan_ids[i:i + MAX_SQL_PARAMS]
            placeholders = ','.join('?' * len(batch))
            c.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", batch)
        conn.commit()
        print(f"  ✓ Deleted {len(orphan_ids):,} orphaned nodes")
    else:
        print(f"  [DRY RUN] Would delete {len(orphan_ids):,} orphaned nodes")

    return len(orphan_ids)


def remove_stopword_entities(conn: sqlite3.Connection, collection: str = None, dry_run: bool = False) -> Tuple[int, int]:
    """
    Remove entities whose labels are generic stopwords.

    Only deletes entities that:
    1. Have labels matching ENTITY_STOPWORDS (from graph_builder.py)
    2. Have only 'mentions' edges (no structural relations)
    3. Are not involved in measurements or parameters

    Returns: (nodes_deleted, edges_deleted)
    """
    c = conn.cursor()
    print("\n[STOPWORD CLEANUP] Removing generic stopword entities...")

    # Stopwords from graph_builder.py
    STOPWORDS = {"system", "process", "method", "section", "table", "figure", "chapter", "page", "example"}

    # Find entities with stopword labels
    collection_filter = "AND collection = ?" if collection else ""
    params = [collection] if collection else []

    c.execute(f"""
        SELECT id, label FROM nodes
        WHERE type = 'entity' {collection_filter}
    """, params)

    stopword_entities = []
    for entity_id, label in c.fetchall():
        if label and label.lower().strip() in STOPWORDS:
            stopword_entities.append((entity_id, label))

    if not stopword_entities:
        print("  No stopword entities found.")
        return 0, 0

    print(f"  Found {len(stopword_entities):,} potential stopword entities")

    # Check structural protection - only delete if entity has NO:
    # - has_measurement edges (parameter nodes)
    # - row_has_parameter edges (table structure)
    # - other structural relations
    c.execute("CREATE TEMP TABLE stopword_candidates(id TEXT PRIMARY KEY, label TEXT)")
    for i in range(0, len(stopword_entities), MAX_SQL_PARAMS):
        batch = stopword_entities[i:i + MAX_SQL_PARAMS]
        c.executemany("INSERT INTO stopword_candidates VALUES (?, ?)", batch)

    # Find entities safe to delete (only have 'mentions' edges, no other relations)
    safe_to_delete = c.execute("""
        SELECT DISTINCT sc.id, sc.label
        FROM stopword_candidates sc
        WHERE sc.id NOT IN (
            SELECT DISTINCT src FROM edges WHERE type != 'mentions'
        )
        AND sc.id NOT IN (
            SELECT DISTINCT dst FROM edges WHERE type != 'mentions'
        )
    """).fetchall()

    c.execute("DROP TABLE stopword_candidates")

    if not safe_to_delete:
        print("  All stopword entities are structurally protected - none deleted.")
        return 0, 0

    entity_ids_to_delete = [eid for eid, _ in safe_to_delete]
    print(f"  Safe to delete: {len(entity_ids_to_delete):,} stopword entities")

    if dry_run:
        print("  [DRY RUN] Stopword entities that would be deleted:")
        for eid, label in safe_to_delete[:10]:
            print(f"    - {label} ({eid})")
        if len(safe_to_delete) > 10:
            print(f"    ... and {len(safe_to_delete) - 10} more")
        return 0, 0

    # Count edges before deletion
    c.execute("CREATE TEMP TABLE stopwords_to_delete(id TEXT PRIMARY KEY)")
    for i in range(0, len(entity_ids_to_delete), MAX_SQL_PARAMS):
        batch = entity_ids_to_delete[i:i + MAX_SQL_PARAMS]
        c.executemany("INSERT INTO stopwords_to_delete VALUES (?)", [(eid,) for eid in batch])

    edge_count = c.execute("""
        SELECT COUNT(*) FROM edges
        WHERE src IN (SELECT id FROM stopwords_to_delete)
           OR dst IN (SELECT id FROM stopwords_to_delete)
    """).fetchone()[0]

    # Delete edges and nodes
    c.execute("DELETE FROM edges WHERE src IN (SELECT id FROM stopwords_to_delete)")
    c.execute("DELETE FROM edges WHERE dst IN (SELECT id FROM stopwords_to_delete)")
    c.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM stopwords_to_delete)")
    c.execute("DROP TABLE stopwords_to_delete")

    conn.commit()
    print(f"  ✓ Deleted {len(entity_ids_to_delete):,} stopword entities and {edge_count:,} edges")
    return len(entity_ids_to_delete), edge_count


def deduplicate_measurements(conn: sqlite3.Connection, collection: str = None, dry_run: bool = False) -> Tuple[int, int]:
    """
    Deduplicate measurement nodes within the same chunk.

    For each chunk, keeps only the first measurement node for each unique (parameter, unit) pair.
    Removes redundant measurements common in table-heavy documents.

    Returns: (nodes_deleted, edges_deleted)
    """
    c = conn.cursor()
    print("\n[MEASUREMENT DEDUP] Deduplicating measurement nodes...")

    collection_filter = "AND m.collection = ?" if collection else ""
    params = [collection] if collection else []

    # Find all measurement nodes with their chunk and parameter in single query
    measurements = c.execute(f"""
        SELECT
            m.id as meas_id,
            m.label as meas_label,
            m.rowid,
            chunk_edge.src as chunk_node,
            param.label as param_label
        FROM nodes m
        JOIN edges chunk_edge ON chunk_edge.dst = m.id AND chunk_edge.type = 'mentions'
        JOIN edges param_edge ON param_edge.dst = m.id AND param_edge.type = 'has_measurement'
        JOIN nodes param ON param.id = param_edge.src
        WHERE m.type = 'measurement' {collection_filter}
        ORDER BY m.rowid
    """, params).fetchall()

    if not measurements:
        print("  No measurement nodes found.")
        return 0, 0

    print(f"  Found {len(measurements):,} measurement nodes")

    # Group measurements by chunk and (normalized_parameter, unit) key
    # Keep first occurrence of each unique combination
    chunk_param_units: Dict[Tuple[str, str, str], str] = {}  # (chunk, param_norm, unit) -> first_meas_id
    duplicates = []

    for meas_id, meas_label, rowid, chunk_node, param_label in measurements:
        # Parse measurement label to extract value and unit
        # Format: "value unit" (e.g., "25 mg/L", "7.2 ")
        parts = meas_label.strip().split(None, 1)
        if len(parts) < 1:
            continue

        # Heuristic: unit is everything after the first token (numeric value)
        unit = parts[1] if len(parts) > 1 else ""
        unit_norm = unit.strip().lower()

        # Normalize parameter label
        param_norm = param_label.strip().lower()

        # Check if we've seen this (chunk, param, unit) combination
        key = (chunk_node, param_norm, unit_norm)
        if key in chunk_param_units:
            # Duplicate - mark for deletion
            duplicates.append(meas_id)
        else:
            # First occurrence - keep it
            chunk_param_units[key] = meas_id

    if not duplicates:
        print("  No duplicate measurements found.")
        return 0, 0

    print(f"  Found {len(duplicates):,} duplicate measurement nodes")

    if dry_run:
        print(f"  [DRY RUN] Would delete {len(duplicates):,} duplicate measurements")
        return 0, 0

    # Count edges before deletion
    c.execute("CREATE TEMP TABLE dupes_to_delete(id TEXT PRIMARY KEY)")
    for i in range(0, len(duplicates), MAX_SQL_PARAMS):
        batch = duplicates[i:i + MAX_SQL_PARAMS]
        c.executemany("INSERT INTO dupes_to_delete VALUES (?)", [(mid,) for mid in batch])

    edge_count = c.execute("""
        SELECT COUNT(*) FROM edges
        WHERE src IN (SELECT id FROM dupes_to_delete)
           OR dst IN (SELECT id FROM dupes_to_delete)
    """).fetchone()[0]

    # Delete edges and nodes
    c.execute("DELETE FROM edges WHERE src IN (SELECT id FROM dupes_to_delete)")
    c.execute("DELETE FROM edges WHERE dst IN (SELECT id FROM dupes_to_delete)")
    c.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM dupes_to_delete)")
    c.execute("DROP TABLE dupes_to_delete")

    conn.commit()
    print(f"  ✓ Deleted {len(duplicates):,} duplicate measurements and {edge_count:,} edges")
    return len(duplicates), edge_count


def vacuum_and_optimize(conn: sqlite3.Connection, dry_run: bool = False):
    """Vacuum and optimize database to reclaim space."""
    if dry_run:
        print("\n[FINAL] Vacuum and optimize...")
        print("  [DRY RUN] Would run: PRAGMA wal_checkpoint(FULL); VACUUM; ANALYZE;")
        return

    print("\n[FINAL] Vacuuming and optimizing database...")
    c = conn.cursor()
    print("  Checkpointing WAL...")
    c.execute("PRAGMA wal_checkpoint(FULL)")
    print("  Vacuuming (this may take a while)...")
    c.execute("VACUUM")
    print("  Analyzing...")
    c.execute("ANALYZE")
    conn.commit()
    print("  ✓ Database optimized")


def main():
    parser = argparse.ArgumentParser(description="Clean up graph.db entity bloat (improved v2 with stopword & measurement dedup)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--backup", action="store_true", help="Create backup before changes")
    parser.add_argument("--collection", help="Only process entities from specific collection")
    parser.add_argument("--min-doc-freq", type=int, default=2, help="Min docs for entity (default: 2)")
    parser.add_argument("--min-tfidf", type=float, default=0.3, help="Min TF-IDF score (default: 0.3)")
    parser.add_argument("--keep-top-pct", type=float, default=90, help="Keep top N%% (default: 90)")
    parser.add_argument("--db", default="data/graph.db", help="Path to graph database")

    # New optional cleanup passes
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove generic stopword entities (system, process, method, etc.)")
    parser.add_argument("--dedupe-measurements", action="store_true", help="Deduplicate measurement nodes within chunks")

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    print(f"Graph Bloat Cleanup v2 (Improved)")
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.collection:
        print(f"Collection filter: {args.collection}")
    print()

    if args.backup and not args.dry_run:
        backup_path = db_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        print(f"Creating backup: {backup_path}")
        shutil.copy2(db_path, backup_path)
        print(f"  ✓ Backup created\n")

    conn = sqlite3.connect(db_path)
    if not args.dry_run:
        conn.execute("PRAGMA journal_mode=DELETE")

    initial_stats = get_db_stats(conn)
    print_stats(initial_stats, "Initial Statistics")

    try:
        # Execute cleanup
        cooccur_removed = remove_cooccurrence_edges(conn, args.dry_run)
        dupes_removed = deduplicate_mention_edges(conn, args.dry_run)
        entity_stats, total_docs = compute_entity_stats_fast(conn, args.collection)
        structural_signals = get_structural_signals(conn, set(entity_stats.keys()))
        entities_to_delete = filter_entities_improved(
            entity_stats, total_docs, structural_signals,
            min_doc_freq=args.min_doc_freq,
            min_tfidf=args.min_tfidf,
            keep_top_pct=args.keep_top_pct,
        )
        nodes_deleted, edges_deleted = delete_entities_and_edges_batched(conn, entities_to_delete, args.dry_run)
        orphan_entities_deleted = delete_orphan_entities(conn, args.collection, args.dry_run)
        orphans_deleted = prune_orphaned_nodes(conn, args.dry_run)

        # New optional cleanup passes
        stopword_nodes = 0
        stopword_edges = 0
        meas_nodes = 0
        meas_edges = 0

        if args.remove_stopwords:
            stopword_nodes, stopword_edges = remove_stopword_entities(conn, args.collection, args.dry_run)

        if args.dedupe_measurements:
            meas_nodes, meas_edges = deduplicate_measurements(conn, args.collection, args.dry_run)

        vacuum_and_optimize(conn, args.dry_run)

        final_stats = get_db_stats(conn)
        print_stats(final_stats, "Final Statistics")

        print(f"\n{'='*60}")
        print("Cleanup Summary")
        print(f"{'='*60}")
        print(f"Co-occurrence edges removed:   {cooccur_removed:,}")
        print(f"Duplicate mentions removed:    {dupes_removed:,}")
        print(f"Entity nodes deleted (TF-IDF): {nodes_deleted:,}")
        print(f"Orphan entities deleted:       {orphan_entities_deleted:,}")
        print(f"Associated edges deleted:      {edges_deleted:,}")
        print(f"Orphaned chunk nodes deleted:  {orphans_deleted:,}")

        if args.remove_stopwords:
            print(f"Stopword entities removed:     {stopword_nodes:,}")
            print(f"Stopword edges removed:        {stopword_edges:,}")

        if args.dedupe_measurements:
            print(f"Duplicate measurements removed:{meas_nodes:,}")
            print(f"Measurement edges removed:     {meas_edges:,}")

        print()
        print(f"File size: {initial_stats['file_size_mb']:.1f} MB → {final_stats['file_size_mb']:.1f} MB")
        size_reduction = initial_stats['file_size_mb'] - final_stats['file_size_mb']
        size_reduction_pct = 100 * size_reduction / initial_stats['file_size_mb']
        print(f"Size reduction: {size_reduction:.1f} MB ({size_reduction_pct:.1f}%)")
        print()

        if args.dry_run:
            print("DRY RUN complete - no changes were made.")
        else:
            print("✓ Cleanup complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
