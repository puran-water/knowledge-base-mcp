#!/usr/bin/env python3
"""
Graph Bloat Cleanup Script

Post-processes existing graph.db to remove bloat without re-ingesting collections.
Implements Codex-recommended cleanup strategy:
1. Remove all co-occurrence edges (N² bloat)
2. Deduplicate entity mentions (element_id loop bug fix)
3. Smart entity filtering using TF-IDF and frequency thresholds
4. Remove low-value entities (ISBN, metadata, generic phrases)
5. Prune orphaned nodes
6. Vacuum and optimize database

Usage:
    python scripts/cleanup_graph_bloat.py [--dry-run] [--backup] [--collection COLLECTION]

Options:
    --dry-run         Show what would be done without making changes
    --backup          Create backup before making changes (recommended)
    --collection STR  Only process entities from specific collection
    --min-doc-freq N  Minimum documents entity must appear in (default: 1)
    --min-tfidf F     Minimum TF-IDF score to keep entity (default: 0.5)
    --keep-top-pct N  Keep top N% of entities by TF-IDF score (default: 75)
"""

import argparse
import sqlite3
import shutil
import sys
import re
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from datetime import datetime


# Low-value entity patterns (ISBN, publisher metadata, generic phrases)
JUNK_PATTERNS = [
    r"^isbn[:\s-]*\d",  # ISBN numbers
    r"^\d{3,}$",  # Long digit sequences
    r"^doi:",  # DOI identifiers
    r"^copyright\s+\d{4}",  # Copyright statements
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


def get_db_stats(conn: sqlite3.Connection) -> Dict:
    """Get current database statistics."""
    c = conn.cursor()

    stats = {}

    # Total counts
    stats['total_nodes'] = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    stats['total_edges'] = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # Nodes by type
    node_types = c.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall()
    stats['nodes_by_type'] = {t: cnt for t, cnt in node_types}

    # Edges by type
    edge_types = c.execute("SELECT type, COUNT(*) FROM edges GROUP BY type").fetchall()
    stats['edges_by_type'] = {t: cnt for t, cnt in edge_types}

    # File size
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

    # Empty or too short
    if len(label_lower) < 2:
        return True

    # Check junk patterns
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, label_lower):
            return True

    return False


def remove_cooccurrence_edges(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Remove all co-occurrence edges (N² bloat source)."""
    c = conn.cursor()

    # Count co-occurrence edges
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
    Keeps one mention edge per (doc_id, src, dst, type) tuple.
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
        # Create temp table with deduplicated edges
        c.execute("""
            CREATE TEMP TABLE mentions_dedup AS
            SELECT DISTINCT src, dst, type, collection, doc_id
            FROM edges
            WHERE type = 'mentions'
        """)

        # Delete all mention edges
        c.execute("DELETE FROM edges WHERE type='mentions'")

        # Reinsert deduplicated edges
        c.execute("""
            INSERT INTO edges (src, dst, type, collection, doc_id)
            SELECT src, dst, type, collection, doc_id
            FROM mentions_dedup
        """)

        c.execute("DROP TABLE mentions_dedup")
        conn.commit()
        print(f"  ✓ Removed {total_duplicate_edges:,} duplicate mention edges")
    else:
        print(f"  [DRY RUN] Would remove {total_duplicate_edges:,} duplicates")

    return total_duplicate_edges


def compute_entity_stats(conn: sqlite3.Connection, collection: str = None) -> Tuple[Dict, int]:
    """
    Compute TF-IDF statistics for entity filtering.
    Returns (entity_stats, total_docs) where entity_stats is:
    {
        entity_id: {
            'label': str,
            'doc_count': int,  # df - document frequency
            'total_mentions': int,  # total mentions across all docs
            'max_tf_doc': int,  # max mentions in any single doc
            'tfidf': float,  # max TF-IDF score
        }
    }
    """
    c = conn.cursor()

    print("\n[3/5] Computing entity statistics for smart filtering...")

    # Get all entity nodes
    if collection:
        entity_query = f"SELECT id, label FROM nodes WHERE type='entity' AND collection=?"
        entities = {eid: label for eid, label in c.execute(entity_query, (collection,)).fetchall()}
    else:
        entity_query = "SELECT id, label FROM nodes WHERE type='entity'"
        entities = {eid: label for eid, label in c.execute(entity_query).fetchall()}

    print(f"  Analyzing {len(entities):,} entities...")

    # Count total documents
    if collection:
        doc_query = "SELECT COUNT(DISTINCT doc_id) FROM nodes WHERE collection=?"
        total_docs = c.execute(doc_query, (collection,)).fetchone()[0]
    else:
        doc_query = "SELECT COUNT(DISTINCT doc_id) FROM nodes"
        total_docs = c.execute(doc_query).fetchone()[0]

    # For each entity, compute stats
    entity_stats = {}

    for entity_id, label in entities.items():
        # Get all mentions of this entity
        mentions_query = """
            SELECT doc_id, COUNT(*) as mentions
            FROM edges
            WHERE dst = ? AND type = 'mentions'
            GROUP BY doc_id
        """
        mentions = c.execute(mentions_query, (entity_id,)).fetchall()

        if not mentions:
            continue

        doc_count = len(mentions)  # df
        total_mentions = sum(cnt for _, cnt in mentions)
        max_tf_doc = max(cnt for _, cnt in mentions)

        # Compute max TF-IDF: max(tf_doc) * log((N+1)/(df+1))
        # Using log1p for numerical stability
        idf = math.log((total_docs + 1) / (doc_count + 1))
        tfidf = math.log1p(max_tf_doc) * idf

        entity_stats[entity_id] = {
            'label': label,
            'doc_count': doc_count,
            'total_mentions': total_mentions,
            'max_tf_doc': max_tf_doc,
            'tfidf': tfidf,
        }

    return entity_stats, total_docs


def filter_entities(
    entity_stats: Dict,
    total_docs: int,
    min_doc_freq: int = 1,
    min_tfidf: float = 0.5,
    keep_top_pct: float = 75,
) -> Set[str]:
    """
    Filter entities using statistical thresholds and pattern matching.
    Returns set of entity IDs to DELETE.
    """
    print("\n[4/5] Filtering low-value entities...")

    to_delete = set()

    # Pattern-based filtering (junk entities)
    junk_count = 0
    for entity_id, stats in entity_stats.items():
        if is_junk_entity(stats['label']):
            to_delete.add(entity_id)
            junk_count += 1

    print(f"  Pattern-based junk filtering: {junk_count:,} entities")

    # Statistical filtering
    remaining = {eid: stats for eid, stats in entity_stats.items() if eid not in to_delete}

    # Filter by document frequency
    low_df = {eid for eid, stats in remaining.items() if stats['doc_count'] < min_doc_freq}
    to_delete.update(low_df)
    print(f"  Low document frequency (df < {min_doc_freq}): {len(low_df):,} entities")

    # Filter by TF-IDF threshold
    remaining = {eid: stats for eid, stats in remaining.items() if eid not in to_delete}
    low_tfidf = {eid for eid, stats in remaining.items() if stats['tfidf'] < min_tfidf}
    to_delete.update(low_tfidf)
    print(f"  Low TF-IDF score (< {min_tfidf:.2f}): {len(low_tfidf):,} entities")

    # Keep top N% by TF-IDF
    remaining = {eid: stats for eid, stats in remaining.items() if eid not in to_delete}
    sorted_by_tfidf = sorted(remaining.items(), key=lambda x: x[1]['tfidf'], reverse=True)
    keep_count = int(len(sorted_by_tfidf) * keep_top_pct / 100)
    bottom_entities = {eid for eid, _ in sorted_by_tfidf[keep_count:]}
    to_delete.update(bottom_entities)
    print(f"  Bottom {100-keep_top_pct}% by TF-IDF: {len(bottom_entities):,} entities")

    print(f"\n  Total entities to delete: {len(to_delete):,} / {len(entity_stats):,} ({100*len(to_delete)/len(entity_stats):.1f}%)")
    print(f"  Entities to keep: {len(entity_stats) - len(to_delete):,} ({100*(1-len(to_delete)/len(entity_stats)):.1f}%)")

    return to_delete


def delete_entities_and_edges(conn: sqlite3.Connection, entity_ids: Set[str], dry_run: bool = False) -> Tuple[int, int]:
    """Delete entity nodes and their associated edges."""
    if not entity_ids:
        return 0, 0

    c = conn.cursor()

    # Count edges to be deleted
    edge_count = c.execute(
        f"SELECT COUNT(*) FROM edges WHERE src IN ({','.join('?' * len(entity_ids))}) OR dst IN ({','.join('?' * len(entity_ids))})",
        list(entity_ids) + list(entity_ids)
    ).fetchone()[0]

    print(f"\n  Deleting {len(entity_ids):,} entity nodes and {edge_count:,} associated edges...")

    if not dry_run:
        # Delete edges
        c.execute(
            f"DELETE FROM edges WHERE src IN ({','.join('?' * len(entity_ids))}) OR dst IN ({','.join('?' * len(entity_ids))})",
            list(entity_ids) + list(entity_ids)
        )

        # Delete nodes
        c.execute(
            f"DELETE FROM nodes WHERE id IN ({','.join('?' * len(entity_ids))})",
            list(entity_ids)
        )

        conn.commit()
        print(f"  ✓ Deleted {len(entity_ids):,} entities and {edge_count:,} edges")
    else:
        print(f"  [DRY RUN] Would delete {len(entity_ids):,} entities and {edge_count:,} edges")

    return len(entity_ids), edge_count


def prune_orphaned_nodes(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Remove nodes with no incoming or outgoing edges."""
    c = conn.cursor()

    print("\n[5/5] Pruning orphaned nodes...")

    # Find nodes with no edges
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
        c.execute(
            f"DELETE FROM nodes WHERE id IN ({','.join('?' * len(orphan_ids))})",
            orphan_ids
        )
        conn.commit()
        print(f"  ✓ Deleted {len(orphan_ids):,} orphaned nodes")
    else:
        print(f"  [DRY RUN] Would delete {len(orphan_ids):,} orphaned nodes")

    return len(orphan_ids)


def vacuum_and_optimize(conn: sqlite3.Connection, dry_run: bool = False):
    """Vacuum and optimize database to reclaim space."""
    if dry_run:
        print("\n[FINAL] Vacuum and optimize...")
        print("  [DRY RUN] Would run: PRAGMA wal_checkpoint(FULL); VACUUM; ANALYZE;")
        return

    print("\n[FINAL] Vacuuming and optimizing database...")
    c = conn.cursor()

    # Checkpoint WAL
    print("  Checkpointing WAL...")
    c.execute("PRAGMA wal_checkpoint(FULL)")

    # Vacuum to reclaim space
    print("  Vacuuming (this may take a while)...")
    c.execute("VACUUM")

    # Analyze for query optimization
    print("  Analyzing...")
    c.execute("ANALYZE")

    conn.commit()
    print("  ✓ Database optimized")


def main():
    parser = argparse.ArgumentParser(description="Clean up graph.db entity bloat")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backup", action="store_true", help="Create backup before making changes")
    parser.add_argument("--collection", help="Only process entities from specific collection")
    parser.add_argument("--min-doc-freq", type=int, default=1, help="Minimum documents entity must appear in")
    parser.add_argument("--min-tfidf", type=float, default=0.5, help="Minimum TF-IDF score to keep entity")
    parser.add_argument("--keep-top-pct", type=float, default=75, help="Keep top N%% of entities by TF-IDF")
    parser.add_argument("--db", default="data/graph.db", help="Path to graph database")

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    print(f"Graph Bloat Cleanup")
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.collection:
        print(f"Collection filter: {args.collection}")
    print()

    # Backup if requested
    if args.backup and not args.dry_run:
        backup_path = db_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        print(f"Creating backup: {backup_path}")
        shutil.copy2(db_path, backup_path)
        print(f"  ✓ Backup created\n")

    # Open database (disable WAL for vacuum)
    conn = sqlite3.connect(db_path)
    if not args.dry_run:
        conn.execute("PRAGMA journal_mode=DELETE")

    # Get initial stats
    initial_stats = get_db_stats(conn)
    print_stats(initial_stats, "Initial Statistics")

    # Execute cleanup steps
    try:
        # Step 1: Remove co-occurrence edges
        cooccur_removed = remove_cooccurrence_edges(conn, args.dry_run)

        # Step 2: Deduplicate mention edges
        dupes_removed = deduplicate_mention_edges(conn, args.dry_run)

        # Step 3: Compute entity statistics
        entity_stats, total_docs = compute_entity_stats(conn, args.collection)

        # Step 4: Filter entities
        entities_to_delete = filter_entities(
            entity_stats,
            total_docs,
            min_doc_freq=args.min_doc_freq,
            min_tfidf=args.min_tfidf,
            keep_top_pct=args.keep_top_pct,
        )

        # Delete entities and edges
        nodes_deleted, edges_deleted = delete_entities_and_edges(conn, entities_to_delete, args.dry_run)

        # Step 5: Prune orphaned nodes
        orphans_deleted = prune_orphaned_nodes(conn, args.dry_run)

        # Final: Vacuum and optimize
        vacuum_and_optimize(conn, args.dry_run)

        # Get final stats
        final_stats = get_db_stats(conn)
        print_stats(final_stats, "Final Statistics")

        # Summary
        print(f"\n{'='*60}")
        print("Cleanup Summary")
        print(f"{'='*60}")
        print(f"Co-occurrence edges removed:  {cooccur_removed:,}")
        print(f"Duplicate mentions removed:   {dupes_removed:,}")
        print(f"Entity nodes deleted:         {nodes_deleted:,}")
        print(f"Associated edges deleted:     {edges_deleted:,}")
        print(f"Orphaned nodes deleted:       {orphans_deleted:,}")
        print()
        print(f"File size: {initial_stats['file_size_mb']:.1f} MB → {final_stats['file_size_mb']:.1f} MB")
        size_reduction = initial_stats['file_size_mb'] - final_stats['file_size_mb']
        size_reduction_pct = 100 * size_reduction / initial_stats['file_size_mb']
        print(f"Size reduction: {size_reduction:.1f} MB ({size_reduction_pct:.1f}%)")
        print()

        if args.dry_run:
            print("DRY RUN complete - no changes were made.")
            print("Run without --dry-run to apply changes.")
        else:
            print("✓ Cleanup complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
