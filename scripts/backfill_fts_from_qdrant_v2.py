#!/usr/bin/env python3
"""
Optimized FTS Backfill Script v2

Improvements over v1:
1. Progress bar with % complete based on collection size
2. Single FTSWriter for entire run (eliminates per-batch connection overhead)
3. Configurable batch size for FTS upsert
4. Better error handling and reporting

Usage:
    python scripts/backfill_fts_from_qdrant_v2.py \
        --collection misc_process_kb \
        --fts-db data/misc_process_kb_fts.db \
        --batch 1000 \
        --fts-batch-size 2000
"""
import argparse
import os
import sys
import pathlib
from typing import Any, Dict, List

from qdrant_client import QdrantClient

# Ensure repo root is on sys.path so we can import lexical_index when invoked as a file
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Backfill FTS from Qdrant with progress tracking")
    ap.add_argument('--qdrant-url', default=os.getenv('QDRANT_URL', 'http://localhost:6333'))
    ap.add_argument('--qdrant-api-key', default=os.getenv('QDRANT_API_KEY'))
    ap.add_argument('--collection', required=True, help='Qdrant collection name')
    ap.add_argument('--fts-db', default=os.getenv('FTS_DB_PATH', 'data/fts.db'), help='FTS database path')
    ap.add_argument('--batch', type=int, default=1000, help='Qdrant scroll batch size (default: 1000)')
    ap.add_argument('--fts-batch-size', type=int, default=2000, help='FTS upsert batch size (default: 2000)')
    ap.add_argument('--limit', type=int, default=0, help='Stop after N points (0 = all)')
    ap.add_argument('--min-words', type=int, default=0, help='Skip chunks with fewer than N words')
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    from lexical_index import ensure_fts, FTSWriter

    ensure_fts(args.fts_db)

    # Get total collection size for progress tracking
    print(f"Backfilling FTS from Qdrant collection: {args.collection}")
    print(f"Target FTS database: {args.fts_db}")
    collection_info = client.get_collection(args.collection)
    total_points = collection_info.points_count
    print(f"Total vectors in collection: {total_points:,}\n")

    # Open single FTSWriter for entire run (performance optimization)
    next_page = None
    total_processed = 0
    total_upserted = 0

    with FTSWriter(args.fts_db) as writer:
        while True:
            points, next_page = client.scroll(
                collection_name=args.collection,
                with_payload=True,
                with_vectors=False,
                limit=args.batch,
                offset=next_page,
            )
            if not points:
                break

            total_processed += len(points)

            rows = []
            for p in points:
                pl: Dict[str, Any] = p.payload or {}
                text = (pl.get('text') or '').strip()
                if not text:
                    continue
                if args.min_words:
                    import re
                    if len(re.findall(r'[A-Za-z]{2,}', text)) < args.min_words:
                        continue
                page_numbers = pl.get('page_numbers') or pl.get('pages')
                section_path = pl.get('section_path')
                element_ids = pl.get('element_ids')
                bboxes = pl.get('bboxes')
                types = pl.get('types')
                source_tools = pl.get('source_tools')
                table_headers = pl.get('table_headers')
                table_units = pl.get('table_units')
                chunk_profile = pl.get('chunk_profile')
                plan_hash = pl.get('plan_hash')
                doc_metadata = pl.get('doc_metadata')
                model_version = pl.get('model_version')
                prompt_sha = pl.get('prompt_sha')
                rows.append({
                    'text': text,
                    'chunk_id': str(p.id),
                    'doc_id': pl.get('doc_id'),
                    'path': pl.get('path'),
                    'filename': pl.get('filename'),
                    'source_path': pl.get('source_path'),
                    'chunk_start': pl.get('chunk_start'),
                    'chunk_end': pl.get('chunk_end'),
                    'mtime': pl.get('mtime'),
                    'page_numbers': page_numbers,
                    'pages': pl.get('pages'),
                    'section_path': section_path,
                    'element_ids': element_ids,
                    'bboxes': bboxes,
                    'types': types,
                    'source_tools': source_tools,
                    'table_headers': table_headers,
                    'table_units': table_units,
                    'chunk_profile': chunk_profile,
                    'plan_hash': plan_hash,
                    'model_version': model_version,
                    'prompt_sha': prompt_sha,
                    'doc_metadata': doc_metadata,
                })

            if rows:
                # Use single FTSWriter with batching (performance optimization from Codex)
                upserted = writer.upsert_many(rows, batch_size=args.fts_batch_size, show_progress=False)
                total_upserted += upserted

                # Progress bar - show % complete
                pct_complete = (total_processed / total_points) * 100 if total_points > 0 else 0
                print(f"\rProgress: {total_upserted:,}/{total_points:,} chunks ({pct_complete:.1f}%) ", end='', flush=True)

            if args.limit and total_upserted >= args.limit:
                break
            if next_page is None:
                break

    print(f"\n\n✓ Backfill complete!")
    print(f"  Processed: {total_processed:,} vectors from Qdrant")
    print(f"  Upserted: {total_upserted:,} chunks to FTS")
    print(f"  Database: {args.fts_db}")


if __name__ == '__main__':
    main()
