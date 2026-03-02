#!/usr/bin/env python3
"""
Deduplicate Dir 1 (CBG Meerut - Documents) against Dir 2 (Final Project Documents).

Builds a symlink tree in data/meerut_dir1_unique/ containing only files from Dir 1
that don't exist in Dir 2 (matched by lowercase filename + file size).

Usage:
    python scripts/dedup_dir1_vs_dir2.py
"""
import os
import sys
from pathlib import Path

DIR1 = Path("/mnt/c/Users/hvksh/Circle H2O LLC/CBG Meerut - Documents")
DIR2 = Path("/mnt/c/Users/hvksh/Circle H2O LLC/Final Project Documents - Documents/Meerut CBG Plant")
OUTPUT = Path("data/meerut_dir1_unique")
EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".doc", ".xls"}
MAX_FILE_MB = 110


def main():
    # Step 1: Build fingerprint set from Dir 2
    print(f"Scanning Dir 2: {DIR2}")
    dir2_fingerprints = set()
    dir2_count = 0
    for root, dirs, files in os.walk(DIR2):
        for f in files:
            fp = Path(root) / f
            if fp.suffix.lower() not in EXTENSIONS:
                continue
            try:
                size = fp.stat().st_size
                if size > MAX_FILE_MB * 1024 * 1024:
                    continue
                dir2_fingerprints.add((f.lower(), size))
                dir2_count += 1
            except OSError:
                continue
    print(f"  Dir 2: {dir2_count} ingestible files, {len(dir2_fingerprints)} unique fingerprints")

    # Step 2: Walk Dir 1, identify unique files
    print(f"\nScanning Dir 1: {DIR1}")
    dir1_total = 0
    dir1_dupes = 0
    dir1_unique = []
    for root, dirs, files in os.walk(DIR1):
        for f in files:
            fp = Path(root) / f
            if fp.suffix.lower() not in EXTENSIONS:
                continue
            try:
                size = fp.stat().st_size
                if size > MAX_FILE_MB * 1024 * 1024:
                    continue
            except OSError:
                continue
            dir1_total += 1
            fingerprint = (f.lower(), size)
            if fingerprint in dir2_fingerprints:
                dir1_dupes += 1
            else:
                # Preserve relative path from DIR1
                rel = fp.relative_to(DIR1)
                dir1_unique.append((fp, rel))

    print(f"  Dir 1: {dir1_total} ingestible files")
    print(f"  Duplicates (in Dir 2): {dir1_dupes}")
    print(f"  Unique to Dir 1: {len(dir1_unique)}")

    # Step 3: Create symlink tree
    if OUTPUT.exists():
        import shutil
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    symlinks_created = 0
    for src, rel in dir1_unique:
        dst = OUTPUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, dst)
            symlinks_created += 1
        except OSError as e:
            print(f"  WARN: symlink failed for {rel}: {e}", file=sys.stderr)

    print(f"\nSymlink tree created: {OUTPUT}/")
    print(f"  Symlinks: {symlinks_created}")
    print(f"  Ready for ingestion with --root {OUTPUT}")


if __name__ == "__main__":
    main()
