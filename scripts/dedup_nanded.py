#!/usr/bin/env python3
"""
Deduplicate Nanded working docs (Dir 1) against Final Project Documents (Dir 2).

Builds a symlink tree in data/nanded_unique/ containing only files from Dir 1
that don't exist in Dir 2 (matched by lowercase filename + file size).

Usage:
    python scripts/dedup_nanded.py
"""
import os
import sys
from pathlib import Path

DIR1 = Path("/mnt/c/Users/hvksh/Circle H2O LLC/Nanded 7.5 MLD MBR - Documents")
DIR2 = Path("/mnt/c/Users/hvksh/Circle H2O LLC/Final Project Documents - Documents/Nanded 7.5 MLD MBR")
OUTPUT = Path("data/nanded_unique")
EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".doc", ".xls"}
MAX_FILE_MB = 110

# Skip these subdirectories (low engineering value)
SKIP_DIRS = {
    "2022 Summer Intership Shared Folder",
    "2022 Fall Internship",
    "Advanced Analytics Planning",
}


def should_skip(dirpath: Path) -> bool:
    """Check if any component of the path matches a skip directory."""
    parts = set(dirpath.parts)
    return bool(parts & SKIP_DIRS)


def main():
    # Step 1: Build fingerprint set from Dir 2 (canonical)
    print(f"Scanning Dir 2 (canonical): {DIR2}")
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

    # Step 2: Walk Dir 1, identify unique files (skipping internship + analytics)
    print(f"\nScanning Dir 1 (working): {DIR1}")
    dir1_total = 0
    dir1_dupes = 0
    dir1_skipped = 0
    dir1_unique = []
    for root, dirs, files in os.walk(DIR1):
        rp = Path(root)
        if should_skip(rp):
            dir1_skipped += len(files)
            continue
        for f in files:
            fp = rp / f
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
                rel = fp.relative_to(DIR1)
                dir1_unique.append((fp, rel))

    print(f"  Dir 1: {dir1_total} ingestible files (after skip filter)")
    print(f"  Skipped (internship/analytics): {dir1_skipped} files")
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
