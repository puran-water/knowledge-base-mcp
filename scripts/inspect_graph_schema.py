#!/usr/bin/env python3
"""Quick script to inspect graph.db schema."""
import sqlite3

conn = sqlite3.connect('data/graph.db')
c = conn.cursor()

tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t[0] for t in tables])
print()

for table in [t[0] for t in tables]:
    schema = c.execute(f'PRAGMA table_info({table})').fetchall()
    print(f'{table}:')
    for col in schema:
        print(f'  {col[1]} {col[2]}')
    print()

conn.close()
