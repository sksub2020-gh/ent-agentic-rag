**SQLiteSparseStore** — SQLite FTS5 with built-in BM25 scoring, no corpus loading into memory:

Key design decisions worth understanding:
- **Two tables not one **— FTS5 virtual tables don't support non-text columns well, so metadata lives in a companion chunks_meta table joined on chunk_id. Clean separation.
- **Incremental upsert** — FTS5 doesn't support ON CONFLICT so it's DELETE + INSERT. chunks_meta uses standard ON CONFLICT DO UPDATE. Re-ingesting the same doc updates in place, no full rebuild.
- **Query sanitisation** — FTS5 has its own query syntax (AND, OR, NOT, *, quotes). Wrapping each token in double quotes treats everything as plain text, preventing syntax errors on arbitrary user input.
- **Score negation** — SQLite's bm25() returns negative values (e.g. -3.2). We negate so higher = more relevant, consistent with every other store in the pipeline.
To wire it in for the Milvus local path — just swap BM25SStore for SQLiteSparseStore:
```
from retrieval.sqlite_sparse_store import SQLiteSparseStore
sparse_store = SQLiteSparseStore()
```
No other changes needed — same SparseStoreBase interface.