"""
SQLite FTS5 sparse store — local alternative to BM25S.

Key advantages over BM25SStore:
  - No corpus loading into memory on startup — FTS5 index lives on disk
  - Incremental updates — upsert individual chunks without rebuilding full index
  - BM25 scoring built into SQLite FTS5 via bm25() function
  - Zero extra dependencies — sqlite3 is Python stdlib
  - Single .db file — easy to inspect, backup, or delete

Usage:
    store = SQLiteSparseStore()
    store.index(chunks)          # upsert chunks into FTS5 table
    results = store.search("query", top_k=20)

FTS5 BM25 notes:
  - SQLite's bm25() returns negative values (more negative = more relevant)
  - We negate the score so higher = better, consistent with rest of pipeline
  - Porter stemming enabled — "running" matches "run", "runs"
  - unicode61 tokenizer handles punctuation and case folding
"""

import json
import logging
import sqlite3
from pathlib import Path

from core.interfaces import SparseStoreBase, Chunk, RetrievedChunk
from config.settings import config

logger = logging.getLogger(__name__)


class SQLiteSparseStore(SparseStoreBase):
    """
    SQLite FTS5 sparse full-text search store.
    Implements SparseStoreBase — drop-in replacement for BM25SStore.

    Two tables:
      chunks_fts  — FTS5 virtual table for full-text search (content + scoring)
      chunks_meta — Regular table for metadata (source, page, section etc.)
                    FTS5 doesn't support non-text columns natively so metadata
                    lives in a companion table joined on chunk_id.
    """

    def __init__(self):
        db_path = Path(config.bm25.index_path) / "fts5.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # access columns by name
        self._create_tables()
        count = self._count()
        logger.info(f"SQLiteSparseStore ready → {db_path} ({count} chunks indexed)")

    # ── SparseStoreBase ───────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """
        Upsert chunks into FTS5 index.
        Unlike BM25S, this is incremental — existing chunks are updated,
        new chunks are inserted. No full rebuild needed.
        """
        if not chunks:
            return

        fts_rows = []
        meta_rows = []

        for chunk in chunks:
            fts_rows.append((chunk.chunk_id, chunk.content))
            meta_rows.append(
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    json.dumps(chunk.metadata),
                )
            )

        with self.conn:
            # FTS5 upsert — DELETE + INSERT (FTS5 doesn't support ON CONFLICT)
            self.conn.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(r[0],) for r in fts_rows],
            )
            self.conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                fts_rows,
            )

            # Metadata upsert — standard ON CONFLICT
            self.conn.executemany(
                """
                INSERT INTO chunks_meta (chunk_id, doc_id, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id   = excluded.doc_id,
                    metadata = excluded.metadata
                """,
                meta_rows,
            )

        logger.info(f"SQLiteSparseStore indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        BM25 full-text search via SQLite FTS5.
        bm25() scores are negative — we negate so higher = more relevant,
        consistent with the rest of the pipeline.
        """
        if not query.strip():
            return []

        # Sanitise query — FTS5 syntax can throw on special chars
        safe_query = self._sanitise_query(query)
        if not safe_query:
            return []

        sql = """
            SELECT
                f.chunk_id,
                f.content,
                m.doc_id,
                m.metadata,
                -bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks_meta m ON m.chunk_id = f.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """

        try:
            rows = self.conn.execute(sql, (safe_query, top_k)).fetchall()
        except sqlite3.OperationalError as e:
            # Malformed FTS query — return empty rather than crash
            logger.warning(f"FTS5 query error for '{query}': {e}")
            return []

        results = []
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                content=row["content"],
                metadata=metadata,
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(row["score"]),
                    source="sparse",
                )
            )

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create FTS5 and metadata tables if they don't exist."""
        with self.conn:
            # FTS5 virtual table — Porter stemming + unicode61 tokenizer
            # content='' means FTS5 stores its own copy of content (simpler)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    tokenize = 'porter unicode61 remove_diacritics 1'
                )
            """)

            # Companion metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks_meta (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id   TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)

    def _sanitise_query(self, query: str) -> str:
        """
        FTS5 treats some chars as syntax (AND, OR, NOT, quotes, *).
        Wrap each token in double quotes to treat the query as plain text.
        This disables FTS5 boolean operators but prevents syntax errors.
        """
        tokens = query.strip().split()
        # Remove empty tokens and escape internal quotes
        safe_tokens = [f'"{t.replace(chr(34), "")}"' for t in tokens if t]
        return " ".join(safe_tokens)

    def _count(self) -> int:
        """Return total indexed chunks."""
        try:
            row = self.conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks from index."""
        with self.conn:
            self.conn.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids],
            )
            self.conn.executemany(
                "DELETE FROM chunks_meta WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids],
            )
        logger.info(f"Deleted {len(chunk_ids)} chunks from SQLiteSparseStore")

    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
