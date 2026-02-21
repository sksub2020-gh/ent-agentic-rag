"""
Supabase (Postgres + pgvector) store.

Replaces BOTH Milvus-Lite (dense) and BM25S (sparse) with a single store.

Dense search:  pgvector  — cosine similarity on embedding column
Sparse search: tsvector  — Postgres full-text search (replaces BM25S)
Hybrid fusion: single SQL query using RRF via CTE (no Python-side merge needed)
Scalar filter: plain SQL WHERE on page, section, doc_type, version columns

Connection: psycopg2 direct Postgres (fastest, no REST overhead)
Add to requirements.txt:
    psycopg2-binary>=2.9.0
    pgvector>=0.2.0          # Python pgvector adapter

Schema setup: run retrieval/supabase_migration.sql in Supabase SQL editor first.
"""
import json
import logging
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from core.interfaces import VectorStoreBase, SparseStoreBase, Chunk, RetrievedChunk
from config.settings import config

logger = logging.getLogger(__name__)


class SupabaseStore(VectorStoreBase, SparseStoreBase):
    """
    Single store that handles dense, sparse, and scalar filtering.

    Implements both VectorStoreBase and SparseStoreBase so HybridRetriever
    can use one instance for both roles:

        store = SupabaseStore()
        retriever = HybridRetriever(
            vector_store=store,
            sparse_store=store,   # same instance
            embedder=embedder,
            reranker=reranker,
        )

    The retrieve() method on HybridRetriever calls both .search() signatures —
    dense via embed+cosine, sparse via full-text — but since both hit the same
    Postgres table, you can optionally bypass the Python-side RRF and use the
    native SQL hybrid query instead (see hybrid_search() below).
    """

    def __init__(self):
        print(config)
        self.conn = psycopg2.connect(config.supabase.connection_string.get_secret_value())
        self.conn.autocommit = False
        register_vector(self.conn)   # enables vector <=> operator
        self.table = config.supabase.table_name
        logger.info(f"SupabaseStore connected → table: {self.table}")

    # ── VectorStoreBase ───────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        """
        Insert or update chunks. Uses ON CONFLICT DO UPDATE so re-ingesting
        the same doc updates content without duplicating rows.
        """
        if not chunks:
            return

        sql = f"""
            INSERT INTO {self.table} (
                chunk_id, doc_id, content, embedding,
                page, section, doc_type, version, extra_metadata
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                content        = EXCLUDED.content,
                embedding      = EXCLUDED.embedding,
                page           = EXCLUDED.page,
                section        = EXCLUDED.section,
                doc_type       = EXCLUDED.doc_type,
                version        = EXCLUDED.version,
                extra_metadata = EXCLUDED.extra_metadata,
                tsv            = to_tsvector('english', EXCLUDED.content)
        """

        rows = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding — skipping")
                continue

            meta = chunk.metadata
            page     = str(meta.get("page", ""))
            section  = str(meta.get("section", ""))
            doc_type = str(meta.get("doc_type", ""))
            version  = str(meta.get("version", ""))
            extra    = {k: v for k, v in meta.items()
                        if k not in ("page", "section", "doc_type", "version")}

            rows.append((
                chunk.chunk_id,
                chunk.doc_id,
                chunk.content,
                chunk.embedding,
                page, section, doc_type, version,
                json.dumps(extra),
            ))

        if rows:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, sql, rows)
            self.conn.commit()
            logger.info(f"Upserted {len(rows)} chunks into Supabase")

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Dense cosine similarity search via pgvector. Called by VectorStoreBase."""
        return self.search_dense(query_embedding, top_k, filter)

    def search_dense(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Dense cosine similarity search via pgvector."""
        where_clause, where_values = self._build_where(filter)

        sql = f"""
            SELECT
                chunk_id, doc_id, content,
                page, section, doc_type, version, extra_metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, [query_embedding] + where_values + [query_embedding, top_k])
            rows = cur.fetchall()

        return [self._row_to_retrieved(row, source="dense") for row in rows]

    def delete(self, chunk_ids: list[str]) -> None:
        placeholders = ",".join(["%s"] * len(chunk_ids))
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table} WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        self.conn.commit()
        logger.info(f"Deleted {len(chunk_ids)} chunks from Supabase")

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]

    # ── SparseStoreBase ───────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """No-op — tsvector maintained automatically by Postgres trigger."""
        logger.info("tsvector maintained by Postgres trigger, no action needed")

    def search_sparse(
        self,
        query: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Sparse full-text search via Postgres tsvector/tsquery.
        Called explicitly — avoids Python overload confusion with search_dense.
        """
        where_clause, where_values = self._build_where(filter)
        tsv_filter = "tsv @@ plainto_tsquery('english', %s)"
        if where_clause:
            full_where = f"{where_clause} AND {tsv_filter}"
        else:
            full_where = f"WHERE {tsv_filter}"

        sql = f"""
            SELECT
                chunk_id, doc_id, content,
                page, section, doc_type, version, extra_metadata,
                ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
            FROM {self.table}
            {full_where}
            ORDER BY score DESC
            LIMIT %s
        """

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, [query] + where_values + [query, top_k])
            rows = cur.fetchall()

        return [self._row_to_retrieved(row, source="sparse") for row in rows]

    # ── Native SQL Hybrid Search (bypasses Python-side RRF) ──────────────────

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        rrf_k: int = 60,
        filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid search in a single SQL query using RRF.
        More efficient than Python-side fusion — one round trip to Postgres.

        This is an optional enhancement. To use it, call this directly instead
        of going through HybridRetriever's retrieve() method.
        """
        where_clause, where_values = self._build_where(filter)

        sql = f"""
            WITH dense AS (
                SELECT
                    chunk_id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                FROM {self.table}
                {where_clause}
            ),
            sparse AS (
                SELECT
                    chunk_id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC
                    ) AS rank
                FROM {self.table}
                {where_clause}
                WHERE tsv @@ plainto_tsquery('english', %s)
            ),
            rrf AS (
                SELECT
                    COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                    COALESCE(1.0 / (%s + d.rank), 0) +
                    COALESCE(1.0 / (%s + s.rank), 0) AS rrf_score
                FROM dense d
                FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
            )
            SELECT
                c.chunk_id, c.doc_id, c.content,
                c.page, c.section, c.doc_type, c.version, c.extra_metadata,
                rrf.rrf_score AS score
            FROM rrf
            JOIN {self.table} c ON c.chunk_id = rrf.chunk_id
            ORDER BY rrf_score DESC
            LIMIT %s
        """

        params = (
            [query_embedding]           # dense embedding
            + where_values              # dense WHERE
            + [query_text]             # sparse ts_rank query
            + where_values              # sparse WHERE
            + [query_text]             # sparse tsquery match
            + [rrf_k, rrf_k]           # RRF k constants
            + [top_k]                  # LIMIT
        )

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [self._row_to_retrieved(row, source="hybrid") for row in rows]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_where(self, filter: dict | None) -> tuple[str, list]:
        """Build a SQL WHERE clause from a filter dict."""
        if not filter:
            return "", []
        conditions = [f"{col} = %s" for col in filter]
        return "WHERE " + " AND ".join(conditions), list(filter.values())

    def _row_to_retrieved(self, row: dict, source: str) -> RetrievedChunk:
        """Convert a Postgres row to a RetrievedChunk."""
        # psycopg2 auto-deserializes JSONB → dict, str stays str
        extra = row["extra_metadata"]
        if isinstance(extra, str):
            extra = json.loads(extra)
        extra = extra or {}

        metadata = {
            **extra,
            "page":     row["page"],
            "section":  row["section"],
            "doc_type": row["doc_type"],
            "version":  row["version"],
            "source":   extra.get("source", ""),
        }
        chunk = Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            content=row["content"],
            metadata=metadata,
        )
        return RetrievedChunk(chunk=chunk, score=float(row["score"]), source=source)

    def close(self):
        self.conn.close()
        logger.info("SupabaseStore connection closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
