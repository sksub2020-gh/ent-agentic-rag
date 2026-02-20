-- Create schema
CREATE SCHEMA IF NOT EXISTS dev;

SET search_path = dev;

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Table
CREATE TABLE IF NOT EXISTS rag_chunks (
  chunk_id       TEXT PRIMARY KEY,
  doc_id         TEXT NOT NULL,
  content        TEXT NOT NULL,
  embedding      vector(768),
  page           TEXT DEFAULT '',
  section        TEXT DEFAULT '',
  doc_type       TEXT DEFAULT '',
  version        TEXT DEFAULT '',
  extra_metadata JSONB DEFAULT '{}'::jsonb,
  tsv            tsvector,
  created_at     timestamptz DEFAULT now(),
  updated_at     timestamptz DEFAULT now()
);

-- HNSW index (pgvector)
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
  ON rag_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Full-text GIN index (on tsv column)
CREATE INDEX IF NOT EXISTS rag_chunks_tsv_idx
  ON rag_chunks
  USING gin (tsv);

-- B-tree indexes
CREATE INDEX IF NOT EXISTS rag_chunks_doc_id_idx   ON rag_chunks (doc_id);
CREATE INDEX IF NOT EXISTS rag_chunks_doc_type_idx ON rag_chunks (doc_type);
CREATE INDEX IF NOT EXISTS rag_chunks_version_idx  ON rag_chunks (version);
CREATE INDEX IF NOT EXISTS rag_chunks_page_idx     ON rag_chunks (page);

-- Trigger function: update tsv and updated_at
CREATE OR REPLACE FUNCTION rag_chunks_tsv_update()
RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS rag_chunks_tsv_trigger ON rag_chunks;

CREATE TRIGGER rag_chunks_tsv_trigger
  BEFORE INSERT OR UPDATE
  ON rag_chunks
  FOR EACH ROW
  EXECUTE FUNCTION rag_chunks_tsv_update();

-- RLS note: disabled for internal/service use. Enable+add policies before exposing to anon users.
ALTER TABLE rag_chunks DISABLE ROW LEVEL SECURITY;

 SELECT extname FROM pg_extension WHERE extname = 'vector';

 SELECT indexname FROM pg_indexes WHERE tablename = 'rag_chunks';