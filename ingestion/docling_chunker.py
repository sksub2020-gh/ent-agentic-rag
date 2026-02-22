"""
IBM Docling Hybrid Chunker — handles PDF, HTML, Tables, Charts.
Implements ChunkerBase so it's swappable.
"""
import hashlib
import logging
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
# from docling_core.transforms.chunker.tokenizer import OpenAITokenizerWrapper

from core.interfaces import ChunkerBase, Document, Chunk
from config.settings import config

logger = logging.getLogger(__name__)


class DoclingHybridChunker(ChunkerBase):
    """
    Wraps IBM Docling's HybridChunker.

    HybridChunker = semantic awareness (doc structure) + token-aware splitting.
    It respects headings, tables, lists — doesn't blindly split mid-sentence.
    """

    def __init__(self, tokenizer, ocr: bool = False):
        """
        Args:
            ocr: Enable EasyOCR for scanned/image-based PDFs.
                 Needed to extract text from image tables and appendices.
                 Slower — only enable when re-ingesting docs with image content.
                 Set via DOCLING_OCR=true in .env or pass ocr=True directly.
        """
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        ocr_enabled = ocr or config.docling.ocr
        pipeline_options = PdfPipelineOptions(do_ocr=ocr_enabled)

        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=config.docling.max_tokens,
            min_tokens=config.docling.min_tokens,
            overlap_tokens=config.docling.overlap_tokens,
        )
        logger.info(f"DoclingHybridChunker ready — OCR: {'enabled' if ocr_enabled else 'disabled'}")

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Convert and chunk a document file.
        document.source = file path or URL.
        """
        logger.info(f"Chunking: {document.source} [{document.doc_type}]")

        # Docling handles PDF, HTML, DOCX, tables automatically
        docling_doc = self.converter.convert(document.source).document
        raw_chunks = list(self.chunker.chunk(docling_doc))

        chunks = []
        for i, raw_chunk in enumerate(raw_chunks):
            content = raw_chunk.text.strip()
            if not content:
                continue

            chunk_id = self._make_chunk_id(document.doc_id, i)

            # Enrich metadata from Docling's structural info
            metadata = {
                **document.metadata,
                "source": document.source,
                "doc_type": document.doc_type,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "page": "1",    # default to page 1 — overridden below if available
            }

            # Docling exposes page numbers and section headings where available
            if hasattr(raw_chunk, "meta") and raw_chunk.meta:
                if hasattr(raw_chunk.meta, "headings") and raw_chunk.meta.headings:
                    metadata["section"] = raw_chunk.meta.headings[-1]  # nearest heading
                # page_no can be an int or a PageItem — handle both
                if hasattr(raw_chunk.meta, "page_no") and raw_chunk.meta.page_no is not None:
                    metadata["page"] = str(raw_chunk.meta.page_no)
                elif hasattr(raw_chunk.meta, "origin") and raw_chunk.meta.origin:
                    origin = raw_chunk.meta.origin
                    if hasattr(origin, "page_no") and origin.page_no is not None:
                        metadata["page"] = str(origin.page_no)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=content,
                metadata=metadata,
            ))

        logger.info(f"  → {len(chunks)} chunks created from {document.source}")
        return chunks

    def chunk_from_file(self, file_path: str) -> list[Chunk]:
        """Convenience: auto-detect doc_type from file extension."""
        path = Path(file_path)
        ext_to_type = {".pdf": "pdf", ".html": "html", ".htm": "html", ".docx": "docx"}
        doc_type = ext_to_type.get(path.suffix.lower(), "text")

        document = Document(
            doc_id=self._make_doc_id(file_path),
            content="",               # Docling reads from source directly
            source=file_path,
            doc_type=doc_type,
        )
        return self.chunk(document)

    @staticmethod
    def _make_doc_id(source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()[:12]

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        return f"{doc_id}_chunk_{index:04d}"
