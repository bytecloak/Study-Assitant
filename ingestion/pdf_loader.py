"""
ingestion/pdf_loader.py
-----------------------
Handles loading, chunking, and preparing PDF documents for the vector store.

Responsibilities:
- Accept one or more PDF file paths
- Extract text page by page using PyPDFLoader
- Split text into overlapping chunks using RecursiveCharacterTextSplitter
- Attach rich metadata to every chunk (source, page, doc_type, etc.)
- Return a flat list of LangChain Document objects ready to be embedded
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration — tweak these to tune retrieval quality
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800        # characters per chunk
                        # Smaller = more precise retrieval, less context per chunk
                        # Larger = more context per chunk, but noisier retrieval

CHUNK_OVERLAP = 100     # characters shared between adjacent chunks
                        # Prevents important sentences from being cut in half at boundaries


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF and return a list of page-level Document objects.

    Each Document has:
        page_content : str  — raw text of that page
        metadata     : dict — source filename, page number, doc_type

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        ValueError:        If the file is not a .pdf.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    loader = PyPDFLoader(str(path))
    pages: List[Document] = loader.load()  # returns one Document per page

    # Enrich metadata on every page-level document
    for page in pages:
        page.metadata.update({
            "source"    : path.name,        # e.g. "chapter1.pdf"
            "doc_type"  : "pdf",
            "file_path" : str(path.resolve()),
        })
        # PyPDFLoader already sets metadata["page"] (0-indexed int)
        # We add a human-friendly version for display in the UI
        page.metadata["page_label"] = f"Page {page.metadata.get('page', 0) + 1}"

    return pages


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of page-level Documents into smaller overlapping chunks.

    Why chunk?
    ----------
    Embedding models and LLM context windows have token limits. Chunking
    lets us embed focused units of meaning, so retrieval is more precise.
    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — preserving natural language boundaries.

    Args:
        documents: Page-level Documents returned by load_pdf().

    Returns:
        List of chunk-level Documents. Metadata from the parent page is
        preserved on every chunk, plus a "chunk_index" field is added.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split at these boundaries in order (most to least preferred)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Document] = splitter.split_documents(documents)

    # Tag each chunk with its position so we can display it in the UI
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def ingest_pdf(file_path: str) -> List[Document]:
    """
    Full ingestion pipeline for a single PDF.

    Steps:
        1. Load PDF → page-level Documents
        2. Chunk pages → smaller overlapping Documents
        3. Return chunks ready to be embedded and stored

    This is the main function you'll call from store_manager.py or app.py.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunk-level Document objects with metadata.

    Example:
        >>> chunks = ingest_pdf("notes/chapter1.pdf")
        >>> print(len(chunks), "chunks ready for embedding")
    """
    print(f"[PDF Loader] Loading: {file_path}")
    pages = load_pdf(file_path)
    print(f"[PDF Loader] Loaded {len(pages)} pages")

    chunks = chunk_documents(pages)
    print(f"[PDF Loader] Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks


def ingest_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Ingest multiple PDF files and return all chunks in a single flat list.

    Skips files that fail (logs the error) so one bad PDF doesn't
    break ingestion of the entire batch.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of chunks from all successfully ingested PDFs.

    Example:
        >>> paths = ["syllabus.pdf", "week1_notes.pdf", "week2_notes.pdf"]
        >>> all_chunks = ingest_multiple_pdfs(paths)
    """
    all_chunks: List[Document] = []

    for path in file_paths:
        try:
            chunks = ingest_pdf(path)
            all_chunks.extend(chunks)
        except (FileNotFoundError, ValueError) as e:
            print(f"[PDF Loader] Skipping {path} — {e}")
        except Exception as e:
            print(f"[PDF Loader] Unexpected error with {path} — {e}")

    print(f"[PDF Loader] Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_pdf_metadata_summary(chunks: List[Document]) -> dict:
    """
    Return a summary of what was ingested — useful for displaying
    in the Streamlit sidebar after a user uploads files.

    Args:
        chunks: List of chunk-level Documents.

    Returns:
        Dict with source filenames as keys, chunk counts as values.

    Example return:
        {
            "chapter1.pdf": 34,
            "syllabus.pdf": 12,
        }
    """
    summary = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        summary[source] = summary.get(source, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify ingestion works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    test_path = sys.argv[1]
    chunks = ingest_pdf(test_path)

    print("\n--- Sample chunk (first) ---")
    print("Content:", chunks[0].page_content[:300])
    print("Metadata:", chunks[0].metadata)

    print("\n--- Sample chunk (last) ---")
    print("Content:", chunks[-1].page_content[:300])
    print("Metadata:", chunks[-1].metadata)

    print("\n--- Ingestion Summary ---")
    summary = get_pdf_metadata_summary(chunks)
    for source, count in summary.items():
        print(f"  {source}: {count} chunks")