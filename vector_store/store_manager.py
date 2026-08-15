"""
vectorstore/store_manager.py
----------------------------
Manages the ChromaDB vector store for the Study Assistant.

Responsibilities:
- Initialize and persist a ChromaDB collection
- Embed and add Document chunks from any loader (PDF, YouTube, Web)
- Retrieve semantically relevant chunks for a query
- List, inspect, and delete ingested sources
- Provide metadata summaries for the Streamlit sidebar

Why ChromaDB?
-------------
- Runs fully locally — no API key, no account, no cost
- Persists to disk automatically between app restarts
- Supports metadata filtering (filter by doc_type, source, etc.)
- Native LangChain integration via langchain-chroma

Install dependencies:
    pip install chromadb langchain-chroma langchain_google_genai google_genailangchain_google_genai python-dotenv
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where ChromaDB persists its data between sessions.
# The directory is created automatically if it doesn't exist.
CHROMA_PERSIST_DIR = "./chroma_store"

# Name of the ChromaDB collection.
# Think of this like a table name in a database.
COLLECTION_NAME = "study_assistant"

# Embedding model — text-embedding-3-small is cheap, fast, and high quality.
# Costs ~$0.00002 per 1K tokens — essentially free for a study project.
EMBEDDING_MODEL = "gemini-embedding-001"

# Default number of chunks to retrieve per query.
# Higher k = more context for the LLM, but slower and more expensive.
DEFAULT_RETRIEVAL_K = 5


# ---------------------------------------------------------------------------
# Core Class
# ---------------------------------------------------------------------------

class StoreManager:
    """
    Wraps ChromaDB and google_genailangchain_google_genai embeddings into a clean interface
    for the Study Assistant.

    Usage:
        store = StoreManager()
        store.add_documents(chunks)             # from any loader
        results = store.retrieve("what is X")   # returns top-k chunks
        store.list_sources()                    # see what's been ingested

    The store persists automatically — restart the app and your
    documents are still there.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        """
        Initialize the vector store, creating it if it doesn't exist
        or loading the existing one from disk.

        Args:
            persist_dir:      Path where ChromaDB stores its data.
            collection_name:  Name of the ChromaDB collection to use.
            embedding_model:  google_genailangchain_google_genai embedding model name.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Validate API key early — better to fail here than mid-ingestion
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found in environment.\n"
                "Add it to your .env file: GOOGLE_API_KEY=sk-..."
            )

        print(f"[Store Manager] Initializing embeddings ({embedding_model})")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            GOOGLE_API_KEY=api_key,
        )

        # Initialize or load the ChromaDB collection
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Store Manager] Loading collection '{collection_name}' "
              f"from {persist_dir}")

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        count = self._get_document_count()
        print(f"[Store Manager] Ready. Collection contains {count} chunks.")


    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and add a list of Document chunks to the vector store.

        Handles deduplication — if a source has already been ingested,
        it will be replaced rather than duplicated. This lets users
        re-upload an updated version of a document cleanly.

        Args:
            documents: List of Document objects from any loader.

        Returns:
            Number of chunks successfully added.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("No documents provided to add_documents().")

        # Check for existing sources and remove them first (deduplication)
        sources_in_batch = set(
            doc.metadata.get("source", "") for doc in documents
        )
        for source in sources_in_batch:
            if source and self._source_exists(source):
                print(f"[Store Manager] Replacing existing source: '{source}'")
                self.delete_source(source)

        print(f"[Store Manager] Embedding {len(documents)} chunks...")

        # ChromaDB's add_documents handles batching internally
        self.vectorstore.add_documents(documents)

        print(f"[Store Manager] Successfully added {len(documents)} chunks.")
        return len(documents)


    def add_documents_from_loaders(
        self,
        pdf_paths: Optional[List[str]] = None,
        youtube_urls: Optional[List[str]] = None,
        web_urls: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Convenience method — run multiple loaders and ingest all results
        in one call. Useful for the initial setup or batch ingestion.

        Args:
            pdf_paths:     List of PDF file paths.
            youtube_urls:  List of YouTube URLs.
            web_urls:      List of website URLs.

        Returns:
            Dict reporting chunks added per source type.
            Example: {"pdf": 45, "youtube": 28, "website": 12}

        Example:
            >>> store.add_documents_from_loaders(
            ...     pdf_paths=["notes.pdf", "syllabus.pdf"],
            ...     youtube_urls=["https://youtu.be/VIDEO_ID"],
            ...     web_urls=["https://en.wikipedia.org/wiki/Transformer"],
            ... )
        """
        # Import here to avoid circular imports at module level
        from ingestion.pdf_loader import ingest_multiple_pdfs
        from ingestion.youtube_loader import ingest_multiple_youtube
        from ingestion.web_loader import ingest_multiple_urls

        results = {}

        if pdf_paths:
            chunks = ingest_multiple_pdfs(pdf_paths)
            if chunks:
                self.add_documents(chunks)
                results["pdf"] = len(chunks)

        if youtube_urls:
            chunks = ingest_multiple_youtube(youtube_urls)
            if chunks:
                self.add_documents(chunks)
                results["youtube"] = len(chunks)

        if web_urls:
            chunks = ingest_multiple_urls(web_urls)
            if chunks:
                self.add_documents(chunks)
                results["website"] = len(chunks)

        total = sum(results.values())
        print(f"[Store Manager] Batch ingestion complete. "
              f"Total chunks added: {total}")
        return results


    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve the top-k most semantically relevant chunks for a query.

        Supports optional metadata filtering to restrict retrieval to
        a specific document type or source — useful when a user wants
        to quiz themselves only on their PDF notes, for example.

        Args:
            query:    Natural language query string.
            k:        Number of chunks to return.
            doc_type: Optional filter — "pdf", "youtube", or "website".
            source:   Optional filter — exact source name (filename or title).

        Returns:
            List of Document objects sorted by relevance (most relevant first).

        Example:
            # Retrieve from all sources
            >>> results = store.retrieve("what is backpropagation", k=5)

            # Retrieve only from PDFs
            >>> results = store.retrieve("gradient descent", doc_type="pdf")

            # Retrieve only from a specific source
            >>> results = store.retrieve("attention", source="lecture1.pdf")
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # Build metadata filter if requested
        where_filter = self._build_filter(doc_type=doc_type, source=source)

        if where_filter:
            print(f"[Store Manager] Retrieving with filter: {where_filter}")
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=where_filter,
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        print(f"[Store Manager] Retrieved {len(results)} chunks for query: "
              f"'{query[:60]}{'...' if len(query) > 60 else ''}'")
        return results


    def retrieve_with_scores(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[tuple[Document, float]]:
        """
        Like retrieve(), but also returns the similarity score for each chunk.

        Scores are cosine similarity values — higher is more relevant.
        Typical range: 0.0 (unrelated) to 1.0 (identical).
        Useful for the "confidence" display in the Streamlit UI and
        for the hallucination guard in Phase 3.

        Args:
            query:    Natural language query string.
            k:        Number of chunks to return.
            doc_type: Optional metadata filter.
            source:   Optional metadata filter.

        Returns:
            List of (Document, score) tuples, sorted by score descending.

        Example:
            >>> results = store.retrieve_with_scores("what is attention?")
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f} | Source: {doc.metadata['source']}")
        """
        where_filter = self._build_filter(doc_type=doc_type, source=source)

        if where_filter:
            results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=where_filter
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Sort by score descending (most relevant first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


    def get_retriever(
        self,
        k: int = DEFAULT_RETRIEVAL_K,
        doc_type: Optional[str] = None,
    ):
        """
        Return a LangChain-compatible retriever object.

        Use this when plugging the store into LangChain chains like
        RetrievalQA or ConversationalRetrievalChain — they expect
        a retriever, not raw similarity_search calls.

        Args:
            k:        Number of chunks to retrieve per query.
            doc_type: Optional metadata filter.

        Returns:
            LangChain BaseRetriever instance.

        Example:
            >>> retriever = store.get_retriever(k=5)
            >>> qa_chain = RetrievalQA.from_chain_type(
            ...     llm=llm,
            ...     retriever=retriever,
            ... )
        """
        search_kwargs: Dict[str, Any] = {"k": k}

        if doc_type:
            search_kwargs["filter"] = {"doc_type": {"$eq": doc_type}}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


    # -----------------------------------------------------------------------
    # Source Management
    # -----------------------------------------------------------------------

    def list_sources(self) -> List[Dict[str, Any]]:
        """
        Return a list of all ingested sources with their metadata.

        Returns one entry per unique source (not per chunk), with:
        - source name
        - doc_type (pdf / youtube / website)
        - chunk count
        - url (for web and YouTube sources)

        Useful for populating the Streamlit sidebar document list.

        Returns:
            List of dicts, one per unique source.

        Example return:
            [
                {"source": "chapter1.pdf", "doc_type": "pdf", "chunks": 34},
                {"source": "CS50 Lecture 1", "doc_type": "youtube",
                 "chunks": 28, "url": "https://youtu.be/..."},
            ]
        """
        try:
            # Pull all stored metadata from ChromaDB
            collection_data = self.vectorstore.get(include=["metadatas"])
            metadatas = collection_data.get("metadatas", [])
        except Exception:
            return []

        # Aggregate by source
        sources: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            source = meta.get("source", "Unknown")
            if source not in sources:
                sources[source] = {
                    "source"   : source,
                    "doc_type" : meta.get("doc_type", "unknown"),
                    "chunks"   : 0,
                    "url"      : meta.get("url", ""),
                    "channel"  : meta.get("channel", ""),
                    "domain"   : meta.get("domain", ""),
                }
            sources[source]["chunks"] += 1

        return sorted(sources.values(), key=lambda x: x["source"])


    def delete_source(self, source_name: str) -> int:
        """
        Delete all chunks belonging to a specific source from the store.

        Args:
            source_name: The exact source name as stored in metadata.
                         Use list_sources() to find the exact name.

        Returns:
            Number of chunks deleted.

        Example:
            >>> store.delete_source("chapter1.pdf")
        """
        try:
            # Get IDs of all chunks matching this source
            collection_data = self.vectorstore.get(
                where={"source": {"$eq": source_name}},
                include=["metadatas"],
            )
            ids_to_delete = collection_data.get("ids", [])

            if not ids_to_delete:
                print(f"[Store Manager] No chunks found for source: '{source_name}'")
                return 0

            self.vectorstore.delete(ids=ids_to_delete)
            print(f"[Store Manager] Deleted {len(ids_to_delete)} chunks "
                  f"for source: '{source_name}'")
            return len(ids_to_delete)

        except Exception as e:
            print(f"[Store Manager] Error deleting source '{source_name}': {e}")
            return 0


    def clear_all(self) -> None:
        """
        Delete ALL documents from the vector store.

        Use with caution — this cannot be undone without re-ingesting.
        Useful for resetting during development or testing.
        """
        try:
            collection_data = self.vectorstore.get()
            all_ids = collection_data.get("ids", [])

            if not all_ids:
                print("[Store Manager] Collection is already empty.")
                return

            self.vectorstore.delete(ids=all_ids)
            print(f"[Store Manager] Cleared all {len(all_ids)} chunks "
                  f"from the collection.")
        except Exception as e:
            print(f"[Store Manager] Error clearing collection: {e}")


    # -----------------------------------------------------------------------
    # Utility / Introspection
    # -----------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a high-level summary of the vector store contents.

        Useful for displaying a status panel in the Streamlit sidebar.

        Returns:
            Dict with total chunks, source count, and breakdown by doc_type.

        Example return:
            {
                "total_chunks": 89,
                "total_sources": 4,
                "by_type": {
                    "pdf": {"sources": 2, "chunks": 45},
                    "youtube": {"sources": 1, "chunks": 28},
                    "website": {"sources": 1, "chunks": 16},
                }
            }
        """
        sources = self.list_sources()

        by_type: Dict[str, Dict[str, int]] = {}
        for s in sources:
            dt = s["doc_type"]
            if dt not in by_type:
                by_type[dt] = {"sources": 0, "chunks": 0}
            by_type[dt]["sources"] += 1
            by_type[dt]["chunks"] += s["chunks"]

        return {
            "total_chunks"  : sum(s["chunks"] for s in sources),
            "total_sources" : len(sources),
            "by_type"       : by_type,
        }


    def _source_exists(self, source_name: str) -> bool:
        """Check if any chunks with this source name already exist."""
        try:
            result = self.vectorstore.get(
                where={"source": {"$eq": source_name}},
                include=["metadatas"],
                limit=1,
            )
            return len(result.get("ids", [])) > 0
        except Exception:
            return False


    def _get_document_count(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0


    def _build_filter(
        self,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Build a ChromaDB metadata filter dict from optional parameters.

        ChromaDB uses a MongoDB-style filter syntax.
        Returns None if no filters are specified.
        """
        filters = []

        if doc_type:
            filters.append({"doc_type": {"$eq": doc_type}})
        if source:
            filters.append({"source": {"$eq": source}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]

        # Combine multiple filters with $and
        return {"$and": filters}


# ---------------------------------------------------------------------------
# Module-level singleton factory
# ---------------------------------------------------------------------------

_store_instance: Optional[StoreManager] = None


def get_store() -> StoreManager:
    """
    Return a module-level singleton StoreManager instance.

    Use this in app.py and chain files instead of instantiating
    StoreManager directly. Ensures only one ChromaDB connection
    is open at a time, which avoids locking issues.

    Example:
        from vectorstore.store_manager import get_store

        store = get_store()
        results = store.retrieve("what is gradient descent")
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = StoreManager()
    return _store_instance


# ---------------------------------------------------------------------------
# Quick test — run directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_core.documents import Document

    print("=== StoreManager Quick Test ===\n")

    store = StoreManager()

    # Create dummy documents
    test_docs = [
        Document(
            page_content="Gradient descent is an optimization algorithm used to minimize a loss function by iteratively moving in the direction of steepest descent.",
            metadata={"source": "test_notes.pdf", "doc_type": "pdf", "page_label": "Page 1", "chunk_index": 0}
        ),
        Document(
            page_content="Backpropagation computes the gradient of the loss function with respect to each weight using the chain rule of calculus.",
            metadata={"source": "test_notes.pdf", "doc_type": "pdf", "page_label": "Page 2", "chunk_index": 1}
        ),
        Document(
            page_content="The attention mechanism allows models to focus on relevant parts of the input when generating each output token.",
            metadata={"source": "CS50 AI Lecture", "doc_type": "youtube", "url": "https://youtu.be/test", "chunk_index": 0}
        ),
    ]

    # Test: Add documents
    print("--- Adding test documents ---")
    store.add_documents(test_docs)

    # Test: List sources
    print("\n--- Sources in store ---")
    for s in store.list_sources():
        print(f"  {s['doc_type'].upper()} | {s['source']} | {s['chunks']} chunks")

    # Test: Retrieve
    print("\n--- Retrieval test ---")
    results = store.retrieve("how does gradient descent work?", k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content: {doc.page_content[:150]}...")

    # Test: Retrieve with scores
    print("\n--- Retrieval with scores ---")
    scored = store.retrieve_with_scores("what is attention?", k=2)
    for doc, score in scored:
        print(f"  Score: {score:.4f} | Source: {doc.metadata['source']}")

    # Test: Summary
    print("\n--- Store summary ---")
    summary = store.get_summary()
    print(f"  Total chunks: {summary['total_chunks']}")
    print(f"  Total sources: {summary['total_sources']}")
    for dtype, info in summary['by_type'].items():
        print(f"  {dtype}: {info['sources']} sources, {info['chunks']} chunks")

    # Test: Delete
    print("\n--- Deleting test source ---")
    store.delete_source("test_notes.pdf")
    print(f"  Remaining chunks: {store._get_document_count()}")

    print("\n=== All tests passed ===")