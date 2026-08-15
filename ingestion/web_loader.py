"""
ingestion/web_loader.py
-----------------------
Handles fetching, cleaning, and chunking web page content
for the Study Assistant vector store.

Responsibilities:
- Accept a single URL or a list of URLs
- Fetch page HTML with proper headers (avoiding bot-blocking)
- Clean HTML → extract readable text (remove nav, ads, footers, scripts)
- Detect and handle special content types: articles, Wikipedia, arXiv papers
- Split into overlapping chunks with rich metadata
- Return LangChain Document objects ready to be embedded

Supported content types:
    - General web articles / blog posts
    - Wikipedia pages
    - arXiv paper abstract pages (arxiv.org/abs/...)
    - Documentation pages (ReadTheDocs, GitBook, etc.)

Install dependencies:
    pip install requests beautifulsoup4 langchain langchain-community
    pip install trafilatura          # best-in-class article text extractor
    pip install arxiv                # optional, for arXiv metadata
"""

import re
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Realistic browser headers — many sites block requests without these
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 15        # seconds before giving up on a slow page
RATE_LIMIT_DELAY = 1.0      # seconds to wait between batch requests (be polite)


# ---------------------------------------------------------------------------
# Step 1 — URL Classification
# ---------------------------------------------------------------------------

def classify_url(url: str) -> str:
    """
    Detect the type of web page to apply the right extraction strategy.

    Returns one of:
        "arxiv"     — arXiv abstract or PDF page
        "wikipedia" — Wikipedia article
        "general"   — everything else (articles, blogs, docs)
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    if "arxiv.org" in domain:
        return "arxiv"
    if "wikipedia.org" in domain:
        return "wikipedia"
    return "general"


# ---------------------------------------------------------------------------
# Step 2 — Fetch Raw HTML
# ---------------------------------------------------------------------------

def fetch_html(url: str) -> Tuple[str, int]:
    """
    Fetch the raw HTML of a web page.

    Args:
        url: The URL to fetch.

    Returns:
        Tuple of (html_content, status_code).

    Raises:
        requests.exceptions.RequestException: On network errors.
        ValueError: If the response content-type is not HTML.
    """
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()

        # Check we actually got HTML back (not a PDF or image)
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise ValueError(
                f"Expected HTML content, got: {content_type}\n"
                f"For PDFs, use pdf_loader.py instead."
            )

        return response.text, response.status_code

    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(
            f"Request timed out after {REQUEST_TIMEOUT}s for: {url}"
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        raise requests.exceptions.HTTPError(
            f"HTTP {status} error fetching: {url}"
        )


# ---------------------------------------------------------------------------
# Step 3 — Content Extraction (per content type)
# ---------------------------------------------------------------------------

def extract_text_trafilatura(html: str, url: str) -> Optional[str]:
    """
    Use trafilatura for best-in-class article extraction.

    trafilatura is specifically designed for extracting main article content,
    filtering out navigation, ads, footers, and sidebars. It outperforms
    BeautifulSoup for most article/blog pages.

    Returns None if trafilatura fails or isn't installed, so we can fall back.
    """
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        return text if text and len(text) > 200 else None
    except ImportError:
        return None
    except Exception:
        return None


def extract_text_wikipedia(html: str) -> str:
    """
    Extract clean content from a Wikipedia page.

    Wikipedia has a consistent structure — we target the #mw-content-text
    div and remove infoboxes, references, navboxes, and edit buttons
    that pollute the main content.

    Args:
        html: Raw HTML of the Wikipedia page.

    Returns:
        Clean article text with section headers preserved.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Target the main content area
    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return extract_text_generic(html)

    # Remove noise elements
    noise_selectors = [
        ".infobox",           # side table with quick facts
        ".navbox",            # bottom navigation boxes
        ".reflist",           # references section
        ".reference",         # inline citation markers [1]
        ".mw-editsection",    # [edit] buttons
        ".hatnote",           # disambiguation notes
        ".thumb",             # image captions
        "table.wikitable",    # data tables (often not useful for study)
        "#toc",               # table of contents
        ".mw-references-wrap",
    ]
    for selector in noise_selectors:
        for el in content_div.select(selector):
            el.decompose()

    # Extract text, preserving section headers
    lines = []
    for element in content_div.find_all(["h2", "h3", "h4", "p", "li", "ul"]):
        text = element.get_text(separator=" ").strip()
        text = re.sub(r'\s+', ' ', text)

        if not text or len(text) < 10:
            continue

        # Add visual separation for headers
        if element.name in ["h2", "h3"]:
            lines.append(f"\n## {text}\n")
        elif element.name == "h4":
            lines.append(f"\n### {text}\n")
        else:
            lines.append(text)

    return "\n".join(lines)


def extract_text_arxiv(html: str, url: str) -> str:
    """
    Extract structured content from an arXiv abstract page.

    arXiv abstract pages have a consistent layout. We extract:
    - Paper title
    - Authors
    - Abstract text
    - Subjects/categories

    Note: We don't fetch the full PDF by default — the abstract is usually
    sufficient for study/retrieval. The PDF URL is stored in metadata.

    Args:
        html: Raw HTML of the arXiv page.
        url:  Original URL (used to construct PDF link).

    Returns:
        Structured text representation of the paper abstract.
    """
    soup = BeautifulSoup(html, "html.parser")

    sections = []

    # Title
    title_el = soup.find("h1", class_="title")
    if title_el:
        title = title_el.get_text().replace("Title:", "").strip()
        sections.append(f"Paper Title: {title}")

    # Authors
    authors_el = soup.find("div", class_="authors")
    if authors_el:
        authors = authors_el.get_text().replace("Authors:", "").strip()
        sections.append(f"Authors: {authors}")

    # Abstract
    abstract_el = soup.find("blockquote", class_="abstract")
    if abstract_el:
        abstract = abstract_el.get_text().replace("Abstract:", "").strip()
        abstract = re.sub(r'\s+', ' ', abstract)
        sections.append(f"Abstract:\n{abstract}")

    # Subjects
    subjects_el = soup.find("td", class_="tablecell subjects")
    if subjects_el:
        subjects = subjects_el.get_text().strip()
        sections.append(f"Subjects: {subjects}")

    # PDF link
    pdf_url = url.replace("/abs/", "/pdf/")
    sections.append(f"Full PDF: {pdf_url}")

    if sections:
        return "\n\n".join(sections)

    # Fallback to generic extraction
    return extract_text_generic(html)


def extract_text_generic(html: str) -> str:
    """
    General-purpose HTML → text extractor using BeautifulSoup.

    Strategy:
    1. Try trafilatura first (best quality)
    2. Fall back to manual BeautifulSoup extraction

    Removes: scripts, styles, nav, header, footer, aside, ads.
    Targets: article, main, .content, .post-body, .entry-content, etc.
    Falls back to <body> if no semantic containers found.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned text string.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags entirely
    for tag in soup(["script", "style", "noscript", "iframe",
                     "nav", "header", "footer", "aside",
                     "form", "button", "svg", "figure"]):
        tag.decompose()

    # Remove common ad/nav class patterns
    noise_classes = [
        "cookie", "banner", "popup", "modal", "newsletter",
        "sidebar", "widget", "advertisement", "social-share",
        "related-posts", "comment", "breadcrumb", "pagination",
    ]
    for cls in noise_classes:
        for el in soup.find_all(class_=re.compile(cls, re.I)):
            el.decompose()

    # Try to find the main content container
    content_candidates = [
        soup.find("article"),
        soup.find("main"),
        soup.find(class_=re.compile(r'content|post|entry|article|body', re.I)),
        soup.find("div", {"id": re.compile(r'content|main|article', re.I)}),
        soup.find("body"),
    ]

    content = next((c for c in content_candidates if c is not None), soup)

    # Extract text from content blocks
    lines = []
    for element in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
        text = element.get_text(separator=" ").strip()
        text = re.sub(r'\s+', ' ', text)

        if not text or len(text) < 15:
            continue

        if element.name in ["h1", "h2"]:
            lines.append(f"\n## {text}\n")
        elif element.name in ["h3", "h4"]:
            lines.append(f"\n### {text}\n")
        else:
            lines.append(text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 4 — Metadata Extraction
# ---------------------------------------------------------------------------

def extract_page_metadata(html: str, url: str) -> dict:
    """
    Extract page-level metadata from HTML <head> tags.

    Pulls: title, description, author, publication date, site name.
    Falls back gracefully when tags are missing.

    Args:
        html: Raw HTML string.
        url:  Original URL.

    Returns:
        Dict of metadata fields.
    """
    soup = BeautifulSoup(html, "html.parser")
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")

    def get_meta(name: str = None, property: str = None) -> str:
        """Helper to get content from <meta> tags by name or property."""
        if name:
            tag = soup.find("meta", attrs={"name": name})
        elif property:
            tag = soup.find("meta", attrs={"property": property})
        else:
            return ""
        return tag.get("content", "").strip() if tag else ""

    # Title — prefer og:title, then <title>, then h1
    title = (
        get_meta(property="og:title") or
        (soup.title.string.strip() if soup.title else "") or
        (soup.find("h1").get_text().strip() if soup.find("h1") else "") or
        url
    )

    # Clean up common title suffixes like " | Site Name" or " - Blog"
    title = re.sub(r'\s*[\|\-–—]\s*.{3,50}$', '', title).strip()

    return {
        "source"      : title or domain,
        "url"         : url,
        "domain"      : domain,
        "doc_type"    : "website",
        "description" : get_meta(name="description") or get_meta(property="og:description"),
        "author"      : get_meta(name="author") or get_meta(property="article:author"),
        "published"   : get_meta(property="article:published_time") or get_meta(name="date"),
        "site_name"   : get_meta(property="og:site_name") or domain,
    }


# ---------------------------------------------------------------------------
# Step 5 — Chunking
# ---------------------------------------------------------------------------

def chunk_web_content(text: str, metadata: dict) -> List[Document]:
    """
    Split web page text into overlapping chunks and wrap in Documents.

    Args:
        text:     Cleaned text content of the page.
        metadata: Page metadata dict from extract_page_metadata().

    Returns:
        List of chunk-level Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    documents = []

    for i, chunk_text in enumerate(raw_chunks):
        doc = Document(
            page_content=chunk_text,
            metadata={
                **metadata,         # spread all page-level metadata
                "chunk_index": i,
            }
        )
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Step 6 — Main Entry Point
# ---------------------------------------------------------------------------

def ingest_url(url: str) -> List[Document]:
    """
    Full ingestion pipeline for a single web page URL.

    Steps:
        1. Classify the URL (arXiv, Wikipedia, or general)
        2. Fetch raw HTML
        3. Extract page metadata from <head>
        4. Extract main text content using the right strategy
        5. Chunk into overlapping Documents with metadata

    Args:
        url: Full URL of the web page (must include https://).

    Returns:
        List of chunk-level Document objects ready to be embedded.

    Raises:
        ValueError:   If the URL is invalid or the page returns non-HTML.
        requests.exceptions.RequestException: On network failures.

    Example:
        >>> chunks = ingest_url("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)")
        >>> print(len(chunks), "chunks ready for embedding")
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    print(f"[Web Loader] Fetching: {url}")
    page_type = classify_url(url)
    print(f"[Web Loader] Detected type: {page_type}")

    # Fetch HTML
    html, status_code = fetch_html(url)
    print(f"[Web Loader] HTTP {status_code} — {len(html):,} characters received")

    # Extract metadata
    metadata = extract_page_metadata(html, url)
    print(f"[Web Loader] Title: {metadata['source']}")

    # Extract text using the right strategy
    if page_type == "arxiv":
        text = extract_text_arxiv(html, url)

    elif page_type == "wikipedia":
        text = extract_text_wikipedia(html)

    else:
        # Try trafilatura first, fall back to BeautifulSoup
        text = extract_text_trafilatura(html, url) or extract_text_generic(html)

    if not text or len(text.strip()) < 100:
        raise ValueError(
            f"Could not extract meaningful content from: {url}\n"
            f"The page may require JavaScript, a login, or block scrapers."
        )

    word_count = len(text.split())
    print(f"[Web Loader] Extracted {word_count} words of content")

    # Chunk
    chunks = chunk_web_content(text, metadata)
    print(f"[Web Loader] Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks


def ingest_multiple_urls(
    urls: List[str],
    rate_limit_delay: float = RATE_LIMIT_DELAY,
) -> List[Document]:
    """
    Ingest multiple URLs and return all chunks in a single flat list.

    Respects a rate limit delay between requests to avoid hammering servers.
    Skips URLs that fail (logs the error) without stopping the batch.

    Args:
        urls:              List of URLs to ingest.
        rate_limit_delay:  Seconds to wait between requests (default: 1.0).

    Returns:
        Combined list of chunks from all successfully ingested pages.

    Example:
        >>> urls = [
        ...     "https://en.wikipedia.org/wiki/Attention_mechanism",
        ...     "https://arxiv.org/abs/1706.03762",
        ...     "https://pytorch.org/tutorials/beginner/transformer_tutorial.html",
        ... ]
        >>> all_chunks = ingest_multiple_urls(urls)
    """
    all_chunks: List[Document] = []

    for i, url in enumerate(urls):
        try:
            chunks = ingest_url(url)
            all_chunks.extend(chunks)
        except ValueError as e:
            print(f"[Web Loader] Content extraction failed — skipping. {e}")
        except requests.exceptions.Timeout:
            print(f"[Web Loader] Timed out — skipping: {url}")
        except requests.exceptions.HTTPError as e:
            print(f"[Web Loader] HTTP error — skipping: {url}. {e}")
        except requests.exceptions.RequestException as e:
            print(f"[Web Loader] Network error — skipping: {url}. {e}")
        except Exception as e:
            print(f"[Web Loader] Unexpected error — skipping: {url}. {e}")

        # Rate limit between requests (skip delay after last URL)
        if i < len(urls) - 1:
            time.sleep(rate_limit_delay)

    print(f"[Web Loader] Total chunks across all URLs: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_web_metadata_summary(chunks: List[Document]) -> dict:
    """
    Return a summary of ingested web content for the Streamlit sidebar.

    Args:
        chunks: List of chunk-level Documents from web ingestion.

    Returns:
        Dict mapping page title to chunk count.

    Example:
        {
            "Attention Is All You Need": 12,
            "Transformer (deep learning) - Wikipedia": 34,
        }
    """
    summary = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Page")
        summary[source] = summary.get(source, 0) + 1
    return summary


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Quick validation before attempting a fetch.

    Returns (is_valid, error_message).
    Useful for giving immediate feedback in the Streamlit UI.

    Example:
        >>> valid, msg = validate_url("not-a-url")
        >>> print(valid, msg)
        False "URL must start with http:// or https://"
    """
    url = url.strip()

    if not url:
        return False, "URL cannot be empty."

    if not url.startswith(("http://", "https://")):
        return False, "URL must start with http:// or https://"

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "URL must include a domain (e.g. https://example.com)"
    except Exception:
        return False, "Invalid URL format."

    return True, ""


# ---------------------------------------------------------------------------
# Quick test — run directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    TEST_URLS = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://arxiv.org/abs/1706.03762",   # Attention Is All You Need
    ]

    test_url = sys.argv[1] if len(sys.argv) > 1 else TEST_URLS[0]

    print(f"Testing with: {test_url}\n")

    try:
        chunks = ingest_url(test_url)

        print("\n--- Sample chunk (first) ---")
        print("Content:", chunks[0].page_content[:400])
        print("Metadata:", chunks[0].metadata)

        print("\n--- Sample chunk (middle) ---")
        mid = len(chunks) // 2
        print("Content:", chunks[mid].page_content[:400])
        print("Metadata:", chunks[mid].metadata)

        print("\n--- Ingestion Summary ---")
        summary = get_web_metadata_summary(chunks)
        for title, count in summary.items():
            print(f"  '{title}': {count} chunks")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)