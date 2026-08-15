"""
ingestion/youtube_loader.py
---------------------------
Handles fetching, processing, and chunking YouTube video transcripts
for the Study Assistant vector store.

Responsibilities:
- Accept a YouTube URL in any common format
- Extract the video ID reliably
- Fetch the transcript using youtube-transcript-api
- Reconstruct transcript into readable text (with optional timestamps)
- Fetch video metadata (title, channel, duration) via yt-dlp (lightweight)
- Split transcript into overlapping chunks
- Attach rich metadata to every chunk
- Return LangChain Document objects ready to be embedded

Install dependencies:
    pip install youtube-transcript-api yt-dlp langchain langchain-community
"""

import re
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000       # Slightly larger than PDF chunks because transcripts
                        # are conversational — more context helps the LLM
                        # understand spoken explanations.

CHUNK_OVERLAP = 150     # Larger overlap too, since ideas in lectures often
                        # span paragraph boundaries naturally.

# Preferred transcript languages, in order of priority.
# Add more language codes if your content is multilingual.
PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]


# ---------------------------------------------------------------------------
# Step 1 — URL Parsing
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from any common URL format.

    Handles:
        https://www.youtube.com/watch?v=VIDEO_ID
        https://youtu.be/VIDEO_ID
        https://www.youtube.com/embed/VIDEO_ID
        https://www.youtube.com/shorts/VIDEO_ID
        VIDEO_ID (raw ID passed directly)

    Args:
        url: YouTube URL or raw video ID string.

    Returns:
        11-character YouTube video ID string.

    Raises:
        ValueError: If no valid video ID can be extracted.

    Example:
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    url = url.strip()

    # If it looks like a raw video ID already (11 alphanumeric chars)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url

    parsed = urlparse(url)

    # youtu.be/VIDEO_ID
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    # youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed.netloc:
        # Standard watch URL
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

        # /embed/VIDEO_ID or /shorts/VIDEO_ID
        path_parts = parsed.path.lstrip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in ("embed", "shorts", "v"):
            return path_parts[1]

    raise ValueError(
        f"Could not extract a YouTube video ID from: '{url}'\n"
        f"Supported formats: youtube.com/watch?v=ID, youtu.be/ID, "
        f"youtube.com/shorts/ID, or a raw 11-character video ID."
    )


# ---------------------------------------------------------------------------
# Step 2 — Fetch Video Metadata
# ---------------------------------------------------------------------------

def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch lightweight metadata for a YouTube video using yt-dlp.

    Returns title, channel name, duration, and the full watch URL.
    Falls back gracefully if yt-dlp is unavailable or the fetch fails.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        Dict with keys: title, channel, duration_seconds, url.
        Falls back to placeholder values if fetch fails.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        import yt_dlp  # optional but recommended

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,       # we only want metadata, not the video
            "extract_flat": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return {
            "title"            : info.get("title", f"YouTube Video ({video_id})"),
            "channel"          : info.get("uploader", "Unknown Channel"),
            "duration_seconds" : info.get("duration", 0),
            "url"              : url,
        }

    except ImportError:
        # yt-dlp not installed — use placeholder metadata
        print("[YouTube Loader] yt-dlp not installed. Skipping metadata fetch.")
        print("                 Install with: pip install yt-dlp")
        return {
            "title"            : f"YouTube Video ({video_id})",
            "channel"          : "Unknown Channel",
            "duration_seconds" : 0,
            "url"              : url,
        }

    except Exception as e:
        print(f"[YouTube Loader] Metadata fetch failed: {e}. Using fallback.")
        return {
            "title"            : f"YouTube Video ({video_id})",
            "channel"          : "Unknown Channel",
            "duration_seconds" : 0,
            "url"              : url,
        }


# ---------------------------------------------------------------------------
# Step 3 — Fetch & Reconstruct Transcript
# ---------------------------------------------------------------------------

def fetch_transcript(video_id: str) -> List[dict]:
    """
    Fetch the raw transcript segments from YouTube.

    Each segment is a dict: {"text": str, "start": float, "duration": float}
    where start and duration are in seconds.

    Tries preferred languages first, then falls back to any available
    transcript (including auto-generated captions).

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        List of transcript segment dicts.

    Raises:
        TranscriptsDisabled: If transcripts are disabled for this video.
        NoTranscriptFound:   If no transcript exists in any language.
        VideoUnavailable:    If the video is private or deleted.
    """
    try:
        # Try preferred languages first
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=PREFERRED_LANGUAGES,
        )
        return transcript

    except NoTranscriptFound:
        # Fall back to whatever language is available (auto-generated captions)
        print("[YouTube Loader] No English transcript. Trying auto-generated captions...")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try manually created transcripts first, then auto-generated
        for transcript in transcript_list:
            try:
                return transcript.fetch()
            except Exception:
                continue

        raise NoTranscriptFound(
            video_id,
            PREFERRED_LANGUAGES,
            "No transcript available in any language."
        )


def reconstruct_transcript_text(
    segments: List[dict],
    include_timestamps: bool = True,
) -> str:
    """
    Convert raw transcript segments into clean, readable text.

    Why not just join raw segments?
    --------------------------------
    Raw segments are sentence fragments (YouTube splits at ~5-word intervals).
    Joining naively creates a wall of text with no paragraph breaks.
    We re-group by injecting paragraph breaks every ~60 seconds of content,
    which mirrors natural topic shifts in lectures.

    Args:
        segments:           Raw transcript segments from fetch_transcript().
        include_timestamps: If True, prepend [MM:SS] timestamps to each
                            paragraph. Useful for citations.

    Returns:
        Clean transcript string with paragraph breaks.
    """
    if not segments:
        return ""

    PARAGRAPH_BREAK_SECONDS = 60    # Start a new paragraph every ~60 seconds

    paragraphs = []
    current_para_texts = []
    current_para_start = segments[0]["start"]
    last_segment_end = current_para_start

    for seg in segments:
        text = seg["text"].strip()
        start = seg["start"]

        # Clean up common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)     # remove [Music], [Applause], etc.
        text = re.sub(r'\s+', ' ', text)        # collapse whitespace
        text = text.strip()

        if not text:
            continue

        # Check if we should start a new paragraph
        gap = start - last_segment_end
        elapsed = start - current_para_start

        should_break = (
            elapsed >= PARAGRAPH_BREAK_SECONDS or  # time-based break
            gap > 3.0                               # pause of > 3 seconds = topic shift
        )

        if should_break and current_para_texts:
            # Flush current paragraph
            paragraph = " ".join(current_para_texts)
            if include_timestamps:
                timestamp = _seconds_to_timestamp(current_para_start)
                paragraph = f"[{timestamp}] {paragraph}"
            paragraphs.append(paragraph)

            # Start new paragraph
            current_para_texts = [text]
            current_para_start = start
        else:
            current_para_texts.append(text)

        last_segment_end = start + seg.get("duration", 0)

    # Flush the final paragraph
    if current_para_texts:
        paragraph = " ".join(current_para_texts)
        if include_timestamps:
            timestamp = _seconds_to_timestamp(current_para_start)
            paragraph = f"[{timestamp}] {paragraph}"
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def _seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS or HH:MM:SS format.

    Example:
        >>> _seconds_to_timestamp(3723)
        '1:02:03'
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Step 4 — Chunk the Transcript
# ---------------------------------------------------------------------------

def chunk_transcript(
    text: str,
    metadata: dict,
) -> List[Document]:
    """
    Split the full transcript text into overlapping chunks and wrap
    each chunk in a LangChain Document with metadata.

    Args:
        text:     Full reconstructed transcript string.
        metadata: Video metadata dict from fetch_video_metadata().

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
                "source"           : metadata["title"],
                "doc_type"         : "youtube",
                "url"              : metadata["url"],
                "channel"          : metadata["channel"],
                "duration_seconds" : metadata["duration_seconds"],
                "chunk_index"      : i,
                # Extract timestamp from chunk text if present (e.g. "[4:32]")
                "timestamp"        : _extract_first_timestamp(chunk_text),
            }
        )
        documents.append(doc)

    return documents


def _extract_first_timestamp(text: str) -> Optional[str]:
    """
    Pull the first [MM:SS] or [H:MM:SS] timestamp out of a chunk,
    so we can link users back to the exact video moment.

    Returns None if no timestamp is found (e.g. if include_timestamps=False).
    """
    match = re.search(r'\[(\d+:\d{2}(?::\d{2})?)\]', text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Step 5 — Main Entry Point
# ---------------------------------------------------------------------------

def ingest_youtube(url: str, include_timestamps: bool = True) -> List[Document]:
    """
    Full ingestion pipeline for a single YouTube video.

    Steps:
        1. Extract video ID from URL
        2. Fetch video metadata (title, channel, duration)
        3. Fetch transcript segments
        4. Reconstruct into readable paragraphed text
        5. Chunk into overlapping Document objects with metadata

    Args:
        url:                YouTube URL in any common format.
        include_timestamps: Whether to embed [MM:SS] markers in chunk text.
                            Recommended True — enables source attribution.

    Returns:
        List of chunk-level Document objects ready to be embedded.

    Raises:
        ValueError:         If the URL is invalid.
        TranscriptsDisabled: If the video has no captions.
        VideoUnavailable:   If the video is private or deleted.

    Example:
        >>> chunks = ingest_youtube("https://youtu.be/dQw4w9WgXcQ")
        >>> print(len(chunks), "chunks ready for embedding")
    """
    print(f"[YouTube Loader] Processing: {url}")

    # Step 1 — Extract video ID
    video_id = extract_video_id(url)
    print(f"[YouTube Loader] Video ID: {video_id}")

    # Step 2 — Fetch metadata
    metadata = fetch_video_metadata(video_id)
    print(f"[YouTube Loader] Title: {metadata['title']}")
    print(f"[YouTube Loader] Channel: {metadata['channel']}")

    # Step 3 — Fetch transcript
    try:
        segments = fetch_transcript(video_id)
        print(f"[YouTube Loader] Fetched {len(segments)} transcript segments")
    except TranscriptsDisabled:
        raise TranscriptsDisabled(
            f"Transcripts are disabled for '{metadata['title']}'. "
            f"Try a different video or upload a manual transcript."
        )
    except VideoUnavailable:
        raise VideoUnavailable(
            f"Video '{video_id}' is unavailable (private or deleted)."
        )

    # Step 4 — Reconstruct text
    transcript_text = reconstruct_transcript_text(segments, include_timestamps)
    word_count = len(transcript_text.split())
    print(f"[YouTube Loader] Transcript reconstructed ({word_count} words)")

    # Step 5 — Chunk
    chunks = chunk_transcript(transcript_text, metadata)
    print(f"[YouTube Loader] Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks


def ingest_multiple_youtube(
    urls: List[str],
    include_timestamps: bool = True,
) -> List[Document]:
    """
    Ingest multiple YouTube videos and return all chunks in a single flat list.

    Skips videos that fail (logs the error) so one bad URL doesn't
    break ingestion of the entire batch.

    Args:
        urls:               List of YouTube URLs.
        include_timestamps: Passed through to ingest_youtube().

    Returns:
        Combined list of chunks from all successfully ingested videos.
    """
    all_chunks: List[Document] = []

    for url in urls:
        try:
            chunks = ingest_youtube(url, include_timestamps)
            all_chunks.extend(chunks)
        except ValueError as e:
            print(f"[YouTube Loader] Invalid URL — skipping. {e}")
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"[YouTube Loader] No transcript available — skipping. {e}")
        except VideoUnavailable as e:
            print(f"[YouTube Loader] Video unavailable — skipping. {e}")
        except Exception as e:
            print(f"[YouTube Loader] Unexpected error for {url} — {e}")

    print(f"[YouTube Loader] Total chunks across all videos: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_youtube_metadata_summary(chunks: List[Document]) -> dict:
    """
    Return a summary of ingested YouTube content for the Streamlit sidebar.

    Args:
        chunks: List of chunk-level Documents from YouTube ingestion.

    Returns:
        Dict mapping video title to chunk count.

    Example:
        {
            "CS50 Lecture 1 - Scratch": 28,
            "MIT 6.006 Introduction to Algorithms": 45,
        }
    """
    summary = {}
    for chunk in chunks:
        title = chunk.metadata.get("source", "Unknown Video")
        summary[title] = summary.get(title, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# Quick test — run directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python youtube_loader.py <youtube_url>")
        print("Example: python youtube_loader.py https://youtu.be/dQw4w9WgXcQ")
        sys.exit(1)

    test_url = sys.argv[1]

    try:
        chunks = ingest_youtube(test_url)

        print("\n--- Sample chunk (first) ---")
        print("Content:", chunks[0].page_content[:400])
        print("Metadata:", chunks[0].metadata)

        print("\n--- Sample chunk (middle) ---")
        mid = len(chunks) // 2
        print("Content:", chunks[mid].page_content[:400])
        print("Metadata:", chunks[mid].metadata)

        print("\n--- Ingestion Summary ---")
        summary = get_youtube_metadata_summary(chunks)
        for title, count in summary.items():
            print(f"  '{title}': {count} chunks")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)