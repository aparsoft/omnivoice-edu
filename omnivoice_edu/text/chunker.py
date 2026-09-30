"""Smart Text Chunker for TTS Synthesis
======================================

Splits long text into natural-sounding chunks before TTS synthesis.

OmniVoice tends to speed up on long texts — the model compresses timing
as the sequence grows, making the speech sound rushed.  This chunker
breaks text at natural sentence boundaries so each segment is short
enough for the model to maintain a natural pace.

Algorithm:
  1. If text is ≤ MAX_CHUNK_CHARS, return it as a single chunk.
  2. Otherwise, split on sentence-ending punctuation (. ! ? … ।).
  3. Greedily merge short fragments into chunks that stay within
     [MIN_CHUNK_CHARS, MAX_CHUNK_CHARS].
  4. If a fragment exceeds MAX_CHUNK_CHARS, split it further on
     clause boundaries (, ; : —) or, as a last resort, on spaces.

Chunks never overlap: overlapping text makes the model re-speak the
shared words, which is audible as repetition.

Usage:
    from omnivoice_edu.text import chunk_text_for_tts

    chunks = chunk_text_for_tts(long_text)
    audios = model.generate(text=[c.text for c in chunks], ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_CHUNK_CHARS = 350
DEFAULT_MIN_CHUNK_CHARS = 80

# Sentence-ending punctuation — primary split points (incl. Devanagari ।, ॥)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…।॥])\s+")

# Clause-level punctuation — secondary split points for long sentences
_CLAUSE_END_RE = re.compile(r"(?<=[,;:—–])\s+")


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    index: int
    is_first: bool
    is_last: bool


def chunk_text_for_tts(
    text: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    overlap_chars: int = 0,  # kept for API compat, always 0
) -> list[Chunk]:
    """Split text into natural-sounding chunks for TTS synthesis.

    Args:
        text: Preprocessed text ready for TTS.
        max_chunk_chars: Maximum characters per chunk. Texts longer
            than this will be split. Default 350.
        min_chunk_chars: Minimum characters per chunk. Short fragments
            will be merged with neighbours to avoid tiny chunks.
            Default 80.
        overlap_chars: **Deprecated — kept for API compatibility only.
            Overlap causes the model to re-speak words, creating
            audible repetition. Always set to 0.**

    Returns:
        List of Chunk objects, each with ``text``, ``index``,
        ``is_first``, and ``is_last`` fields.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Short text — no chunking needed
    if len(text) <= max_chunk_chars:
        return [Chunk(text=text, index=0, is_first=True, is_last=True)]

    # Step 1: Split on sentence boundaries
    raw_fragments = _split_sentences(text)

    # Step 2: Merge short fragments into chunks within [min, max]
    chunks_text = _merge_fragments(
        raw_fragments,
        max_chunk_chars=max_chunk_chars,
        min_chunk_chars=min_chunk_chars,
    )

    # Step 3: Break any chunk that still exceeds max_chunk_chars
    chunks_text = _split_long_chunks(chunks_text, max_chunk_chars=max_chunk_chars)

    # Build Chunk objects — NO overlap (overlap causes repeated narration)
    result = []
    for i, chunk_text in enumerate(chunks_text):
        result.append(
            Chunk(
                text=chunk_text.strip(),
                index=i,
                is_first=(i == 0),
                is_last=(i == len(chunks_text) - 1),
            )
        )

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation, preserving the punctuation."""
    # Split after . ! ? … । ॥ followed by whitespace
    parts = _SENTENCE_END_RE.split(text)
    # Filter empty strings
    return [p.strip() for p in parts if p.strip()]


def _merge_fragments(
    fragments: list[str],
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> list[str]:
    """Greedily merge short fragments into chunks within [min, max] size."""
    if not fragments:
        return []

    chunks: list[str] = []
    current = fragments[0]

    for frag in fragments[1:]:
        candidate = current + " " + frag

        # If merging would exceed max, finalize current chunk
        if len(candidate) > max_chunk_chars:
            # Only finalize if current chunk meets minimum size
            if len(current) >= min_chunk_chars or chunks:
                chunks.append(current)
                current = frag
            else:
                # Current is too short — merge anyway to avoid tiny chunks
                current = candidate
        else:
            current = candidate

    # Don't forget the last chunk
    if current.strip():
        chunks.append(current)

    return chunks


def _split_long_chunks(chunks: list[str], max_chunk_chars: int) -> list[str]:
    """Break any chunk that still exceeds max_chunk_chars using clause or space splits."""
    result = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_chars:
            result.append(chunk)
            continue

        # Try splitting on clause boundaries first
        sub_parts = _CLAUSE_END_RE.split(chunk)
        sub_parts = [p.strip() for p in sub_parts if p.strip()]

        # If clause splitting didn't help much (e.g. one giant sub-part),
        # fall back to space-based splitting
        if len(sub_parts) <= 1 or max(len(p) for p in sub_parts) > max_chunk_chars:
            sub_parts = _split_on_spaces(chunk, max_chunk_chars)
        else:
            # Merge sub_parts respecting max_chunk_chars
            sub_parts = _merge_fragments(
                sub_parts,
                max_chunk_chars=max_chunk_chars,
                min_chunk_chars=20,  # lower min for sub-splits
            )

        result.extend(sub_parts)

    return result


def _split_on_spaces(text: str, max_chunk_chars: int) -> list[str]:
    """Split text on word boundaries, keeping each chunk ≤ max_chunk_chars."""
    words = text.split()
    chunks = []
    current = ""

    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > max_chunk_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate

    if current.strip():
        chunks.append(current)

    return chunks
