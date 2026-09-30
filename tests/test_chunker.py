"""Tests for the text chunker module.

Verifies that long text is split at natural boundaries, short text passes
through unchanged, and no overlap is added (overlap was removed because
it caused the model to re-speak words, creating audible repetition).
"""

from omnivoice_edu.text.chunker import (
    Chunk,
    chunk_text_for_tts,
    _merge_fragments,
    _split_long_chunks,
    _split_on_spaces,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    """Sentences split on . ! ? followed by whitespace."""
    text = "Hello world. How are you? I am fine!"
    parts = _split_sentences(text)
    assert len(parts) == 3
    assert parts[0] == "Hello world."
    assert parts[1] == "How are you?"
    assert parts[2] == "I am fine!"


def test_split_sentences_no_punctuation():
    """Text without sentence-ending punctuation stays as one fragment."""
    text = "Hello world this is one long sentence"
    parts = _split_sentences(text)
    assert len(parts) == 1
    assert parts[0] == text


def test_split_sentences_ellipsis():
    """Ellipsis (…) is treated as a sentence boundary."""
    text = "Wait… Let me think. Okay."
    parts = _split_sentences(text)
    assert len(parts) == 3


# ---------------------------------------------------------------------------
# _merge_fragments
# ---------------------------------------------------------------------------


def test_merge_fragments_short():
    """Short fragments are merged until they reach min_chunk_chars."""
    fragments = ["Hi.", "My name is.", "I like pie."]
    merged = _merge_fragments(fragments, max_chunk_chars=220, min_chunk_chars=80)
    # All three are short, so they should be merged into one
    assert len(merged) == 1
    assert "Hi." in merged[0]


def test_merge_fragments_long():
    """Fragments that exceed max_chunk_chars are split into separate chunks."""
    fragments = [
        "A" * 150 + ".",
        "B" * 150 + ".",
        "C" * 150 + ".",
    ]
    merged = _merge_fragments(fragments, max_chunk_chars=220, min_chunk_chars=80)
    # Each fragment is ~151 chars, so they can't all merge into one 220-char chunk
    assert len(merged) >= 2


# ---------------------------------------------------------------------------
# _split_on_spaces
# ---------------------------------------------------------------------------


def test_split_on_spaces_basic():
    """Words are split on space boundaries respecting max_chunk_chars."""
    text = "word1 word2 word3 word4 word5 word6 word7 word8"
    chunks = _split_on_spaces(text, max_chunk_chars=20)
    for chunk in chunks:
        assert len(chunk) <= 20


# ---------------------------------------------------------------------------
# chunk_text_for_tts (integration)
# ---------------------------------------------------------------------------


def test_chunk_short_text():
    """Text shorter than max_chunk_chars returns a single chunk."""
    text = "Hello, this is a short text."
    chunks = chunk_text_for_tts(text, max_chunk_chars=220)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].is_first is True
    assert chunks[0].is_last is True


def test_chunk_empty_text():
    """Empty text returns no chunks."""
    assert chunk_text_for_tts("") == []
    assert chunk_text_for_tts("   ") == []


def test_chunk_long_text():
    """Long text is split into multiple chunks at sentence boundaries."""
    # Build a text that's clearly longer than 220 chars
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "Peter Piper picked a peck of pickled peppers.",
        "A stitch in time saves nine.",
        "The early bird catches the worm.",
        "Actions speak louder than words.",
        "All that glitters is not gold.",
    ]
    text = " ".join(sentences)
    assert len(text) > 220

    chunks = chunk_text_for_tts(text, max_chunk_chars=220, overlap_chars=30)
    assert len(chunks) > 1

    # Verify chunk metadata
    assert chunks[0].is_first is True
    assert chunks[0].is_last is False
    assert chunks[-1].is_last is True
    assert chunks[-1].is_first is False

    # Verify each chunk is within reasonable bounds
    for chunk in chunks:
        assert len(chunk.text) > 0
        # Chunks may exceed max_chunk_chars slightly due to overlap,
        # but the core content should be within bounds
        assert len(chunk.text) < 400  # generous upper bound


def test_chunk_preserves_content():
    """All original words appear in the concatenated chunks (allowing overlap)."""
    text = (
        "The first sentence is here. The second sentence follows. "
        "The third sentence continues. The fourth sentence wraps up. "
        "The fifth sentence adds more. The sixth sentence concludes."
    )
    chunks = chunk_text_for_tts(text, max_chunk_chars=120)

    # Every word from the original should appear in at least one chunk
    original_words = set(text.lower().split())
    chunk_words = set()
    for chunk in chunks:
        chunk_words.update(chunk.text.lower().split())

    # All original words should be present (no words lost during chunking)
    assert original_words.issubset(
        chunk_words
    ), f"Missing words: {original_words - chunk_words}"


def test_chunk_indices_are_sequential():
    """Chunk indices are 0-based and sequential."""
    text = (
        "Sentence one is here. Sentence two follows. "
        "Sentence three continues. Sentence four wraps up. "
        "Sentence five adds more. Sentence six concludes."
    )
    chunks = chunk_text_for_tts(text, max_chunk_chars=100)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_chunk_no_repeated_words():
    """Chunks should NOT contain repeated words from overlap (overlap removed)."""
    text = (
        "First sentence here. Second sentence follows. "
        "Third sentence continues. Fourth sentence wraps up."
    )
    chunks = chunk_text_for_tts(text, max_chunk_chars=80)

    # Verify no chunk starts with words that belong to the end of the previous chunk
    # (this was the bug — overlap caused the model to re-speak words)
    for i in range(1, len(chunks)):
        prev_end_words = chunks[i - 1].text.split()[-3:]
        curr_start_words = chunks[i].text.split()[:3]
        # The current chunk should NOT start with the same words the previous chunk ended with
        overlap = set(prev_end_words) & set(curr_start_words)
        # Allow at most 1 common word (coincidental), but not 2+ (which would indicate overlap)
        assert len(overlap) < 2, (
            f"Chunks {i-1} and {i} share start/end words: {overlap} — "
            f"possible overlap causing repeated narration"
        )


def test_chunk_very_long_single_sentence():
    """A single very long sentence (no punctuation) gets split on spaces."""
    # 300 chars with no sentence-ending punctuation
    text = "word " * 60  # ~300 chars
    chunks = chunk_text_for_tts(text.strip(), max_chunk_chars=220)
    assert len(chunks) >= 2
    # Each chunk should be non-empty
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0


if __name__ == "__main__":
    from tests.runner import run_tests

    run_tests(globals())
