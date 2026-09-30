"""Audio post-processing: joining chunks, trimming silence, volume, WAV."""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def join_with_gaps(
    segments: list[np.ndarray], gap_ms: int, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Concatenate audio segments with ``gap_ms`` of silence between them.

    Chunks are NOT trimmed individually: OmniVoice's first phonemes have a
    soft onset, and trimming each chunk at a fixed threshold would clip the
    first word. The silence gap alone gives natural chunk transitions.
    """
    if len(segments) == 1:
        return segments[0]
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    parts = [segments[0]]
    for seg in segments[1:]:
        parts.append(gap)
        parts.append(seg)
    return np.concatenate(parts)


def trim_leading_silence(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -60.0,
    lead_guard_ms: int = 120,
) -> np.ndarray:
    """Trim only the true dead air at the start of an utterance.

    Uses a very low amplitude threshold (default -60 dBFS, near the digital
    noise floor and quieter than OmniVoice's internal -50 dBFS trimmer) and
    keeps ``lead_guard_ms`` of audio before the first sample that exceeds it.
    Because the model's soft first-phoneme onset rises above -60 dBFS well
    before it becomes audible, the guard band guarantees the onset is never
    clipped — this only removes excess leading silence, never speech.

    Args:
        audio: 1-D numpy array (assumed roughly in [-1, 1]).
        sample_rate: Sample rate of ``audio``.
        threshold_db: Silence threshold in dBFS relative to full scale.
        lead_guard_ms: Milliseconds of audio to keep before the onset.

    Returns:
        The audio with excess leading silence removed.
    """
    if audio.size == 0:
        return audio

    threshold = 10.0 ** (threshold_db / 20.0)  # dBFS -> linear amplitude
    above = np.flatnonzero(np.abs(audio) > threshold)
    if above.size == 0:
        return audio  # entirely below threshold — leave untouched

    guard_samples = int(sample_rate * lead_guard_ms / 1000)
    start = max(0, int(above[0]) - guard_samples)
    return audio[start:]


def apply_volume(audio: np.ndarray, volume: float, peak_limit: float = 0.95) -> np.ndarray:
    """Scale by ``volume``, then peak-limit so the result cannot clip.

    OmniVoice output tends to be quiet, so the server amplifies by default.
    """
    if volume == 1.0:
        return audio
    audio = audio * volume
    peak = np.max(np.abs(audio))
    if peak > peak_limit:
        logger.info(
            "Normalizing audio: peak=%.2f exceeds %.2f, scaling down", peak, peak_limit
        )
        audio = audio * (peak_limit / peak)
    return audio


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode audio as an in-memory WAV file."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()
