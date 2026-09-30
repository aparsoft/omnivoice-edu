"""Unit tests for the server internals that need no GPU and no running server:
voice-path sandboxing, audio post-processing and duration splitting.

Run: python -m tests.test_units
"""

import dataclasses
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from omnivoice_edu import audio, voices
from omnivoice_edu.config import SAMPLE_RATE
from omnivoice_edu.errors import InvalidRequest
from omnivoice_edu.synthesis import split_duration
from omnivoice_edu.text import Chunk


# ---------------------------------------------------------------------------
# Voices directory sandbox
# ---------------------------------------------------------------------------
def _with_voices_dir(fn):
    """Run ``fn(voices_dir, outside_file)`` with a temporary voices directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        vdir = root / "voices"
        vdir.mkdir()
        (vdir / "narrator.wav").write_bytes(b"RIFF")
        outside = root / "secret.txt"
        outside.write_text("do not read")
        original = voices.settings
        voices.settings = dataclasses.replace(original, voices_dir=vdir)
        try:
            fn(vdir, outside)
        finally:
            voices.settings = original


def _rejected(path: str) -> bool:
    try:
        voices.resolve_voice_path(path)
    except InvalidRequest:
        return True
    return False


def test_voice_bare_name_resolves():
    def check(vdir, _):
        assert voices.resolve_voice_path("narrator.wav") == str(vdir / "narrator.wav")

    _with_voices_dir(check)


def test_voice_absolute_path_inside_resolves():
    def check(vdir, _):
        path = str(vdir / "narrator.wav")
        assert voices.resolve_voice_path(path) == path

    _with_voices_dir(check)


def test_voice_traversal_rejected():
    def check(vdir, _):
        assert _rejected("../secret.txt"), "path traversal was allowed"

    _with_voices_dir(check)


def test_voice_absolute_path_outside_rejected():
    def check(vdir, outside):
        assert _rejected(str(outside)), "file outside voices dir was allowed"

    _with_voices_dir(check)


def test_voice_missing_file_rejected():
    def check(vdir, _):
        assert _rejected("nobody.wav"), "missing file was accepted"

    _with_voices_dir(check)


# ---------------------------------------------------------------------------
# Audio post-processing
# ---------------------------------------------------------------------------
def test_join_with_gaps_inserts_silence():
    a, b = np.ones(100, dtype=np.float32), np.ones(50, dtype=np.float32)
    out = audio.join_with_gaps([a, b], gap_ms=10)
    gap = int(SAMPLE_RATE * 10 / 1000)
    assert len(out) == 150 + gap
    assert not out[100 : 100 + gap].any()


def test_join_single_segment_unchanged():
    a = np.arange(5, dtype=np.float32)
    assert audio.join_with_gaps([a], gap_ms=80) is a


def test_trim_keeps_guard_band_before_onset():
    sig = np.zeros(SAMPLE_RATE, dtype=np.float32)
    sig[12000:] = 0.5  # onset at 0.5 s
    out = audio.trim_leading_silence(sig, lead_guard_ms=120)
    guard = int(SAMPLE_RATE * 0.12)
    assert len(out) == len(sig) - (12000 - guard)
    assert out[guard] == 0.5 and not out[:guard].any()


def test_trim_leaves_silence_untouched():
    sig = np.zeros(1000, dtype=np.float32)
    assert len(audio.trim_leading_silence(sig)) == 1000


def test_volume_is_peak_limited():
    out = audio.apply_volume(np.array([0.5, -0.5], dtype=np.float32), 3.0)
    assert np.isclose(np.max(np.abs(out)), 0.95)


def test_volume_one_is_identity():
    a = np.array([0.1, 0.2], dtype=np.float32)
    assert audio.apply_volume(a, 1.0) is a


def test_wav_roundtrip():
    a = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)
    data, sr = sf.read(io.BytesIO(audio.to_wav_bytes(a)))
    assert sr == SAMPLE_RATE and len(data) == len(a)


# ---------------------------------------------------------------------------
# Duration splitting across chunks
# ---------------------------------------------------------------------------
def _chunks(*lengths):
    return [
        Chunk(text="x" * n, index=i, is_first=i == 0, is_last=i == len(lengths) - 1)
        for i, n in enumerate(lengths)
    ]


def test_split_duration_none():
    assert split_duration(None, _chunks(10, 10), 80) is None


def test_split_duration_single_chunk():
    assert split_duration(12.0, _chunks(50), 80) == [12.0]


def test_split_duration_proportional_and_sums_to_target():
    parts = split_duration(10.0, _chunks(100, 300), chunk_gap_ms=100)
    assert np.isclose(parts[1] / parts[0], 3.0)
    assert np.isclose(sum(parts) + 0.1, 10.0)  # speech + one 100 ms gap


if __name__ == "__main__":
    from tests.runner import run_tests

    run_tests(globals())
