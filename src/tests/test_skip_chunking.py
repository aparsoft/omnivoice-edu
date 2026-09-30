#!/usr/bin/env python3
"""
Test: skip_chunking flag on the OmniVoice TTS API
=================================================

Verifies that the ``skip_chunking`` flag correctly bypasses the
auto-chunking pipeline.  Two scenarios are tested:

  1. **Long text WITH chunking** (default) — the server should split
     the text into multiple chunks, synthesize each, and concatenate
     the audio with inter-chunk silence gaps.

  2. **Same long text WITHOUT chunking** (``skip_chunking=true``) —
     the server should pass the entire text to a single ``generate()``
     call, producing one continuous audio stream with no gaps.

Both outputs are saved to ``test_output/`` so you can listen and
compare the pacing and naturalness.

Usage:
    # Start the server first:
    python -m src.server.tts_server

    # Then run this test:
    python -m src.tests.test_skip_chunking
"""

import base64
import os
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_BASE = os.environ.get("TTS_API_URL", "http://localhost:8444")
OUTPUT_DIR = _PROJECT_ROOT / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# A deliberately long paragraph that exceeds the default chunk_threshold
# of 350 characters, forcing the chunker to split it into multiple
# segments when skip_chunking is False.
LONG_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, "
    "stories, and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The field of artificial intelligence "
    "research was founded at a workshop held on the campus of Dartmouth "
    "College during the summer of nineteen fifty-six. The attendees, including "
    "John McCarthy, Marvin Minsky, Allen Newell, and Herbert Simon, became "
    "the leaders of AI research for several decades. They and their students "
    "produced programs that were described as astonishing. Computers were "
    "learning to play chess, proving mathematical theorems, and speaking "
    "English fluently. By the mid nineteen seventies, it had become clear "
    "that the early optimism had been misplaced, and several governments "
    "withdrew funding, leading to the first AI winter."
)

# Reference voice: a file name inside the server's voices directory.
TEST_VOICE = os.environ.get("TTS_TEST_VOICE_EN", "af_heart.wav")


def _save_wav(audio_b64: str, filename: str) -> Path:
    """Decode base64 audio and save to test_output/."""
    wav_bytes = base64.b64decode(audio_b64)
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(wav_bytes)
    return out_path


def test_health():
    """Confirm the server is up before running the real tests."""
    print("\n=== Health Check ===")
    resp = requests.get(f"{API_BASE}/health", timeout=10)
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response: {data}")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    assert data["status"] == "ok", f"Server not ready: {data}"
    print("  ✅ Health check passed")


def test_long_text_with_chunking():
    """Synthesize LONG_TEXT with default chunking enabled.

    Contract:
      - The server should auto-split the text (it exceeds 350 chars).
      - The response should succeed and return valid audio.
      - The output duration should be reasonable for the text length.
    """
    print("\n=== Long Text WITH Chunking (default) ===")
    print(f"  Text length: {len(LONG_TEXT)} chars")

    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": LONG_TEXT,
            "num_step": 48,
            "speed": 1.0,
            "volume": 2.0,
            "ref_audio_path": TEST_VOICE,
            # skip_chunking defaults to False
            "chunk_threshold": 350,
            "chunk_gap_ms": 80,
        },
        timeout=180,
    )

    assert resp.status_code == 200, f"Request failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(
        f"  Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    out_path = _save_wav(data["audio_base64"], "skip_chunking_WITH_chunking.wav")
    print(
        f"  ✅ Saved to {out_path} ({len(base64.b64decode(data['audio_base64']))} bytes)"
    )

    # Sanity: audio should be at least a few seconds for this paragraph
    assert (
        data["duration_seconds"] > 3.0
    ), f"Audio too short ({data['duration_seconds']}s) — expected at least 3s for this text"


def test_long_text_skip_chunking():
    """Synthesize LONG_TEXT with skip_chunking=True.

    Contract:
      - The server should bypass chunking entirely and pass the full
        text to a single generate() call.
      - The response should succeed and return valid audio.
      - The output should be one continuous audio stream (no inter-chunk
        silence gaps), so it may be slightly shorter than the chunked
        version due to the absence of gap padding.
    """
    print("\n=== Long Text WITHOUT Chunking (skip_chunking=True) ===")
    print(f"  Text length: {len(LONG_TEXT)} chars")

    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": LONG_TEXT,
            "num_step": 48,
            "speed": 1.0,
            "volume": 2.0,
            "ref_audio_path": TEST_VOICE,
            "skip_chunking": True,
        },
        timeout=180,
    )

    assert resp.status_code == 200, f"Request failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(
        f"  Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    out_path = _save_wav(data["audio_base64"], "skip_chunking_NO_chunking.wav")
    print(
        f"  ✅ Saved to {out_path} ({len(base64.b64decode(data['audio_base64']))} bytes)"
    )

    # Sanity: audio should still be at least a few seconds
    assert (
        data["duration_seconds"] > 3.0
    ), f"Audio too short ({data['duration_seconds']}s) — expected at least 3s for this text"


def test_short_text_skip_chunking():
    """Synthesize a short text with skip_chunking=True.

    Contract:
      - For short texts that would not be chunked anyway, skip_chunking=True
        should produce the same result as the default (no functional
        difference, just avoids the chunker overhead).
      - The response should succeed and return valid audio.
    """
    print("\n=== Short Text WITH skip_chunking=True ===")

    short_text = "Hello, this is a short test of skip chunking."

    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": short_text,
            "num_step": 48,
            "speed": 1.0,
            "volume": 2.0,
            "ref_audio_path": TEST_VOICE,
            "skip_chunking": True,
        },
        timeout=120,
    )

    assert resp.status_code == 200, f"Request failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(
        f"  Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    out_path = _save_wav(data["audio_base64"], "skip_chunking_short_text.wav")
    print(
        f"  ✅ Saved to {out_path} ({len(base64.b64decode(data['audio_base64']))} bytes)"
    )

    # Short text should produce a short clip
    assert (
        data["duration_seconds"] > 1.0
    ), f"Audio too short ({data['duration_seconds']}s) — expected at least 1s"


def main():
    """Run all skip_chunking tests."""
    print("OmniVoice TTS API – skip_chunking Test Suite")
    print(f"API URL: {API_BASE}")
    print(f"Output dir: {OUTPUT_DIR}")

    test_health()
    test_long_text_with_chunking()
    test_long_text_skip_chunking()
    test_short_text_skip_chunking()

    print("\n🎉 All skip_chunking tests completed!")
    print(
        f"🎧 Listen to the outputs in {OUTPUT_DIR} to compare chunked vs unchunked audio."
    )


if __name__ == "__main__":
    main()
