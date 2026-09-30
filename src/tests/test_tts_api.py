#!/usr/bin/env python3
"""
Quick test client for the OmniVoice TTS API.

Tests all three modes:
  1. Auto Voice
  2. Voice Design
  3. Voice Cloning via file upload (if a reference voice file is found)

The upload test reads TTS_TEST_VOICE_FILE, defaulting to af_heart.wav in the
voices directory (OMNIVOICE_VOICES_DIR).

Saves generated WAV files to <project_root>/test_output/
"""

import base64
import os
from pathlib import Path

import requests

# Project root is two levels up from this file (src/tests/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

API_BASE = os.environ.get("TTS_API_URL", "http://localhost:8444")
OUTPUT_DIR = _PROJECT_ROOT / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

VOICES_DIR = Path(
    os.environ.get("OMNIVOICE_VOICES_DIR", _PROJECT_ROOT / "local_folder" / "voice_prompts")
)
TEST_VOICE_FILE = Path(
    os.environ.get("TTS_TEST_VOICE_FILE", VOICES_DIR / "af_heart.wav")
)


def test_health():
    print("\n=== Health Check ===")
    resp = requests.get(f"{API_BASE}/health", timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("✅ Health check passed")


def test_auto_voice():
    print("\n=== Auto Voice Mode ===")
    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": "Hello! This is a test of the OmniVoice text to speech system running in auto voice mode.",
            "num_step": 16,
            "speed": 1.0,
        },
        timeout=120,
    )
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    print(
        f"Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    # Decode and save
    wav_bytes = base64.b64decode(data["audio_base64"])
    out_path = OUTPUT_DIR / "auto_voice.wav"
    out_path.write_bytes(wav_bytes)
    print(f"✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def test_voice_design():
    print("\n=== Voice Design Mode ===")
    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": "Good morning everyone! Today we will explore the capabilities of voice design technology.",
            "instruct": "female, low pitch, british accent",
            "num_step": 16,
        },
        timeout=120,
    )
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    print(
        f"Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    wav_bytes = base64.b64decode(data["audio_base64"])
    out_path = OUTPUT_DIR / "voice_design.wav"
    out_path.write_bytes(wav_bytes)
    print(f"✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def test_voice_cloning_file():
    """Test voice cloning via the /tts file endpoint."""
    print("\n=== Voice Cloning Mode (file endpoint) ===")

    ref_path = TEST_VOICE_FILE
    if not ref_path.exists():
        print(f"⚠️  No reference voice at {ref_path} – skipping voice cloning test")
        return

    with open(ref_path, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/tts",
            data={
                "text": "This is a voice cloning test using the OmniVoice API.",
                "num_step": "16",
            },
            files={"ref_audio": ("ref.wav", f, "audio/wav")},
            timeout=120,
        )
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200

    out_path = OUTPUT_DIR / "voice_cloning.wav"
    out_path.write_bytes(resp.content)
    print(f"✅ Saved to {out_path} ({len(resp.content)} bytes)")


def main():
    print("OmniVoice TTS API – Test Suite")
    print(f"API URL: {API_BASE}")
    print(f"Output dir: {OUTPUT_DIR}")

    test_health()
    test_auto_voice()
    test_voice_design()
    test_voice_cloning_file()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()
