#!/usr/bin/env python3
"""
Test the OmniVoice TTS API with text preprocessing + voice cloning.

Tests:
  1. Health check
  2. Simple English voice cloning
  3. Math-heavy English text with preprocessing
  4. Speed variations — 1.0x, 1.25x, 1.5x
  5. Hindi voice cloning
  6. Volume level variations — 1.0x, 2.0x, 3.0x

Reference voices are file names inside the server's voices directory
(OMNIVOICE_VOICES_DIR). Override them with TTS_TEST_VOICE_EN / TTS_TEST_VOICE_HI.

Text preprocessing is always-on (the default pipeline) — no opt-out.

Saves output WAV files to <project_root>/output/ so you can hear the results.
"""

import base64
import os
from pathlib import Path

import requests

# Project root is one level up from this file (tests/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_BASE = os.environ.get("TTS_API_URL", "http://localhost:8444")
OUTPUT_DIR = _PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Reference voices for cloning (file names inside the server's voices dir)
VOICE_PROMPT_EN = os.environ.get("TTS_TEST_VOICE_EN", "af_heart.wav")
VOICE_PROMPT_HI = os.environ.get("TTS_TEST_VOICE_HI", "aa_hi_female.wav")


def test_health():
    print("\n=== Health Check ===")
    resp = requests.get(f"{API_BASE}/health", timeout=10)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(
        f"Model: {data['model']} | GPU: {data['gpu']} | Device: {data.get('device_name')}"
    )
    assert resp.status_code == 200
    assert data["status"] == "ok"
    print("✅ Health check passed")


def test_simple_voice_clone():
    """Simple English text with voice cloning — sanity check."""
    print("\n=== Voice Cloning + Simple Text (English) ===")

    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": "Hello! This is a test of voice cloning with a reference voice. The text preprocessor will clean up any math symbols automatically.",
            "ref_audio_path": VOICE_PROMPT_EN,
            "num_step": 16,
            "speed": 1.0,
        },
        timeout=300,
    )
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()
    print(
        f"Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    wav_bytes = base64.b64decode(data["audio_base64"])
    out_path = OUTPUT_DIR / "simple_voice_clone.wav"
    out_path.write_bytes(wav_bytes)
    print(f"✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def test_math_voice_clone():
    """Voice cloning + math text — preprocessing is always-on."""
    print("\n=== Voice Cloning + Math Text (English) ===")

    # Math-heavy text that the preprocessor will clean up
    math_text = (
        "The equation is 3x + 5 = 20. "
        "If x = 5, then 3 times 5 plus 5 equals 20. "
        "The area of a circle is π times r squared. "
        "For r = 7, the area is approximately 153.94 square units. "
        "The temperature today is 35°C. "
        "The stock price increased by 12.5% to $1,250. "
        "The speed of light is 299,792,458 meters per second."
    )

    print(f"Original text: {math_text[:100]}…")

    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": math_text,
            "ref_audio_path": VOICE_PROMPT_EN,
            "num_step": 16,
            "speed": 1.0,
        },
        timeout=300,
    )
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()
    print(
        f"Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    # Decode and save
    wav_bytes = base64.b64decode(data["audio_base64"])
    out_path = OUTPUT_DIR / "math_voice_clone.wav"
    out_path.write_bytes(wav_bytes)
    print(f"✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def test_speed_variations():
    """Same text at different speed factors — 1.0x, 1.25x, 1.5x."""
    print("\n=== Speed Variations (English) ===")

    text = (
        "The speed of light is approximately 299,792,458 meters per second. "
        "This sentence is being tested at different speech rates."
    )

    for speed in [1.0, 1.25, 1.5]:
        print(f"\n  --- speed={speed}x ---")
        resp = requests.post(
            f"{API_BASE}/tts/json",
            json={
                "text": text,
                "ref_audio_path": VOICE_PROMPT_EN,
                "num_step": 16,
                "speed": speed,
            },
            timeout=300,
        )
        print(f"  Status: {resp.status_code}")
        assert resp.status_code == 200, f"Failed at speed={speed}: {resp.text}"
        data = resp.json()
        print(
            f"  Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
        )

        wav_bytes = base64.b64decode(data["audio_base64"])
        out_path = OUTPUT_DIR / f"speed_{speed}x.wav"
        out_path.write_bytes(wav_bytes)
        print(f"  ✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def test_hindi_voice_clone():
    """Hindi voice cloning."""
    print("\n=== Voice Cloning + Hindi Text ===")

    hindi_text = (
        "नमस्ते! यह ओम्नीवॉइस टीटीएस सिस्टम का परीक्षण है। "
        "भारत एक विविधतापूर्ण देश है जहाँ कई भाषाएँ बोली जाती हैं। "
        "गणित एक महत्वपूर्ण विषय है। दो और तीन मिलाकर पाँच होते हैं।"
    )

    print(f"Hindi text: {hindi_text[:80]}…")

    resp = requests.post(
        f"{API_BASE}/tts/json",
        json={
            "text": hindi_text,
            "ref_audio_path": VOICE_PROMPT_HI,
            "num_step": 16,
            "speed": 1.0,
        },
        timeout=300,
    )
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()
    print(
        f"Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
    )

    wav_bytes = base64.b64decode(data["audio_base64"])
    out_path = OUTPUT_DIR / "hindi_voice_clone.wav"
    out_path.write_bytes(wav_bytes)
    print(f"✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def test_volume_levels():
    """Test different volume levels — 1.0 (original), 2.0, 3.0 (default)."""
    print("\n=== Volume Level Variations (English) ===")

    text = (
        "This is a volume test. "
        "Higher volume multipliers should sound noticeably louder than one."
    )

    for volume in [1.0, 2.0, 3.0]:
        print(f"\n  --- volume={volume}x ---")
        resp = requests.post(
            f"{API_BASE}/tts/json",
            json={
                "text": text,
                "ref_audio_path": VOICE_PROMPT_EN,
                "num_step": 16,
                "speed": 1.0,
                "volume": volume,
            },
            timeout=300,
        )
        print(f"  Status: {resp.status_code}")
        assert resp.status_code == 200, f"Failed at volume={volume}: {resp.text}"
        data = resp.json()
        print(
            f"  Duration: {data['duration_seconds']}s | Sample rate: {data['sample_rate']}Hz"
        )

        wav_bytes = base64.b64decode(data["audio_base64"])
        out_path = OUTPUT_DIR / f"volume_{volume}x.wav"
        out_path.write_bytes(wav_bytes)
        print(f"  ✅ Saved to {out_path} ({len(wav_bytes)} bytes)")


def main():
    print("OmniVoice TTS API – Full Test Suite")
    print(f"API URL: {API_BASE}")
    print(f"Output dir: {OUTPUT_DIR}")

    print(f"English voice: {VOICE_PROMPT_EN}")
    print(f"Hindi voice:    {VOICE_PROMPT_HI}")

    test_health()
    test_simple_voice_clone()
    test_math_voice_clone()
    test_speed_variations()
    test_hindi_voice_clone()
    test_volume_levels()

    print(f"\n🎉 All tests completed! WAV files saved to {OUTPUT_DIR}/")
    print("Text preprocessing is always-on — the default pipeline.")


if __name__ == "__main__":
    main()
