"""Server settings, read once from environment variables at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 24000  # OmniVoice's output sample rate (Hz)
MODEL_ID = "k2-fsa/OmniVoice"


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    device: str
    dtype: str
    asr_device: Optional[str]
    # The only directory `ref_audio_path` may read from. Requests name a
    # file inside it (e.g. "narrator.wav"); paths that resolve outside it
    # are refused, so API callers cannot make the server read other files.
    voices_dir: Path
    # Encoded voice prompts, persisted across restarts.
    cache_dir: Path
    # Optional shared secret. When set, the TTS endpoints require
    # "Authorization: Bearer <key>"; /health stays open for monitoring.
    api_key: Optional[str]
    # Max chunks synthesized together in one batched generate() call.
    max_batch: int

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            host=env("OMNIVOICE_HOST", "0.0.0.0"),
            port=int(env("OMNIVOICE_PORT", "8444")),
            device=env("OMNIVOICE_DEVICE", "cuda:0"),
            dtype=env("OMNIVOICE_DTYPE", "float16"),
            asr_device=env("OMNIVOICE_ASR_DEVICE") or None,
            voices_dir=Path(env("OMNIVOICE_VOICES_DIR", "voices")).resolve(),
            cache_dir=Path(env("OMNIVOICE_CACHE_DIR", ".cache/voice_prompts")).resolve(),
            api_key=env("OMNIVOICE_API_KEY") or None,
            max_batch=int(env("OMNIVOICE_MAX_BATCH", "8")),
        )


settings = Settings.from_env()
