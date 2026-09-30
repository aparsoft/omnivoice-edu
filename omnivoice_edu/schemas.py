"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .config import SAMPLE_RATE


class TTSRequest(BaseModel):
    """JSON body for the /tts/json endpoint."""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to synthesize"
    )
    ref_text: Optional[str] = Field(
        None, description="Transcription of reference audio (voice cloning mode)"
    )
    ref_audio_path: Optional[str] = Field(
        None,
        description=(
            "Reference audio for voice cloning: a file name inside the "
            "server's voices directory, e.g. 'narrator.wav'"
        ),
    )
    instruct: Optional[str] = Field(
        None,
        description="Voice design attributes, e.g. 'female, low pitch, british accent'",
    )
    language: Optional[str] = Field(
        None,
        description=(
            "Optional language hint, as an ID ('en', 'hi') or a name "
            "('English', 'Hindi'). Omit to let the model infer it."
        ),
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Random seed for reproducible output. Most useful in auto-voice "
            "mode, where it pins which random voice is picked."
        ),
    )
    num_step: int = Field(
        48,
        ge=1,
        le=128,
        description="Diffusion steps (64 for max quality, 32 for speed)",
    )
    speed: float = Field(
        1.0,
        gt=0.0,
        le=5.0,
        description=(
            "Speed factor (>1 faster, <1 slower). Speeds below 1.0 can "
            "clip the onset of the first word in a segment, so 1.0 (the "
            "default) is recommended."
        ),
    )
    duration: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Fixed output duration in seconds (overrides speed). When the "
            "text is chunked, it is split across chunks by text length."
        ),
    )
    volume: float = Field(
        3.0,
        gt=0.0,
        le=10.0,
        description=(
            "Volume multiplier. 1.0 = original model output, "
            "3.0 = tripled (recommended default for OmniVoice), "
            "values > 1.0 amplify, < 1.0 reduce volume"
        ),
    )
    skip_chunking: bool = Field(
        False,
        description=(
            "When True, bypass auto-chunking entirely and pass the "
            "full text to the model in a single generate() call. "
            "Useful for short-to-medium texts where you want the model's "
            "native phrasing, or when calling from a pipeline that has "
            "already pre-chunked the text. Overrides chunk_threshold "
            "and chunk_gap_ms."
        ),
    )
    chunk_threshold: int = Field(
        350,
        ge=50,
        le=99999,
        description=(
            "Max characters per chunk. Texts longer than this are "
            "auto-split at sentence/clause boundaries to prevent "
            "the model from speeding up. Ignored when skip_chunking "
            "is True."
        ),
    )
    chunk_gap_ms: int = Field(
        80,
        ge=0,
        le=500,
        description=(
            "Silence gap in milliseconds between chunks. Set to 0 for no "
            "gap. Ignored when skip_chunking is True."
        ),
    )


class TTSJSONResponse(BaseModel):
    """Response for the /tts/json endpoint."""

    audio_base64: str = Field(..., description="Base64-encoded WAV audio")
    sample_rate: int = Field(SAMPLE_RATE, description="Audio sample rate in Hz")
    duration_seconds: float = Field(..., description="Duration of generated audio")
