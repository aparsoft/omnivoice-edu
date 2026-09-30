#!/usr/bin/env python3
"""
OmniVoice TTS FastAPI Server
=============================
A FastAPI service wrapping the OmniVoice TTS model, exposing it as an
HTTP API (port 8444 by default) so other services can consume TTS.

Three generation modes are supported:
  1. Voice Cloning  – provide reference audio (+ optional ref_text)
  2. Voice Design   – provide an instruct string (e.g. "female, british accent")
  3. Auto Voice     – no reference needed; the model picks a voice

Text preprocessing is always-on — math/digit verbalization, markdown
flattening, LaTeX cleanup, currency expansion, etc. are applied to
every request before synthesis.

Requires omnivoice >= 0.2.1 (VoiceClonePrompt.save/load, asr_device).

Usage:
    python -m src.server.tts_server
    # Server starts at http://0.0.0.0:8444

Endpoints:
    GET  /health           – health check
    POST /tts              – generate TTS audio (returns WAV file)
    POST /tts/json         – generate TTS audio (returns base64 JSON)

Security: server-side reference audio is only read from OMNIVOICE_VOICES_DIR,
and setting OMNIVOICE_API_KEY requires a Bearer token on the TTS endpoints.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import logging
import os
import sys
import tempfile
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so `src.utils` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Text preprocessor – cleans math, digits, markdown, etc. for TTS
from src.utils.text_preprocessor import preprocess_text_for_tts

# Smart text chunker – splits long text into natural-sounding chunks
# so the model doesn't speed up on longer inputs
from src.utils.text_chunker import Chunk, chunk_text_for_tts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("omnivoice-tts")

# ---------------------------------------------------------------------------
# Global model holder
# ---------------------------------------------------------------------------
MODEL = None
SAMPLE_RATE = 24000

# One model instance on one GPU: serialize generate() calls so concurrent
# requests don't interleave on the GPU (or race the lazy Whisper load).
# Inference runs in a worker thread, so /health stays responsive meanwhile.
_GPU_LOCK = threading.Lock()

# Max chunks synthesized together in one batched generate() call.
MAX_BATCH = int(os.environ.get("OMNIVOICE_MAX_BATCH", 8))

# Encoded reference voices, keyed by hash(audio bytes + ref_text). Kept in
# memory (LRU) and on disk, so the reference audio is tokenized — and, when
# ref_text is omitted, Whisper-transcribed — once per voice, not per request.
_PROMPT_CACHE: "OrderedDict[str, object]" = OrderedDict()
_PROMPT_CACHE_SIZE = 64
PROMPT_CACHE_DIR = _PROJECT_ROOT / ".cache" / "voice_prompts"
PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# The only directory `ref_audio_path` may read from. Requests name a file
# inside it (e.g. "narrator.wav"); paths that resolve outside it are refused,
# so API callers cannot make the server read arbitrary files.
VOICES_DIR = Path(
    os.environ.get("OMNIVOICE_VOICES_DIR", _PROJECT_ROOT / "local_folder" / "voice_prompts")
).resolve()

# Optional shared secret. When set, /tts and /tts/json require
# "Authorization: Bearer <key>"; /health stays open for monitoring.
API_KEY = os.environ.get("OMNIVOICE_API_KEY") or None

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def version_info() -> dict:
    """Versions of the model package and runtime stack (for logs and /health)."""
    from importlib.metadata import PackageNotFoundError, version

    def _pkg(name: str) -> Optional[str]:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "omnivoice": _pkg("omnivoice"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _pkg("transformers"),
        "fastapi": _pkg("fastapi"),
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def load_model() -> None:
    """Load the OmniVoice model into GPU memory (once, at startup)."""
    global MODEL
    if MODEL is not None:
        return

    v = version_info()
    logger.info(
        "Versions: omnivoice %s | torch %s (CUDA %s) | transformers %s | "
        "fastapi %s | python %s",
        v["omnivoice"],
        v["torch"],
        v["cuda"],
        v["transformers"],
        v["fastapi"],
        v["python"],
    )
    logger.info("GPU: %s", v["gpu"] or "none (CPU only)")
    logger.info("Loading OmniVoice model … (this may take a while on first run)")
    from omnivoice import OmniVoice

    device = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
    dtype_str = os.environ.get("OMNIVOICE_DTYPE", "float16")
    if dtype_str not in _DTYPES:
        raise ValueError(
            f"OMNIVOICE_DTYPE must be one of {sorted(_DTYPES)}, got {dtype_str!r}"
        )

    kwargs: dict = {}
    # Whisper (for auto-transcribing ref audio) defaults to the model's device.
    if os.environ.get("OMNIVOICE_ASR_DEVICE"):
        kwargs["asr_device"] = os.environ["OMNIVOICE_ASR_DEVICE"]

    MODEL = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=_DTYPES[dtype_str],
        **kwargs,
    )
    logger.info("OmniVoice model loaded successfully on %s (%s)", device, dtype_str)


# ---------------------------------------------------------------------------
# Lifespan – load model at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Cleanup: free GPU memory
    global MODEL
    if MODEL is not None:
        del MODEL
        MODEL = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed.")


app = FastAPI(
    title="OmniVoice TTS API",
    description="Zero-shot multilingual TTS service powered by OmniVoice (600+ languages)",
    version="1.0.0",
    lifespan=lifespan,
)

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Enforce the Bearer token when OMNIVOICE_API_KEY is set."""
    if API_KEY is None:
        return
    if creds is None or not hmac.compare_digest(creds.credentials, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _resolve_voice_path(ref_audio_path: str) -> str:
    """Resolve a requested reference audio file inside VOICES_DIR.

    Accepts a file name relative to VOICES_DIR ("narrator.wav"), a path
    relative to the project root, or an absolute path — but only if it
    resolves to a file inside VOICES_DIR.
    """
    requested = Path(ref_audio_path)
    candidates = (
        [requested]
        if requested.is_absolute()
        else [VOICES_DIR / requested, _PROJECT_ROOT / requested]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(VOICES_DIR) and resolved.is_file():
            return str(resolved)
    raise HTTPException(
        status_code=400,
        detail=f"ref_audio_path not found in the voices directory: {ref_audio_path}",
    )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    """JSON body for /tts/json endpoint."""

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
            "the model from speeding up. Set to a very large value "
            "(e.g. 99999) to disable chunking. Ignored when skip_chunking "
            "is True."
        ),
    )
    chunk_gap_ms: int = Field(
        80,
        ge=0,
        le=500,
        description=(
            "Silence gap in milliseconds between chunks. "
            "Adds a brief pause between chunk boundaries for natural pacing. "
            "Default 80ms. Set to 0 for no gap. Ignored when skip_chunking "
            "is True."
        ),
    )


class TTSJSONResponse(BaseModel):
    """Response for /tts/json endpoint."""

    audio_base64: str = Field(..., description="Base64-encoded WAV audio")
    sample_rate: int = Field(24000, description="Audio sample rate in Hz")
    duration_seconds: float = Field(..., description="Duration of generated audio")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _trim_leading_silence(
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


def _get_voice_clone_prompt(ref_audio_path: str, ref_text: Optional[str]):
    """Return a cached VoiceClonePrompt for this reference audio + text.

    OmniVoice's generate(ref_audio=...) re-encodes the reference audio —
    and re-runs Whisper when ref_text is omitted — on every call, i.e. once
    per chunk. Encoding once and reusing the prompt avoids all of that.
    Cache key is a hash of the audio bytes, so edited or re-uploaded files
    with the same content are handled correctly. Must hold _GPU_LOCK.
    """
    from omnivoice import VoiceClonePrompt

    h = hashlib.sha256(Path(ref_audio_path).read_bytes())
    h.update(b"\0" + (ref_text or "").encode("utf-8"))
    key = h.hexdigest()[:32]

    prompt = _PROMPT_CACHE.get(key)
    if prompt is not None:
        _PROMPT_CACHE.move_to_end(key)
        return prompt

    cache_file = PROMPT_CACHE_DIR / f"{key}.pt"
    prompt = None
    if cache_file.exists():
        try:
            prompt = VoiceClonePrompt.load(str(cache_file))
        except Exception:
            logger.warning("Ignoring unreadable voice cache %s", cache_file.name)
    if prompt is None:
        logger.info("Encoding reference voice %s", Path(ref_audio_path).name)
        prompt = MODEL.create_voice_clone_prompt(
            ref_audio=ref_audio_path, ref_text=ref_text
        )
        prompt.save(str(cache_file))

    _PROMPT_CACHE[key] = prompt
    if len(_PROMPT_CACHE) > _PROMPT_CACHE_SIZE:
        _PROMPT_CACHE.popitem(last=False)
    return prompt


def _split_duration(
    duration: Optional[float], chunks: list[Chunk], chunk_gap_ms: int
) -> Optional[list[float]]:
    """Share a total target duration across chunks, proportional to text length.

    Without this, a fixed ``duration`` would be applied to *every* chunk,
    multiplying the output length by the number of chunks.
    """
    if duration is None:
        return None
    if len(chunks) == 1:
        return [duration]
    speech = max(duration - (len(chunks) - 1) * chunk_gap_ms / 1000, 0.5)
    total_chars = sum(len(c.text) for c in chunks)
    return [max(speech * len(c.text) / total_chars, 0.3) for c in chunks]


def _synthesize_chunks(
    chunks: list[Chunk],
    voice_prompt,
    instruct: Optional[str],
    language: Optional[str],
    speed: float,
    durations: Optional[list[float]],
    gen_config,
) -> list[np.ndarray]:
    """Generate audio for every chunk, keeping one consistent voice.

    Chunks are synthesized in batched generate() calls (up to MAX_BATCH at
    a time), which is much faster than one call per chunk.

    In voice-design / auto-voice mode each generate() call samples a fresh
    voice, so independently generated chunks would drift to different
    speakers. Like OmniVoice's own long-form generation, we synthesize the
    first chunk alone and use it as the voice-clone reference for the rest.
    """
    texts = [c.text for c in chunks]
    audios: list[np.ndarray] = []

    def _run(start: int, end: int, prompt) -> None:
        out = MODEL.generate(
            text=texts[start:end],
            voice_clone_prompt=prompt,
            instruct=instruct,
            language=language,
            speed=speed,
            duration=durations[start:end] if durations else None,
            generation_config=gen_config,
        )
        if len(out) != end - start:
            raise HTTPException(status_code=500, detail="Model returned no audio")
        audios.extend(out)

    start = 0
    if voice_prompt is None and len(chunks) > 1:
        _run(0, 1, None)
        voice_prompt = MODEL.create_voice_clone_prompt(
            ref_audio=(torch.from_numpy(audios[0].astype(np.float32)), SAMPLE_RATE),
            ref_text=texts[0],
        )
        start = 1

    for batch_start in range(start, len(chunks), MAX_BATCH):
        _run(batch_start, min(batch_start + MAX_BATCH, len(chunks)), voice_prompt)

    return audios


def _generate_wav_bytes(
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    language: Optional[str] = None,
    seed: Optional[int] = None,
    num_step: int = 48,
    speed: float = 1.0,
    duration: Optional[float] = None,
    volume: float = 3.0,
    skip_chunking: bool = False,
    chunk_threshold: int = 350,
    chunk_gap_ms: int = 80,
) -> tuple[bytes, float]:
    """Run OmniVoice inference and return (wav_bytes, duration_seconds).

    Blocking (GPU-bound) — call via run_in_threadpool from async endpoints.

    When ``skip_chunking`` is True, the full text is passed to the model
    in a single generate() call — no splitting, no silence gaps, no
    concatenation.  This is ideal for short-to-medium texts or when
    the caller has already pre-chunked the input.

    Otherwise, for texts longer than ``chunk_threshold`` characters,
    the text is automatically split into natural-sounding chunks (at
    sentence / clause boundaries).  The chunks are synthesized in batches
    with a shared voice and the resulting audio arrays are concatenated
    with a small silence gap between them.  This prevents the model from
    speeding up on long inputs.
    """

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Always preprocess text for TTS (math/digit verbalization, markdown flattening, etc.)
    text = preprocess_text_for_tts(text).strip()
    if not text:
        raise HTTPException(
            status_code=400, detail="Text is empty after preprocessing"
        )
    logger.info("Preprocessed text: %.120s…", text)

    # ------------------------------------------------------------------
    # Auto-chunk long text to prevent model speedup
    # ------------------------------------------------------------------
    if skip_chunking:
        logger.info(
            "Chunking skipped (skip_chunking=True); synthesizing full text in one pass"
        )
        chunks = [Chunk(text=text, index=0, is_first=True, is_last=True)]
    else:
        chunks = chunk_text_for_tts(
            text,
            max_chunk_chars=chunk_threshold,
        )

    if len(chunks) > 1:
        logger.info(
            "Auto-chunked text into %d segments (total %d chars)",
            len(chunks),
            len(text),
        )
    for chunk in chunks:
        logger.info(
            "  Chunk %d/%d (%d chars): %.80s…",
            chunk.index + 1,
            len(chunks),
            len(chunk.text),
            chunk.text,
        )

    # ------------------------------------------------------------------
    # Build the generation config with postprocess_output DISABLED.
    #
    # OmniVoice's default post-processing runs remove_silence() on the
    # generated audio, which calls detect_leading_silence() at a fixed
    # -50 dBFS threshold. The diffusion model's first phoneme has a soft,
    # gradual onset that often dips below that threshold, so it gets
    # trimmed away — clipping the first word of a sentence. This fires
    # per generate() call (i.e. per chunk), so it hits the start of
    # every utterance.
    #
    # Setting postprocess_output=False skips ONLY remove_silence(). The
    # beneficial fade-in/out, edge padding and RMS volume normalisation
    # in _post_process_audio still run (they are not gated by this flag),
    # so audio edges stay click-free while the natural onset is preserved.
    # ------------------------------------------------------------------
    from omnivoice import OmniVoiceGenerationConfig

    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        postprocess_output=False,
    )

    with _GPU_LOCK:
        if seed is not None:
            torch.manual_seed(seed)

        voice_prompt = (
            _get_voice_clone_prompt(ref_audio_path, ref_text)
            if ref_audio_path
            else None
        )
        # instruct alongside a voice prompt is supported upstream: when the
        # two agree it stabilises the attributes it describes.
        audio_segments = _synthesize_chunks(
            chunks,
            voice_prompt=voice_prompt,
            instruct=instruct,
            language=language,
            speed=speed,
            durations=_split_duration(duration, chunks, chunk_gap_ms),
            gen_config=gen_config,
        )

    # ------------------------------------------------------------------
    # Add inter-chunk silence gaps for natural pacing
    # ------------------------------------------------------------------
    # NOTE: We do NOT trim leading silence from chunks. OmniVoice's
    # diffusion model produces a gradual onset where the first phonemes
    # are quieter — trimming at a fixed threshold eats those phonemes
    # and clips the first word.  The silence gap
    # alone is sufficient for natural-sounding chunk transitions.
    # ------------------------------------------------------------------
    silence_samples = int(SAMPLE_RATE * chunk_gap_ms / 1000)  # e.g. 1920 samples for 80ms
    silence_gap = np.zeros(silence_samples, dtype=np.float32)

    # Concatenate with silence gaps between chunks
    if len(audio_segments) > 1:
        parts = [audio_segments[0]]
        for seg in audio_segments[1:]:
            parts.append(silence_gap)
            parts.append(seg)
        audio_np = np.concatenate(parts)
    else:
        audio_np = audio_segments[0]

    # ------------------------------------------------------------------
    # Tidy the leading silence (safely).
    #
    # With postprocess_output=False the model preserves the full, soft
    # onset of the first word (this is what prevents the clipping),
    # but it also leaves a longer, variable run of leading silence. We
    # trim that excess down to a small, consistent pad WITHOUT risking
    # the onset by using a very low threshold (-60 dBFS — essentially the
    # digital noise floor, quieter than the model's own -50 dBFS) and a
    # generous guard band. The soft onset rises above -60 dBFS long
    # before it is audible, and we keep LEAD_GUARD_MS before that point,
    # so no phoneme is ever clipped — we only remove true dead air.
    # ------------------------------------------------------------------
    audio_np = _trim_leading_silence(
        audio_np, sample_rate=SAMPLE_RATE, threshold_db=-60.0, lead_guard_ms=120
    )

    dur_seconds = len(audio_np) / SAMPLE_RATE

    # Apply volume amplification.
    # OmniVoice output tends to be very quiet, so we amplify by default.
    # volume=1.0 leaves it as-is, volume=2.0 doubles amplitude, etc.
    # We use peak normalization to a target level so the audio is loud
    # without clipping distortion.
    if volume != 1.0:
        audio_np = audio_np * volume
        # Peak-normalize to 0.95 to prevent clipping while staying loud
        peak = np.max(np.abs(audio_np))
        if peak > 0.95:
            logger.info(
                "Normalizing audio: peak=%.2f exceeds 0.95, scaling to 0.95",
                peak,
            )
            audio_np = audio_np * (0.95 / peak)

    # Encode to WAV in memory
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV")
    wav_bytes = buf.getvalue()

    return wav_bytes, dur_seconds


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check – confirms the model is loaded and ready."""
    v = version_info()
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": "k2-fsa/OmniVoice",
        "omnivoice_version": v["omnivoice"],
        "gpu": torch.cuda.is_available(),
        "device_name": v["gpu"],
        "cached_voices": len(_PROMPT_CACHE),
        "versions": v,
    }


@app.post("/tts", dependencies=[Depends(require_api_key)])
async def tts_file(
    text: str = Form(..., description="Text to synthesize"),
    ref_audio: Optional[UploadFile] = File(
        None, description="Reference audio for voice cloning"
    ),
    ref_text: Optional[str] = Form(
        None, description="Transcription of reference audio"
    ),
    instruct: Optional[str] = Form(None, description="Voice design attributes"),
    language: Optional[str] = Form(
        None, description="Optional language hint, e.g. 'en', 'hi', 'Hindi'"
    ),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
    num_step: int = Form(48, description="Diffusion steps (64=max quality, 32=speed)"),
    speed: float = Form(
        1.0,
        description="Speed factor (>1 faster, <1 slower). 1.0 recommended; <1.0 can clip the first word.",
    ),
    duration: Optional[float] = Form(
        None, description="Fixed output duration (seconds)"
    ),
    volume: float = Form(
        3.0, description="Volume multiplier (1.0=original, 3.0=recommended default)"
    ),
    skip_chunking: bool = Form(
        False, description="Skip auto-chunking; synthesize full text in one pass"
    ),
    chunk_threshold: int = Form(
        350, description="Max chars per chunk (auto-split longer texts)"
    ),
    chunk_gap_ms: int = Form(
        80,
        description="Silence gap in ms between chunks",
    ),
):
    """Generate TTS audio and return it as a downloadable WAV file.

    Modes:
      - Voice Cloning: upload ref_audio (+ optional ref_text)
      - Voice Design:  provide instruct string
      - Auto Voice:    omit both ref_audio and instruct
    """
    ref_audio_path: Optional[str] = None

    try:
        # Save uploaded ref_audio to a temp file if provided
        if ref_audio is not None:
            suffix = Path(ref_audio.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await ref_audio.read())
            ref_audio_path = f.name

        wav_bytes, _ = await run_in_threadpool(
            _generate_wav_bytes,
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            instruct=instruct,
            language=language,
            seed=seed,
            num_step=num_step,
            speed=speed,
            duration=duration,
            volume=volume,
            skip_chunking=skip_chunking,
            chunk_threshold=chunk_threshold,
            chunk_gap_ms=chunk_gap_ms,
        )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="tts_output.wav"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up temp ref audio
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)


@app.post(
    "/tts/json",
    response_model=TTSJSONResponse,
    dependencies=[Depends(require_api_key)],
)
async def tts_json(body: TTSRequest):
    """Generate TTS audio and return it as base64-encoded JSON.

    This is the most convenient endpoint for programmatic consumption.
    """
    try:
        # Resolve server-side ref_audio_path (must live in VOICES_DIR)
        resolved_ref_audio = (
            _resolve_voice_path(body.ref_audio_path) if body.ref_audio_path else None
        )

        wav_bytes, dur = await run_in_threadpool(
            _generate_wav_bytes,
            text=body.text,
            ref_audio_path=resolved_ref_audio,
            ref_text=body.ref_text,
            instruct=body.instruct,
            language=body.language,
            seed=body.seed,
            num_step=body.num_step,
            speed=body.speed,
            duration=body.duration,
            volume=body.volume,
            skip_chunking=body.skip_chunking,
            chunk_threshold=body.chunk_threshold,
            chunk_gap_ms=body.chunk_gap_ms,
        )

        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return TTSJSONResponse(
            audio_base64=audio_b64,
            sample_rate=SAMPLE_RATE,
            duration_seconds=round(dur, 3),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("OMNIVOICE_PORT", 8444))
    host = os.environ.get("OMNIVOICE_HOST", "0.0.0.0")

    logger.info("Starting OmniVoice TTS API on %s:%s", host, port)
    uvicorn.run(
        "src.server.tts_server:app",
        host=host,
        port=port,
        log_level="info",
    )
