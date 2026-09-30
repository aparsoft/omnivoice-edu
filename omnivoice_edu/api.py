"""FastAPI application: routes, authentication and error handling.

Endpoints:
    GET  /health     – readiness and runtime versions (never needs a key)
    POST /tts        – multipart form in, WAV file out
    POST /tts/json   – JSON in, base64 WAV out

Three generation modes are supported:
    Voice cloning  – reference audio (+ optional ref_text)
    Voice design   – an instruct string, e.g. "female, british accent"
    Auto voice     – neither; the model picks a voice
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import __version__, model
from .config import MODEL_ID, SAMPLE_RATE, settings
from .errors import TTSError
from .schemas import TTSJSONResponse, TTSRequest
from .synthesis import SynthesisRequest, synthesize
from .voices import cached_voice_count, resolve_voice_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model()
    yield
    model.unload_model()


app = FastAPI(
    title="OmniVoice Edu",
    description=(
        "Zero-shot multilingual TTS (600+ languages) powered by OmniVoice, "
        "with speech-ready text normalization for educational content"
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(TTSError)
async def _tts_error_handler(request: Request, exc: TTSError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Enforce the Bearer token when OMNIVOICE_API_KEY is set."""
    if settings.api_key is None:
        return
    if creds is None or not hmac.compare_digest(creds.credentials, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _run_synthesis(req: SynthesisRequest) -> tuple[bytes, float]:
    """Run the blocking pipeline in a worker thread, mapping failures to HTTP."""
    try:
        return await run_in_threadpool(synthesize, req)
    except (HTTPException, TTSError):
        raise
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    """Health check – confirms the model is loaded and ready."""
    v = model.version_info()
    return {
        "status": "ok" if model.is_loaded() else "loading",
        "model": MODEL_ID,
        "omnivoice_version": v["omnivoice"],
        "gpu": torch.cuda.is_available(),
        "device_name": v["gpu"],
        "cached_voices": cached_voice_count(),
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
    chunk_gap_ms: int = Form(80, description="Silence gap in ms between chunks"),
):
    """Generate TTS audio and return it as a downloadable WAV file.

    For voice cloning, upload the reference audio as ``ref_audio``.
    """
    upload_path: Optional[str] = None
    try:
        if ref_audio is not None:
            suffix = Path(ref_audio.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(await ref_audio.read())
            upload_path = f.name

        wav_bytes, _ = await _run_synthesis(
            SynthesisRequest(
                text=text,
                ref_audio_path=upload_path,
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
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="tts_output.wav"'},
        )
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)


@app.post(
    "/tts/json",
    response_model=TTSJSONResponse,
    dependencies=[Depends(require_api_key)],
)
async def tts_json(body: TTSRequest):
    """Generate TTS audio and return it as base64-encoded JSON.

    This is the most convenient endpoint for programmatic consumption.
    """
    ref_audio_path = (
        resolve_voice_path(body.ref_audio_path) if body.ref_audio_path else None
    )
    wav_bytes, duration = await _run_synthesis(
        SynthesisRequest(
            **body.model_dump(exclude={"ref_audio_path"}),
            ref_audio_path=ref_audio_path,
        )
    )
    return TTSJSONResponse(
        audio_base64=base64.b64encode(wav_bytes).decode("ascii"),
        sample_rate=SAMPLE_RATE,
        duration_seconds=round(duration, 3),
    )
