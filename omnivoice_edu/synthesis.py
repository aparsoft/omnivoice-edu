"""The synthesis pipeline: text → chunks → batched generation → WAV."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import audio
from .config import SAMPLE_RATE, settings
from .errors import InvalidRequest, TTSError
from .model import GPU_LOCK, get_model
from .text import Chunk, chunk_text_for_tts, preprocess_text_for_tts
from .voices import get_voice_clone_prompt

logger = logging.getLogger(__name__)


@dataclass
class SynthesisRequest:
    """Everything needed to synthesize one request (validated by the API)."""

    text: str
    ref_audio_path: Optional[str] = None  # already resolved, or a temp upload
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    seed: Optional[int] = None
    num_step: int = 48
    speed: float = 1.0
    duration: Optional[float] = None
    volume: float = 3.0
    skip_chunking: bool = False
    chunk_threshold: int = 350
    chunk_gap_ms: int = 80


def split_duration(
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


def synthesize_chunks(
    chunks: list[Chunk],
    voice_prompt,
    instruct: Optional[str],
    language: Optional[str],
    speed: float,
    durations: Optional[list[float]],
    gen_config,
) -> list[np.ndarray]:
    """Generate audio for every chunk, keeping one consistent voice.

    Chunks are synthesized in batched generate() calls (up to
    ``settings.max_batch`` at a time), which is much faster than one call
    per chunk.

    In voice-design / auto-voice mode each generate() call samples a fresh
    voice, so independently generated chunks would drift to different
    speakers. Like OmniVoice's own long-form generation, we synthesize the
    first chunk alone and use it as the voice-clone reference for the rest.
    Callers must hold model.GPU_LOCK.
    """
    model = get_model()
    texts = [c.text for c in chunks]
    audios: list[np.ndarray] = []

    def _run(start: int, end: int, prompt) -> None:
        out = model.generate(
            text=texts[start:end],
            voice_clone_prompt=prompt,
            instruct=instruct,
            language=language,
            speed=speed,
            duration=durations[start:end] if durations else None,
            generation_config=gen_config,
        )
        if len(out) != end - start:
            raise TTSError("Model returned no audio")
        audios.extend(out)

    start = 0
    if voice_prompt is None and len(chunks) > 1:
        _run(0, 1, None)
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=(torch.from_numpy(audios[0].astype(np.float32)), SAMPLE_RATE),
            ref_text=texts[0],
        )
        start = 1

    for batch_start in range(start, len(chunks), settings.max_batch):
        _run(
            batch_start,
            min(batch_start + settings.max_batch, len(chunks)),
            voice_prompt,
        )

    return audios


def synthesize(req: SynthesisRequest) -> tuple[bytes, float]:
    """Run the full pipeline and return (wav_bytes, duration_seconds).

    Blocking (GPU-bound) — call via run_in_threadpool from async endpoints.

    When ``skip_chunking`` is True, the full text is passed to the model in
    a single generate() call — no splitting, no silence gaps. Otherwise,
    texts longer than ``chunk_threshold`` characters are split at sentence /
    clause boundaries, synthesized in batches with a shared voice, and
    joined with a short silence gap. This prevents the model from speeding
    up on long inputs.
    """
    get_model()  # fail fast with 503 before any work if not loaded

    # Always preprocess (math/digit verbalization, markdown flattening, etc.)
    text = preprocess_text_for_tts(req.text).strip()
    if not text:
        raise InvalidRequest("Text is empty after preprocessing")
    logger.info("Preprocessed text: %.120s…", text)

    if req.skip_chunking:
        logger.info(
            "Chunking skipped (skip_chunking=True); synthesizing full text in one pass"
        )
        chunks = [Chunk(text=text, index=0, is_first=True, is_last=True)]
    else:
        chunks = chunk_text_for_tts(text, max_chunk_chars=req.chunk_threshold)

    if len(chunks) > 1:
        logger.info(
            "Auto-chunked text into %d segments (total %d chars)", len(chunks), len(text)
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
    # Generation config with postprocess_output DISABLED.
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
    # still run (they are not gated by this flag), so audio edges stay
    # click-free while the natural onset is preserved.
    # ------------------------------------------------------------------
    from omnivoice import OmniVoiceGenerationConfig

    gen_config = OmniVoiceGenerationConfig(
        num_step=req.num_step,
        postprocess_output=False,
    )

    with GPU_LOCK:
        if req.seed is not None:
            torch.manual_seed(req.seed)

        voice_prompt = (
            get_voice_clone_prompt(req.ref_audio_path, req.ref_text)
            if req.ref_audio_path
            else None
        )
        # instruct alongside a voice prompt is supported upstream: when the
        # two agree it stabilises the attributes it describes.
        segments = synthesize_chunks(
            chunks,
            voice_prompt=voice_prompt,
            instruct=req.instruct,
            language=req.language,
            speed=req.speed,
            durations=split_duration(req.duration, chunks, req.chunk_gap_ms),
            gen_config=gen_config,
        )

    wav = audio.join_with_gaps(segments, req.chunk_gap_ms)

    # With postprocess_output=False the model keeps the full soft onset of
    # the first word, but also a longer, variable run of leading silence.
    # Trim only that excess, at the -60 dBFS noise floor with a 120 ms
    # guard band, so no phoneme is ever clipped.
    wav = audio.trim_leading_silence(wav, threshold_db=-60.0, lead_guard_ms=120)
    duration_seconds = len(wav) / SAMPLE_RATE

    wav = audio.apply_volume(wav, req.volume)
    return audio.to_wav_bytes(wav), duration_seconds
