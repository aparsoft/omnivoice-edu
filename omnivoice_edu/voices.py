"""Reference voices: the sandboxed voices directory and the prompt cache."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .config import settings
from .errors import InvalidRequest
from .model import get_model

logger = logging.getLogger(__name__)

# Encoded reference voices, keyed by hash(audio bytes + ref_text). Kept in
# memory (LRU) and on disk, so the reference audio is tokenized — and, when
# ref_text is omitted, Whisper-transcribed — once per voice, not per request.
_PROMPT_CACHE: "OrderedDict[str, object]" = OrderedDict()
_PROMPT_CACHE_SIZE = 64


def cached_voice_count() -> int:
    return len(_PROMPT_CACHE)


def resolve_voice_path(ref_audio_path: str) -> str:
    """Resolve a requested reference audio file inside the voices directory.

    Accepts a file name relative to the voices directory ("narrator.wav"),
    a path relative to the working directory, or an absolute path — but only
    if it resolves to a file inside the voices directory.
    """
    voices_dir = settings.voices_dir
    requested = Path(ref_audio_path)
    candidates = (
        [requested]
        if requested.is_absolute()
        else [voices_dir / requested, Path.cwd() / requested]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(voices_dir) and resolved.is_file():
            return str(resolved)
    raise InvalidRequest(
        f"ref_audio_path not found in the voices directory: {ref_audio_path}"
    )


def get_voice_clone_prompt(ref_audio_path: str, ref_text: Optional[str]):
    """Return a cached VoiceClonePrompt for this reference audio + text.

    OmniVoice's generate(ref_audio=...) re-encodes the reference audio —
    and re-runs Whisper when ref_text is omitted — on every call, i.e. once
    per chunk. Encoding once and reusing the prompt avoids all of that.
    The cache key hashes the audio bytes, so edited or re-uploaded files
    are handled correctly. Callers must hold model.GPU_LOCK.
    """
    from omnivoice import VoiceClonePrompt

    h = hashlib.sha256(Path(ref_audio_path).read_bytes())
    h.update(b"\0" + (ref_text or "").encode("utf-8"))
    key = h.hexdigest()[:32]

    prompt = _PROMPT_CACHE.get(key)
    if prompt is not None:
        _PROMPT_CACHE.move_to_end(key)
        return prompt

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = settings.cache_dir / f"{key}.pt"
    if cache_file.exists():
        try:
            prompt = VoiceClonePrompt.load(str(cache_file))
        except Exception:
            logger.warning("Ignoring unreadable voice cache %s", cache_file.name)
    if prompt is None:
        logger.info("Encoding reference voice %s", Path(ref_audio_path).name)
        prompt = get_model().create_voice_clone_prompt(
            ref_audio=ref_audio_path, ref_text=ref_text
        )
        prompt.save(str(cache_file))

    _PROMPT_CACHE[key] = prompt
    if len(_PROMPT_CACHE) > _PROMPT_CACHE_SIZE:
        _PROMPT_CACHE.popitem(last=False)
    return prompt
