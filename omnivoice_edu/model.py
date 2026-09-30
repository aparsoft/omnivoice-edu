"""Loading and holding the OmniVoice model."""

from __future__ import annotations

import logging
import sys
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import torch

from . import __version__
from .config import MODEL_ID, settings
from .errors import ModelNotReady

logger = logging.getLogger(__name__)

_MODEL = None

# One model instance on one GPU: serialize generate() calls so concurrent
# requests don't interleave on the GPU (or race the lazy Whisper load).
# Inference runs in a worker thread, so /health stays responsive meanwhile.
GPU_LOCK = threading.Lock()

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def get_model():
    """The loaded OmniVoice model; raises ModelNotReady before startup."""
    if _MODEL is None:
        raise ModelNotReady("Model not loaded yet")
    return _MODEL


def is_loaded() -> bool:
    return _MODEL is not None


def version_info() -> dict:
    """Versions of this server, the model package and the runtime stack."""

    def _pkg(name: str) -> Optional[str]:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "omnivoice_edu": __version__,
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
    global _MODEL
    if _MODEL is not None:
        return

    v = version_info()
    logger.info(
        "Versions: omnivoice-edu %s | omnivoice %s | torch %s (CUDA %s) | "
        "transformers %s | fastapi %s | python %s",
        v["omnivoice_edu"],
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

    if settings.dtype not in _DTYPES:
        raise ValueError(
            f"OMNIVOICE_DTYPE must be one of {sorted(_DTYPES)}, got {settings.dtype!r}"
        )

    kwargs: dict = {}
    # Whisper (for auto-transcribing ref audio) defaults to the model's device.
    if settings.asr_device:
        kwargs["asr_device"] = settings.asr_device

    _MODEL = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=settings.device,
        dtype=_DTYPES[settings.dtype],
        **kwargs,
    )
    logger.info(
        "OmniVoice model loaded successfully on %s (%s)", settings.device, settings.dtype
    )


def unload_model() -> None:
    """Release the model and free GPU memory."""
    global _MODEL
    if _MODEL is not None:
        _MODEL = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed.")
