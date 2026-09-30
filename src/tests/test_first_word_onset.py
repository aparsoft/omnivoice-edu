"""A/B check: postprocess_output=False preserves the first word's onset.

Generates the same short phrase with OmniVoice's default post-processing
(postprocess_output=True, which runs remove_silence and can clip the soft
onset of the first phoneme) and with the server's config
(postprocess_output=False).

We measure the energy in the first 200 ms of generated audio. The default
mode trims the soft onset, so its leading-window energy ramps up later; the
server's config keeps the natural onset, so meaningful energy appears earlier.

Loads the model directly (needs a GPU; the API server is not required):
    python -m src.tests.test_first_word_onset

Set TTS_TEST_VOICE_FILE to choose the reference audio.
"""

import os
from pathlib import Path

import numpy as np
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REF = os.environ.get(
    "TTS_TEST_VOICE_FILE",
    str(_PROJECT_ROOT / "local_folder" / "voice_prompts" / "af_heart.wav"),
)
# Starts with a soft fricative "S" — the case most prone to onset clipping.
PHRASE = "So today we are going to learn something amazing about numbers."
SR = 24000

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", device_map="cuda:0", dtype="float16"
)


def gen(postprocess: bool) -> np.ndarray:
    cfg = OmniVoiceGenerationConfig(num_step=48, postprocess_output=postprocess)
    out = model.generate(text=PHRASE, ref_audio=REF, speed=0.85, generation_config=cfg)
    return np.asarray(out[0]).astype(np.float64)


def onset_report(name: str, a: np.ndarray) -> None:
    # Time (ms) until cumulative energy first reaches 1% of total — a proxy
    # for "how much of the start is silence/trimmed".
    e = a**2
    total = e.sum() + 1e-12
    csum = np.cumsum(e) / total
    idx = int(np.argmax(csum >= 0.01))
    onset_ms = idx / SR * 1000.0
    first200 = a[: int(0.2 * SR)]
    rms200 = float(np.sqrt(np.mean(first200**2)))
    print(
        f"{name:>26}: dur={len(a)/SR:5.2f}s  onset@1%={onset_ms:6.1f}ms  first200ms_rms={rms200:.4f}"
    )


a_true = gen(True)
a_false = gen(False)
print("\n--- First-word onset comparison ---")
onset_report("postprocess=True (default)", a_true)
onset_report("postprocess=False (server)", a_false)
print("\nLower onset@1% and higher first200ms_rms for the server config => first word preserved.")
