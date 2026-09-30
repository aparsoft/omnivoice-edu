# OmniVoice Edu

A production-ready text-to-speech server for [OmniVoice](https://github.com/k2-fsa/OmniVoice), the zero-shot multilingual text-to-speech model from k2-fsa that supports 600+ languages. It adds a text-normalization layer built for educational and technical content, so math, LaTeX, currency and markdown are read aloud the way a teacher would say them.

Built and maintained by [Aparsoft](https://aparsoft.com) as the voice layer for AI tutoring and oral-assessment products for Indian schools.

## Why this server

OmniVoice produces excellent speech, but a raw model call is not yet a service. This project adds:

- **Speakable text, always.** LLM output and textbook content are full of `3x + 5 = 20`, `$\frac{a}{b}$`, `₹3.5 crore`, `35°C` and markdown. A 1,500-line preprocessing pipeline turns them into natural spoken English before synthesis, and leaves Hindi and other languages intact.
- **Long text that stays natural.** OmniVoice speeds up on long inputs. The server splits text at sentence boundaries (including the Devanagari `।`), synthesizes the chunks in batches, and joins them with short pauses.
- **One consistent voice.** In voice-design and auto-voice modes the first chunk becomes the voice reference for the rest, so the speaker doesn't change partway through.
- **No clipped first words.** OmniVoice's default silence trimming can cut the soft onset of the first word. The server disables it and trims only true dead air (see [Implementation notes](#implementation-notes)).
- **Fast repeat requests.** Each reference voice is encoded, and transcribed by Whisper if needed, once. After that it's reused from memory and from disk, including across restarts.
- **Service basics.** Requests are serialized on the GPU in a worker thread, so `/health` stays responsive. The server also offers optional Bearer-token auth, a sandboxed voices directory, and version reporting.

Measured on an RTX PRO 4500 (Blackwell) with `num_step=32`: 43 seconds of cloned speech is generated in about 3 seconds.

## Requirements

- Python 3.10+ (tested on 3.12)
- An NVIDIA GPU with CUDA. Tested with CUDA 12.8 on an RTX PRO 4500 Blackwell (32 GB).
- About 4 GB of VRAM for OmniVoice, plus about 1.6 GB for Whisper if you clone voices without providing `ref_text`

## Installation

```bash
git clone https://github.com/aparsoft/omnivoice-edu.git
cd omnivoice-edu
python3 -m venv .venv
source .venv/bin/activate

# 1. PyTorch for your platform (CUDA 12.8 shown)
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 2. OmniVoice Edu and its dependencies (plus the test client extras)
pip install -e ".[test]"
```

Model weights download from Hugging Face on first start.

## Quick start

```bash
# Put reference voices (3–10 s clips) in the voices directory
mkdir -p voices
cp /path/to/narrator.wav voices/

omnivoice-edu            # or: python -m omnivoice_edu
```

The server listens on `http://0.0.0.0:8444`. It logs the versions it runs with at startup:

```
INFO  Versions: omnivoice-edu 0.1.0 | omnivoice 0.2.1 | torch 2.8.0+cu128 (CUDA 12.8) | transformers 5.12.0 | fastapi 0.136.3 | python 3.12.3
INFO  GPU: NVIDIA RTX PRO 4500 Blackwell
```

Interactive API docs are served at `http://localhost:8444/docs`.

Generate speech:

```bash
curl -s http://localhost:8444/tts/json \
  -H "Content-Type: application/json" \
  -d '{"text": "The area of a circle is $\\pi r^2$.", "ref_audio_path": "narrator.wav"}' \
  | python -c "import sys, json, base64; open('out.wav', 'wb').write(base64.b64decode(json.load(sys.stdin)['audio_base64']))"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OMNIVOICE_HOST` | `0.0.0.0` | Bind address |
| `OMNIVOICE_PORT` | `8444` | Port |
| `OMNIVOICE_DEVICE` | `cuda:0` | Device for the TTS model |
| `OMNIVOICE_DTYPE` | `float16` | `float16`, `bfloat16` or `float32` |
| `OMNIVOICE_VOICES_DIR` | `voices` | The only directory `ref_audio_path` may read from |
| `OMNIVOICE_CACHE_DIR` | `.cache/voice_prompts` | Where encoded voice prompts are stored between restarts |
| `OMNIVOICE_API_KEY` | unset | If set, `/tts` and `/tts/json` require `Authorization: Bearer <key>` |
| `OMNIVOICE_ASR_DEVICE` | model's device | Device for Whisper, e.g. `cpu` to save about 1.6 GB of VRAM |
| `OMNIVOICE_MAX_BATCH` | `8` | Maximum number of chunks synthesized together in one batch |

## API

### `GET /health`

Reports readiness and the runtime versions. It never requires an API key.

```json
{
  "status": "ok",
  "model": "k2-fsa/OmniVoice",
  "omnivoice_version": "0.2.1",
  "gpu": true,
  "device_name": "NVIDIA RTX PRO 4500 Blackwell",
  "cached_voices": 2,
  "versions": {
    "omnivoice_edu": "0.1.0",
    "omnivoice": "0.2.1",
    "torch": "2.8.0+cu128",
    "cuda": "12.8",
    "transformers": "5.12.0",
    "fastapi": "0.136.3",
    "python": "3.12.3",
    "gpu": "NVIDIA RTX PRO 4500 Blackwell"
  }
}
```

### `POST /tts/json`

Takes a JSON body and returns base64-encoded WAV audio.

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize (1–5000 characters) |
| `ref_audio_path` | string | — | Voice cloning: a file name inside `OMNIVOICE_VOICES_DIR` |
| `ref_text` | string | — | Transcript of the reference audio. If omitted, Whisper transcribes it once and the result is cached |
| `instruct` | string | — | Voice design attributes, e.g. `"female, low pitch, british accent"` |
| `language` | string | — | Language hint as an ID (`"hi"`) or name (`"Hindi"`). Omit to let the model infer it |
| `seed` | int | — | Seed for reproducible output. In auto-voice mode it fixes which voice is chosen |
| `num_step` | int | `48` | Diffusion steps (1–128). 32 is OmniVoice's default; 16 is fastest |
| `speed` | float | `1.0` | Speaking rate (>1 is faster). 1.0 is recommended; rates below 1.0 can clip the first word |
| `duration` | float | — | Fixed total duration in seconds (overrides `speed`). It's shared across chunks in proportion to their text length |
| `volume` | float | `3.0` | Gain (0–10). The output is then peak-limited to 0.95 so it can't clip |
| `skip_chunking` | bool | `false` | Synthesize the whole text in one pass. Use it when the caller has already split the text |
| `chunk_threshold` | int | `350` | Maximum characters per chunk (50–99999) |
| `chunk_gap_ms` | int | `80` | Silence between chunks, in milliseconds (0–500) |

Response:

```json
{ "audio_base64": "<base64 WAV>", "sample_rate": 24000, "duration_seconds": 8.43 }
```

### `POST /tts`

Accepts the same fields as multipart form data and returns the WAV file directly. For voice cloning, upload the reference audio in the `ref_audio` file field instead of naming a server-side file.

```bash
curl -s http://localhost:8444/tts -F text="Hello from a cloned voice." \
  -F ref_audio=@my_voice.wav -o out.wav
```

### Generation modes

| Mode | How to use it | Notes |
|---|---|---|
| Voice cloning | `ref_audio_path` (JSON) or a `ref_audio` upload (form) | The most stable mode. Use a clean 3–10 s clip in the same language as the target speech |
| Voice design | `instruct` | Attributes: gender, age, pitch, whisper style, English accent, Chinese dialect. Trained on Chinese and English |
| Auto voice | neither of the above | The model chooses a voice. Set `seed` to get the same voice every time |

`instruct` can be combined with a reference voice. When the two agree, it makes those attributes more stable.

## Text preprocessing

Every request passes through [`text/preprocessor.py`](omnivoice_edu/text/preprocessor.py). These are real outputs, checked by `tests/test_preprocessor.py`:

| Input | Spoken as |
|---|---|
| `3x + 5 = 20` | 3x plus five equals twenty |
| `x² + y² = z²` | x squared plus y squared equals z squared |
| `$\pi r^2$` | pi r squared |
| `$\frac{a}{b}$` | a over b |
| `₹3.5 crore` | three point five crore rupees |
| `₹2,500` | two thousand five hundred rupees |
| `$1,250` | one thousand two hundred fifty dollars |
| `12.5%` | twelve point five percent |
| `35°C` | thirty-five degrees Celsius |
| `H₂O` | H two O |
| `01/15/2024` | January fifteenth, twenty twenty-four |
| `3:45 PM` | three forty-five PM |
| `1st, 2nd, 3rd` | first, second, third |
| `**bold** and [a link](https://example.com)` | bold and a link |
| `Great job 🎉` | Great job |

It also handles LaTeX sums, integrals, limits, roots and Greek letters, as well as Unicode fractions, arrows and bullets, list markers, HTML entities, URLs and emails. It verbalizes English only. Text in other languages keeps its words, and only Unicode and punctuation are cleaned.

OmniVoice's inline controls pass through unchanged, so you can use them in any request:

- Non-verbal sounds: `[laughter]`, `[sigh]`, `[question-en]`, `[surprise-ah]`, …
- Pronunciation overrides with CMU phonemes: `He plays the [B EY1 S] guitar.`

## Long text

1. **Splitting.** Text longer than `chunk_threshold` is split at sentence ends (`. ! ? … ।`), then at clauses (`, ; : —`), and at word boundaries only as a last resort. Chunks never overlap.
2. **Consistent voice.** In cloning mode every chunk uses the same encoded reference. In design and auto modes the first chunk is synthesized alone and then used as the reference for the rest.
3. **Batching.** The remaining chunks are generated together in batches of up to `OMNIVOICE_MAX_BATCH`.
4. **Joining.** The chunks are joined with a `chunk_gap_ms` pause between them.

For short texts, or when the caller controls segmentation, set `skip_chunking: true`.

## Implementation notes

**First-word clipping.** OmniVoice's default post-processing trims leading silence at a fixed −50 dBFS threshold. The model starts each utterance with a soft, gradual onset, and that onset often sits below the threshold, so the start of the first word gets trimmed. The server sets `postprocess_output=False`, which skips only that trim; fades, padding and loudness normalization still run. It then removes excess leading silence at −60 dBFS and keeps a 120 ms guard band, so no phoneme is lost. Run `python -m tests.test_first_word_onset` to see the A/B measurement.

**Voice consistency across chunks.** Each OmniVoice `generate()` call without a reference samples a new voice, so chunks generated independently can sound like different speakers. Using the first chunk as the reference for the rest keeps one speaker. This is the same approach OmniVoice uses in its own long-form generation. In our measurements, speaker similarity between the start and end of a 43-second clip rose from 0.61 to 0.99.

## Architecture

The engine is independent of the web layer: `synthesis.py` raises plain exceptions (`errors.py`), and `api.py` turns them into HTTP responses.

```
omnivoice_edu/
├── api.py           FastAPI app: routes, Bearer auth, error mapping
├── synthesis.py     the pipeline: preprocess → chunk → batched generate → WAV
├── voices.py        voices-directory sandbox and the voice-prompt cache
├── audio.py         chunk joining, leading-silence trim, volume, WAV encoding
├── model.py         model loading, GPU lock, version reporting
├── schemas.py       request / response models
├── config.py        settings from environment variables
├── errors.py        engine errors with their HTTP status codes
├── text/
│   ├── preprocessor.py   text normalization for speech
│   └── chunker.py        sentence-aware chunking
└── __main__.py      entry point (`omnivoice-edu`, `python -m omnivoice_edu`)
tests/               unit tests and API test clients
```

Reference voices, generated audio and the voice cache (`voices/`, `output/`, `test_output/`, `.cache/`) stay local and are git-ignored.

## Tests

Unit tests run in seconds and need no GPU or server:

```bash
python -m tests.test_chunker        # sentence-aware chunking
python -m tests.test_preprocessor   # text normalization, incl. every README example
python -m tests.test_units          # voices sandbox, audio processing, duration split
```

The API tests expect a running server. Point them elsewhere with `TTS_API_URL`, and choose reference voices with `TTS_TEST_VOICE_EN`, `TTS_TEST_VOICE_HI` and `TTS_TEST_VOICE_FILE`.

```bash
python -m tests.test_tts_api          # auto voice, voice design, upload cloning
python -m tests.test_math_tts         # math text, speed, volume, Hindi cloning
python -m tests.test_skip_chunking    # chunked vs single-pass synthesis
python -m tests.test_first_word_onset # onset A/B (loads the model directly)
```

## Security

- `ref_audio_path` resolves only inside `OMNIVOICE_VOICES_DIR`. Paths outside it are rejected.
- Authentication is off by default. Set `OMNIVOICE_API_KEY` before exposing the server beyond a trusted network, and ideally put it behind a reverse proxy with TLS.

## Responsible use

Voice cloning is powerful. Clone only voices you own or have explicit, documented consent to use. Never use this software for impersonation, fraud or deception, and follow the laws that apply to you. See also OmniVoice's [disclaimer](https://github.com/k2-fsa/OmniVoice#disclaimer).

## Support

This project is shared as-is. Issues and pull requests are welcome, but responses aren't guaranteed. For commercial deployments and custom voice AI work, contact [Aparsoft](https://aparsoft.com) at contact@aparsoft.com.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

OmniVoice is © the k2-fsa authors (Apache-2.0). Whisper is © OpenAI (MIT).
