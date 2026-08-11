# AI model routing

All language-model calls use semantic roles from `mediaverwerker/config.py` and
the shared helpers in `mediaverwerker/ai.py`. Model IDs must not be added inside
task modules.

| Role | Default | Used for |
| --- | --- | --- |
| Bulk | `gpt-5-nano` | Reserved for high-volume, low-complexity work |
| Structured | `gpt-5.4-nano` | Command parsing, segment selection, quality scores |
| Polish | `gpt-5.6-luna` | Transcript cleanup |
| Editorial | `gpt-5.6-terra` | Podcast article writing and improvement |
| Transcription | `whisper-large-v3-turbo` on Groq | Speech-to-text with timestamps |

Each default can be overridden with the corresponding environment variable.
Production keeps Groq as the transcription provider because its timestamped
Whisper path is substantially cheaper. OpenAI `whisper-1` remains available as
an operational fallback through `TRANSCRIPTION_PROVIDER=openai`.

Review the defaults quarterly against the providers' current model and pricing
pages. Runtime logs emit one JSON `ai_request` record per language-model call,
including task, role, model, latency, token usage, and failure type.
