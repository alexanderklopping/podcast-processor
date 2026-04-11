"""Audio transcription via OpenAI Whisper API or Groq."""

import logging
import time

from anthropic import Anthropic
from openai import OpenAI

from ..config import (
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    MAX_WHISPER_SIZE,
    OPENAI_API_KEY,
    TRANSCRIPTION_PROVIDER,
    TRANSCRIPTS_DIR,
)
from ..util import retry_with_backoff, sanitize_filename, split_audio

logger = logging.getLogger("mediaverwerker")

TRANSCRIPT_CLEANUP_MAX_ITERATIONS = 2
TRANSCRIPT_CLEANUP_MIN_WORDS = 500  # Skip cleanup for very short transcripts

CLEANUP_SYSTEM_PROMPT = """Je bent een transcriptie-editor.
Je krijgt een ruwe transcriptie van een podcast aflevering
(via Whisper/Groq spraakherkenning).
Je taak is de tekst opschonen ZONDER inhoud te veranderen.

Regels:
1. **Eigennamen corrigeren**: Fix foute spellingen van bekende namen
   (personen, bedrijven, producten). Voorbeelden:
   "Elan Musk" → "Elon Musk", "Chat GBT" → "ChatGPT",
   "Antrophic" → "Anthropic", "Goegel" → "Google".
2. **Gebroken zinnen samenvoegen**: Whisper breekt soms zinnen op verkeerde plekken.
   Voeg fragmenten samen tot vloeiende zinnen.
3. **Filler words verwijderen**: Verwijder "um", "uh", "eh", "you know",
   "like" (als filler), "sort of", "kind of" (als filler),
   "eigenlijk" (als filler), "zeg maar".
4. **Herhalingen verwijderen**: Als dezelfde zin of frase direct herhaald wordt (stotteren/herhaling), houd er één.
5. **Niets toevoegen**: Voeg GEEN nieuwe inhoud toe. Geen samenvattingen, geen commentaar, geen headers.
6. **Niets weglaten**: Verwijder geen inhoudelijke zinnen. Alleen filler en herhalingen.

Geef de opgeschoonde tekst terug. Niets anders."""

CLEANUP_SCORE_PROMPT = """Vergelijk de originele en opgeschoonde transcriptie. Beoordeel:

1. **eigennamen** (1-10): Zijn bekende namen correct gespeld?
2. **vloeiendheid** (1-10): Lezen de zinnen vloeiend? Geen gebroken fragmenten?
3. **schoonheid** (1-10): Zijn filler words en herhalingen verwijderd?
4. **volledigheid** (1-10): Is alle inhoud behouden? Niets onterecht verwijderd?

Antwoord ALLEEN in JSON:
{
  "scores": {
    "eigennamen": N,
    "vloeiendheid": N,
    "schoonheid": N,
    "volledigheid": N
  },
  "gemiddelde": N,
  "problemen": ["probleem 1", "probleem 2"]
}"""


def _cleanup_transcript(text):
    """Run iterative cleanup loop on raw transcript text.

    Fixes proper nouns, merges broken sentences, removes filler words.
    Returns cleaned text.
    """
    word_count = len(text.split())
    if word_count < TRANSCRIPT_CLEANUP_MIN_WORDS:
        logger.info(f"Transcript too short for cleanup ({word_count} words), skipping")
        return text

    # Guard: skip if text is very large (>200k chars ~ 50k words)
    if len(text) > 200_000:
        logger.info(f"Transcript too large for cleanup ({len(text):,} chars), skipping")
        return text

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    current = text
    for iteration in range(TRANSCRIPT_CLEANUP_MAX_ITERATIONS):
        logger.info(f"Transcript cleanup iteration {iteration + 1}/{TRANSCRIPT_CLEANUP_MAX_ITERATIONS}")

        # Cleanup pass
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16000,
            system=CLEANUP_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": current}],
        )
        cleaned = message.content[0].text.strip()

        # Sanity check: cleaned version shouldn't lose more than 20% of content
        if len(cleaned.split()) < word_count * 0.7:
            logger.warning(
                f"Cleanup removed too much content ({len(cleaned.split())} vs {word_count} words), keeping original"
            )
            return current

        # Score pass
        score_message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=CLEANUP_SCORE_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"**ORIGINEEL:**\n{text[:5000]}\n\n**OPGESCHOOND:**\n{cleaned[:5000]}",
                }
            ],
        )

        try:
            import json

            score_text = score_message.content[0].text.strip()
            if score_text.startswith("```"):
                lines = score_text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                score_text = "\n".join(lines).strip()
            score_result = json.loads(score_text)
            scores = score_result.get("scores", {})
            avg = score_result.get("gemiddelde", 0)
            if not avg:
                dims = ["eigennamen", "vloeiendheid", "schoonheid", "volledigheid"]
                avg = sum(scores.get(d, 0) for d in dims) / len(dims)

            logger.info("Cleanup scores: " + " | ".join(f"{k}={v}" for k, v in scores.items()) + f" | avg={avg:.1f}")

            problems = score_result.get("problemen", [])
            if avg >= 8 or not problems:
                logger.info(f"Transcript cleanup complete: avg {avg:.1f}/10")
                return cleaned

            # Feed problems back into the next iteration
            current = cleaned

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse cleanup score: {e}")
            return cleaned

    return current


def _get_transcription_client():
    """Return (client, model) based on configured provider."""
    if TRANSCRIPTION_PROVIDER == "groq":
        logger.info("Using Groq for transcription (whisper-large-v3-turbo)")
        return (
            OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY),
            "whisper-large-v3-turbo",
        )
    logger.info("Using OpenAI for transcription (whisper-1)")
    return OpenAI(api_key=OPENAI_API_KEY), "whisper-1"


@retry_with_backoff(max_retries=3, delay=5)
def transcribe_single_file(client, model, audio_path, language="en", timestamps=False):
    """Transcribe a single audio file with retry logic."""
    with open(audio_path, "rb") as audio_file:
        response_format = "verbose_json" if timestamps else "text"
        result = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language,
            response_format=response_format,
        )
        return result


def transcribe_audio(audio_path, language="en", timestamps=False):
    """Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to audio/video file.
        language: Language code (e.g., "en", "nl").
        timestamps: If True, return verbose_json with segment timestamps.

    Returns:
        If timestamps=False: plain text transcript string.
        If timestamps=True: dict with 'text' and 'segments' (each with start/end/text).
    """
    logger.info(f"Transcribing: {audio_path.name} (language: {language})")

    file_size = audio_path.stat().st_size
    client, model = _get_transcription_client()

    if file_size > MAX_WHISPER_SIZE:
        logger.info(f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds 25MB limit, splitting...")

        chunk_paths = split_audio(audio_path)
        if not chunk_paths:
            raise Exception("Failed to split audio file into chunks")

        all_text = []
        all_segments = []
        time_offset = 0.0

        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}...")
            try:
                if timestamps:
                    result = transcribe_single_file(client, model, chunk_path, language, timestamps=True)
                    all_text.append(result.text)
                    for seg in result.segments:
                        all_segments.append(
                            {
                                "start": seg["start"] + time_offset,
                                "end": seg["end"] + time_offset,
                                "text": seg["text"],
                            }
                        )
                    # Estimate chunk duration from last segment
                    if result.segments:
                        time_offset += result.segments[-1]["end"]
                else:
                    transcript = transcribe_single_file(client, model, chunk_path, language)
                    all_text.append(transcript)

                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i + 1}: {e}")
                raise
            finally:
                try:
                    chunk_path.unlink()
                except Exception:
                    pass

        # Clean up chunks directory
        chunks_dir = audio_path.parent / "chunks"
        try:
            if chunks_dir.exists() and not any(chunks_dir.iterdir()):
                chunks_dir.rmdir()
        except Exception:
            pass

        full_text = " ".join(all_text)
        logger.info(f"Combined {len(all_text)} chunks into full transcript")

        # Run cleanup loop on combined text
        full_text = _cleanup_transcript(full_text)

        if timestamps:
            return {"text": full_text, "segments": all_segments}
        return full_text
    else:
        if timestamps:
            result = transcribe_single_file(client, model, audio_path, language, timestamps=True)
            segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.segments]
            cleaned_text = _cleanup_transcript(result.text)
            return {"text": cleaned_text, "segments": segments}
        else:
            raw = transcribe_single_file(client, model, audio_path, language)
            return _cleanup_transcript(raw)

    logger.info("Transcription complete")


def save_transcript(episode, transcript):
    """Save transcript to file."""
    storage_key = episode.get("feed_storage_key") or episode.get("podcast_name", "unknown")
    filename = f"{storage_key}_{sanitize_filename(episode['title'])}.txt"
    filepath = TRANSCRIPTS_DIR / filename

    text = transcript if isinstance(transcript, str) else transcript.get("text", "")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {episode['title']}\n")
        if episode.get("podcast_name"):
            f.write(f"Bron: {episode['podcast_name']}\n")
        f.write(f"Gepubliceerd: {episode['published']}\n\n")
        if episode.get("source_url"):
            f.write(f"URL: {episode['source_url']}\n\n")
        f.write("---\n\n")
        f.write(text)

    logger.info(f"Transcript saved: {filename}")
    return filepath
