"""Audio transcription via OpenAI Whisper API or Groq."""

import logging
import time

from openai import OpenAI

from ..config import (
    GROQ_API_KEY,
    MAX_WHISPER_SIZE,
    OPENAI_API_KEY,
    TRANSCRIPTION_PROVIDER,
    TRANSCRIPTS_DIR,
)
from ..util import retry_with_backoff, sanitize_filename, split_audio

logger = logging.getLogger("mediaverwerker")


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

        if timestamps:
            return {"text": full_text, "segments": all_segments}
        return full_text
    else:
        if timestamps:
            result = transcribe_single_file(client, model, audio_path, language, timestamps=True)
            segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.segments]
            return {"text": result.text, "segments": segments}
        else:
            return transcribe_single_file(client, model, audio_path, language)

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
