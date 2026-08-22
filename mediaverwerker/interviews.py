"""Structured interview workflow for RSS Reader."""

import json
import logging
import os
import re
import subprocess
from pathlib import Path

import requests
from openai import OpenAI
from vercel.blob import BlobClient

from .ai import generate_json
from .config import AUDIO_DIR, OPENAI_API_KEY, OPENAI_DIARIZATION_MODEL
from .tasks.download import download_url_audio, fetch_url_metadata
from .util import get_audio_duration, split_audio

logger = logging.getLogger("mediaverwerker")

MINIMUM_CONTENT_RETENTION = 0.75
MAX_TRANSCRIPTION_BYTES = 24 * 1024 * 1024

TURN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["turns"],
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text"],
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
            },
        }
    },
}

STRUCTURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "language", "speakerMappings", "chapters"],
    "properties": {
        "title": {"type": "string"},
        "language": {"type": "string"},
        "speakerMappings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker", "label", "confidence", "role", "roleConfidence"],
                "properties": {
                    "speaker": {"type": "string"},
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "role": {"type": "string", "enum": ["interviewer", "guest", "speaker"]},
                    "roleConfidence": {"type": "number"},
                },
            },
        },
        "chapters": {
            "type": "array",
            "minItems": 6,
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "startTime"],
                "properties": {"title": {"type": "string"}, "startTime": {"type": "number"}},
            },
        },
    },
}


class InterviewCallback:
    """Send monotonic, authenticated progress updates to RSS Reader."""

    def __init__(self, callback_url, interview_id, attempt):
        self.callback_url = callback_url
        self.interview_id = interview_id
        self.attempt = attempt
        self.sequence = 0
        self.secret = os.getenv("INTERVIEW_CALLBACK_SECRET")
        if not self.secret:
            raise RuntimeError("INTERVIEW_CALLBACK_SECRET ontbreekt")

    def send(self, status, *, detail=None, result=None, error=None):
        self.sequence += 1
        payload = {
            "interviewId": self.interview_id,
            "attempt": self.attempt,
            "sequence": self.sequence,
            "status": status,
        }
        if detail:
            payload["detail"] = detail
        if result:
            payload["result"] = result
        if error:
            payload["error"] = {"code": "processing_failed", "message": str(error)}
        response = requests.post(
            self.callback_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.secret}"},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Callback {status} geweigerd ({response.status_code}): {response.text[:1000]}")


def normalize_audio(source_path, interview_id):
    """Normalize every source to a stable mono MP3."""
    output_dir = AUDIO_DIR / "interviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{interview_id}.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "96k",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _value(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _transcribe_file(path, speaker_prefix=""):
    client = OpenAI(api_key=OPENAI_API_KEY)
    with path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=OPENAI_DIARIZATION_MODEL,
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
        )
    segments = _value(response, "segments", []) or []
    turns = []
    for index, segment in enumerate(segments):
        text = str(_value(segment, "text", "")).strip()
        if not text:
            continue
        speaker = str(_value(segment, "speaker", "Spreker 1"))
        turns.append(
            {
                "id": f"{speaker_prefix}{index}",
                "speaker": f"{speaker_prefix}{speaker}",
                "start": float(_value(segment, "start", 0)),
                "end": float(_value(segment, "end", 0)),
                "text": text,
            }
        )
    if not turns:
        raise RuntimeError("Diarization leverde geen transcriptbeurten op")
    return turns


def transcribe_diarized(audio_path):
    """Transcribe with speaker labels and timestamps, splitting only when required."""
    if audio_path.stat().st_size <= MAX_TRANSCRIPTION_BYTES:
        return _transcribe_file(audio_path)

    combined = []
    offset = 0.0
    for chunk_index, chunk in enumerate(split_audio(audio_path, chunk_duration_seconds=1200)):
        chunk_path = Path(chunk)
        turns = _transcribe_file(chunk_path, speaker_prefix=f"c{chunk_index}-")
        for turn in turns:
            turn["start"] += offset
            turn["end"] += offset
            combined.append(turn)
        offset += get_audio_duration(chunk_path) or 1200
    return combined


def _word_count(text):
    return len(str(text).split())


def clean_transcript(turns):
    """Correct errors without removing meaning or changing turn boundaries."""
    cleaned = []
    for batch_start in range(0, len(turns), 60):
        batch = turns[batch_start : batch_start + 60]
        compact = [{"id": turn["id"], "speaker": turn["speaker"], "text": turn["text"]} for turn in batch]
        result = generate_json(
            task="clean-interview-transcript",
            instructions=(
                "Corrigeer in de oorspronkelijke taal alleen herkenningsfouten, eigennamen, interpunctie, "
                "stopwoorden en directe herhaling. Behoud betekenis, taal, volgorde en precies alle IDs. "
                "Vat niets samen en voeg niets toe. Geef alleen id en gecorrigeerde tekst terug."
            ),
            input_text=json.dumps(compact, ensure_ascii=False),
            schema=TURN_SCHEMA,
            max_output_tokens=12000,
            reasoning_effort="low",
        )
        candidate = result.get("turns", [])
        expected_ids = [item["id"] for item in batch]
        candidate_by_id = {item.get("id"): item for item in candidate}
        if len(candidate) == len(batch) and set(candidate_by_id) == set(expected_ids):
            ordered_candidate = [candidate_by_id[item_id] for item_id in expected_ids]
        else:
            logger.warning("Transcriptredactie wijzigde beurt-IDs; oorspronkelijke batch blijft behouden")
            ordered_candidate = [{"id": item["id"], "text": item["text"]} for item in batch]
        normalized_candidate = []
        for original, edited in zip(batch, ordered_candidate):
            edited_text = str(edited.get("text", "")).strip() or original["text"].strip()
            normalized_candidate.append({**edited, "text": edited_text})
        original_words = sum(_word_count(item["text"]) for item in batch)
        cleaned_words = sum(_word_count(item["text"]) for item in normalized_candidate)
        retention = cleaned_words / max(original_words, 1)
        if retention < MINIMUM_CONTENT_RETENTION:
            raise RuntimeError(f"Volledigheidscontrole mislukt: {retention:.0%} tekst behouden; minimaal 75% vereist")
        for original, edited in zip(batch, normalized_candidate):
            cleaned.append({**original, "text": edited["text"]})
    return cleaned


def _normalize_language(value, episode):
    language = str(value or "").strip().lower()
    aliases = {
        "dutch": "nl",
        "nederlands": "nl",
        "english": "en",
        "engels": "en",
        "german": "de",
        "deutsch": "de",
        "duits": "de",
        "french": "fr",
        "français": "fr",
        "frans": "fr",
        "spanish": "es",
        "español": "es",
        "spaans": "es",
    }
    language = aliases.get(language, language)
    if re.fullmatch(r"[a-z]{2}", language):
        return language
    source_language = str(episode.get("language") or "").split("-", 1)[0].lower()
    return source_language if re.fullmatch(r"[a-z]{2}", source_language) else "und"


def _localized_labels(language):
    return {
        "nl": {"speaker": "Spreker", "interviewer": "Interviewer", "guest": "Gast"},
        "de": {"speaker": "Sprecher", "interviewer": "Interviewer", "guest": "Gast"},
        "fr": {"speaker": "Intervenant", "interviewer": "Intervieweur", "guest": "Invité"},
        "es": {"speaker": "Hablante", "interviewer": "Entrevistador", "guest": "Invitado"},
    }.get(language, {"speaker": "Speaker", "interviewer": "Interviewer", "guest": "Guest"})


def _fallback_speaker_labels(speakers, language):
    speaker_label = _localized_labels(language)["speaker"]
    return {speaker: f"{speaker_label} {index + 1}" for index, speaker in enumerate(speakers)}


def _is_generic_speaker_label(value):
    normalized = re.sub(r"[^a-zà-ÿ]+", " ", str(value or "").lower()).strip()
    return normalized in {
        "speaker",
        "spreker",
        "guest",
        "gast",
        "interviewer",
        "interviewee",
        "host",
        "unknown",
        "onbekend",
    }


def apply_interview_structure(turns, episode):
    transcript = "\n".join(f"[{t['start']:.1f}] {t['speaker']}: {t['text']}" for t in turns)
    result = generate_json(
        task="structure-interview",
        instructions=(
            "Maak metadata voor een trouw gesprekstranscript. Bepaal eerst de dominante GESPROKEN taal en "
            "geef die als ISO 639-1 code van twee letters. Schrijf titel en alle hoofdstukken uitsluitend in "
            "die gesproken taal, nooit automatisch in de taal van deze instructie. Titel: "
            "'<zekere gastnaam of het lokale neutrale woord voor gast> — <specifiek onderwerp>'; gebruik nooit "
            "letterlijke placeholders zoals onderwerp, topic of subject. Maak 6-10 concrete hoofdstukken in "
            "dezelfde taal met een starttijd die voorkomt in het transcript. "
            "Bepaal per spreker ook de rol en een afzonderlijke roleConfidence. Gebruik een echte naam alleen "
            "bij hoge zekerheid (confidence >= 0.9); anders een rol alleen als roleConfidence >= 0.8, en anders "
            "Spreker 1/2. Uitspraken blijven in hun gesproken taal."
        ),
        input_text=json.dumps(
            {
                "sourceTitle": episode.get("title"),
                "sourceDescription": episode.get("description"),
                "transcript": transcript,
            },
            ensure_ascii=False,
        ),
        schema=STRUCTURE_SCHEMA,
        max_output_tokens=5000,
        reasoning_effort="low",
        role="editorial",
    )
    language = _normalize_language(result["language"], episode)
    duration = max((turn["end"] for turn in turns), default=0)
    chapters = sorted(result["chapters"], key=lambda item: item["startTime"])
    if not 6 <= len(chapters) <= 10 or any(
        chapter["startTime"] < 0 or chapter["startTime"] > duration for chapter in chapters
    ):
        raise RuntimeError("Hoofdstukcontrole mislukt: verwacht 6-10 geldige tijdcodes")

    speakers = list(dict.fromkeys(turn["speaker"] for turn in turns))
    localized_labels = _localized_labels(language)
    labels = _fallback_speaker_labels(speakers, language)
    roles = {speaker: "speaker" for speaker in speakers}
    for mapping in result["speakerMappings"]:
        speaker = mapping["speaker"]
        if speaker not in labels:
            continue
        if mapping["roleConfidence"] >= 0.8:
            roles[speaker] = mapping["role"]
            labels[speaker] = localized_labels.get(mapping["role"], labels[speaker])
        if mapping["confidence"] >= 0.9 and not _is_generic_speaker_label(mapping["label"]):
            labels[speaker] = mapping["label"].strip()
    structured_turns = [
        {
            "speakerId": turn["speaker"],
            "speakerLabel": labels[turn["speaker"]],
            "startSec": turn["start"],
            "endSec": turn["end"],
            "text": turn["text"],
        }
        for turn in turns
    ]
    speaker_data = [
        {
            "id": speaker,
            "label": labels[speaker],
            "role": roles[speaker],
            "confidence": next((m["confidence"] for m in result["speakerMappings"] if m["speaker"] == speaker), 0),
        }
        for speaker in speakers
    ]
    structured_chapters = []
    for index, chapter in enumerate(chapters):
        turn_index = min(
            range(len(structured_turns)),
            key=lambda turn: abs(structured_turns[turn]["startSec"] - chapter["startTime"]),
        )
        structured_chapters.append(
            {
                "id": f"hoofdstuk-{index + 1}",
                "title": chapter["title"].strip(),
                "startSec": structured_turns[turn_index]["startSec"],
                "turnIndex": turn_index,
            }
        )
    title = result["title"].strip()
    if re.search(r"\s—\s(?:onderwerp|topic|subject)\s*$", title, re.IGNORECASE):
        title = f"{localized_labels['guest']} — {structured_chapters[0]['title']}"
    return {
        "title": title,
        "language": language,
        "speakers": speaker_data,
        "turns": structured_turns,
        "chapters": structured_chapters,
    }


def _upload_blob(client, pathname, data, content_type, *, multipart=False):
    blob = client.put(
        pathname,
        data,
        access="private",
        content_type=content_type,
        overwrite=True,
        multipart=multipart,
    )
    return {"url": _value(blob, "url"), "pathname": _value(blob, "pathname"), "etag": _value(blob, "etag")}


def upload_artifacts(interview_id, audio_path, structured):
    """Upload private audio plus immutable transcript and chapter manifests."""
    token = os.getenv("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN ontbreekt")
    client = BlobClient(token=token)
    prefix = f"interviews/{interview_id}"
    with audio_path.open("rb") as file:
        audio = _upload_blob(client, f"{prefix}/audio.mp3", file.read(), "audio/mpeg", multipart=True)
    manifest_bytes = json.dumps(structured, ensure_ascii=False).encode("utf-8")
    chapters_bytes = json.dumps(
        {
            "version": "1.2.0",
            "chapters": [{"startTime": item["startSec"], "title": item["title"]} for item in structured["chapters"]],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    manifest = _upload_blob(client, f"{prefix}/transcript.json", manifest_bytes, "application/json")
    chapters = _upload_blob(client, f"{prefix}/chapters.json", chapters_bytes, "application/json")
    return {"audio": audio, "manifest": manifest, "chapters": chapters}


def _published_iso(episode):
    published = str(episode.get("published") or "")
    return f"{published}T00:00:00.000Z" if len(published) == 10 else None


def process_interview(interview_id, source_url, callback_url, attempt=1):
    """Run one safe, structured interview job."""
    callback = InterviewCallback(callback_url, interview_id, attempt)
    try:
        callback.send("downloading", detail="Bronaudio downloaden")
        episode, _metadata = fetch_url_metadata(source_url, return_raw=True)
        downloaded = download_url_audio(episode.get("audio_url") or source_url)
        audio_path = normalize_audio(downloaded, interview_id)
        duration = get_audio_duration(audio_path)
        if not duration:
            raise RuntimeError("Kon de duur van de genormaliseerde audio niet bepalen")

        callback.send("transcribing", detail="Sprekers en tijdcodes herkennen")
        raw_turns = transcribe_diarized(audio_path)

        callback.send("editing", detail="Transcript corrigeren en hoofdstukken maken")
        turns = clean_transcript(raw_turns)
        structured = apply_interview_structure(turns, episode)

        callback.send("publishing", detail="Private audio en manifests publiceren")
        blobs = upload_artifacts(interview_id, audio_path, structured)
        result = {
            **structured,
            "audio": {
                "pathname": blobs["audio"]["pathname"],
                "url": blobs["audio"]["url"],
                "mimeType": "audio/mpeg",
                "bytes": audio_path.stat().st_size,
                "durationSec": duration,
            },
            "manifestBlobUrl": blobs["manifest"]["url"],
            "chaptersBlobUrl": blobs["chapters"]["url"],
        }
        if episode.get("title"):
            result["sourceTitle"] = episode["title"]
        if episode.get("podcast_name"):
            result["sourceName"] = episode["podcast_name"]
        if _published_iso(episode):
            result["sourcePublishedAt"] = _published_iso(episode)
        callback.send("ready", detail="Klaar", result=result)
        return result
    except Exception as exc:
        logger.exception("Interview %s failed", interview_id)
        try:
            callback.send("failed", detail="Verwerking mislukt", error=str(exc)[:2000])
        except Exception:
            logger.exception("Could not report failure for interview %s", interview_id)
        raise
