"""Audio and video download tasks."""

import base64
import html
import json
import logging
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..config import (
    AUDIO_DIR,
    INDIVIDUAL_FEED_SLUG,
    YTDLP_COOKIES_B64,
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_FROM_BROWSER,
    YTDLP_EXTRACTOR_ARGS,
    YTDLP_IMPERSONATE,
    YTDLP_JS_RUNTIMES,
    YTDLP_REMOTE_COMPONENTS,
)
from ..util import retry_with_backoff, sanitize_filename

logger = logging.getLogger("mediaverwerker")
_YTDLP_COOKIES_TEMP_FILE = None


class YtDlpError(RuntimeError):
    """Raised when yt-dlp fails and includes its useful output."""


def _yt_dlp_cmd():
    """Build a yt-dlp command using the active Python environment."""
    cmd = [sys.executable, "-m", "yt_dlp"]
    cookies_file = _yt_dlp_cookies_file()
    if YTDLP_JS_RUNTIMES:
        cmd.extend(["--js-runtimes", YTDLP_JS_RUNTIMES])
    if YTDLP_REMOTE_COMPONENTS:
        cmd.extend(["--remote-components", YTDLP_REMOTE_COMPONENTS])
    if YTDLP_EXTRACTOR_ARGS and not cookies_file:
        cmd.extend(["--extractor-args", YTDLP_EXTRACTOR_ARGS])
    if YTDLP_IMPERSONATE:
        cmd.extend(["--impersonate", YTDLP_IMPERSONATE])
    if YTDLP_COOKIES_FROM_BROWSER:
        cmd.extend(["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER])
    elif cookies_file:
        cmd.extend(["--cookies", cookies_file])
    return cmd


def _yt_dlp_cookies_file():
    """Return a cookies file path, creating one from the base64 secret if needed."""
    global _YTDLP_COOKIES_TEMP_FILE
    if YTDLP_COOKIES_FILE:
        return YTDLP_COOKIES_FILE
    if not YTDLP_COOKIES_B64:
        return None
    if _YTDLP_COOKIES_TEMP_FILE:
        return _YTDLP_COOKIES_TEMP_FILE

    try:
        cookie_bytes = base64.b64decode(YTDLP_COOKIES_B64, validate=True)
    except Exception as exc:
        raise YtDlpError("YTDLP_COOKIES_B64 is not valid base64") from exc

    temp_file = tempfile.NamedTemporaryFile("wb", delete=False, prefix="yt-dlp-cookies-", suffix=".txt")
    with temp_file:
        temp_file.write(cookie_bytes)
    _YTDLP_COOKIES_TEMP_FILE = temp_file.name
    return _YTDLP_COOKIES_TEMP_FILE


def _run_yt_dlp(cmd):
    """Run yt-dlp and preserve stderr/stdout when it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result

    details = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part and part.strip())
    if not details:
        details = f"yt-dlp exited with status {result.returncode}"
    raise YtDlpError(details)


def _metadata_to_episode(metadata, url):
    source_url = metadata.get("webpage_url") or metadata.get("original_url") or metadata.get("url") or url
    extractor = (metadata.get("extractor_key") or metadata.get("extractor") or "url").lower()
    media_id = metadata.get("id")
    guid = f"url:{extractor}:{media_id}" if media_id else f"url:{source_url}"

    source_name = (
        metadata.get("channel")
        or metadata.get("uploader")
        or metadata.get("creator")
        or metadata.get("series")
        or metadata.get("podcast")
        or urlparse(source_url).netloc
        or "Onbekende bron"
    )

    upload_date = metadata.get("upload_date", "")
    if len(upload_date) == 8 and upload_date.isdigit():
        published = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    elif metadata.get("release_timestamp"):
        published = datetime.utcfromtimestamp(int(metadata["release_timestamp"])).strftime("%Y-%m-%d")
    elif metadata.get("timestamp"):
        published = datetime.utcfromtimestamp(int(metadata["timestamp"])).strftime("%Y-%m-%d")
    else:
        published = metadata.get("release_date") or ""

    language = (metadata.get("language") or "en").split("-", 1)[0].lower()
    if len(language) > 5:
        language = "en"

    return {
        "guid": guid,
        "title": metadata.get("title") or media_id or "Untitled",
        "published": str(published),
        "audio_url": source_url,
        "description": metadata.get("description") or "",
        "podcast_name": source_name,
        "language": language or "en",
        "source_type": "individual_url",
        "source_url": source_url,
        "feed_storage_key": INDIVIDUAL_FEED_SLUG,
    }


def search_podcast(query):
    """Search for a podcast by name using the iTunes Search API.

    Args:
        query: Podcast name to search for (e.g., "Hard Fork").

    Returns:
        dict with 'name', 'url' (RSS feed), 'language', or None if not found.
    """
    logger.info(f"Searching for podcast: {query}")
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "podcast", "entity": "podcast", "limit": 5},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            logger.warning(f"No podcast found for: {query}")
            return None

        # Return the first result with a feed URL
        for result in data["results"]:
            feed_url = result.get("feedUrl")
            if feed_url:
                name = result.get("trackName", query)
                # Guess language from country
                country = result.get("country", "")
                lang = "nl" if country == "NLD" else "en"
                logger.info(f"Found podcast: {name} -> {feed_url}")
                return {"name": name, "url": feed_url, "language": lang}

        logger.warning(f"No podcast with RSS feed found for: {query}")
        return None
    except Exception as e:
        logger.error(f"Podcast search failed: {e}")
        return None


@retry_with_backoff(max_retries=3, delay=10)
def download_episode(episode):
    """Download podcast audio file with retry logic."""
    filename = sanitize_filename(episode["title"]) + ".mp3"
    filepath = AUDIO_DIR / filename
    temp_filepath = filepath.with_suffix(".mp3.tmp")

    if filepath.exists():
        if filepath.stat().st_size > 1000:
            logger.info(f"Audio already exists: {filename}")
            return filepath
        else:
            logger.warning(f"Existing file too small, re-downloading: {filename}")
            filepath.unlink()

    logger.info(f"Downloading: {episode['title']}")

    response = requests.get(
        episode["audio_url"], stream=True, timeout=300, headers={"User-Agent": "Mediaverwerker/1.0"}
    )
    response.raise_for_status()

    expected_size = int(response.headers.get("content-length", 0))

    downloaded_size = 0
    with open(temp_filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded_size += len(chunk)

    if expected_size > 0 and downloaded_size < expected_size * 0.95:
        temp_filepath.unlink()
        raise Exception(f"Incomplete download: {downloaded_size}/{expected_size} bytes")

    temp_filepath.replace(filepath)
    logger.info(f"Downloaded: {filename} ({downloaded_size / (1024 * 1024):.1f}MB)")
    return filepath


def download_video(url, output_dir=None):
    """Download video using yt-dlp. Supports YouTube, NPO, and most platforms."""
    if output_dir is None:
        output_dir = AUDIO_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading video: {url}")

    # First get the filename
    cmd = _yt_dlp_cmd() + [
        "--print",
        "filename",
        "-o",
        "%(title)s.%(ext)s",
        url,
    ]
    result = _run_yt_dlp(cmd)
    expected_filename = result.stdout.strip()

    # Download
    cmd = _yt_dlp_cmd() + [
        "-o",
        str(output_dir / "%(title)s.%(ext)s"),
        url,
    ]
    _run_yt_dlp(cmd)

    output_path = output_dir / expected_filename
    if output_path.exists():
        logger.info(f"Video downloaded: {output_path.name}")
        return output_path

    # Fallback: find the most recent file in output_dir
    files = sorted(output_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        logger.info(f"Video downloaded: {files[0].name}")
        return files[0]

    raise Exception("Download completed but output file not found")


def fetch_url_metadata(url, *, return_raw=False):
    """Fetch metadata for a single media URL via yt-dlp."""
    logger.info(f"Fetching URL metadata: {url}")

    cmd = _yt_dlp_cmd() + [
        "--dump-single-json",
        "--no-download",
        "--no-playlist",
        "--ignore-no-formats-error",
        "--quiet",
        "--no-warnings",
        url,
    ]
    result = _run_yt_dlp(cmd)
    metadata = json.loads(result.stdout)
    episode = _metadata_to_episode(metadata, url)
    if return_raw:
        return episode, metadata
    return episode


def _caption_language_candidates(language):
    language = (language or "en").split("-", 1)[0].lower()
    candidates = [language, f"{language}-orig", "en", "en-orig", "nl", "nl-orig"]
    return list(dict.fromkeys(candidates))


def _select_caption_track(captions, language):
    if not captions:
        return None, None

    def preferred_track(tracks):
        for ext in ("json3", "vtt", "srt"):
            track = next((item for item in tracks if item.get("ext") == ext), None)
            if track:
                return track
        return None

    for candidate in _caption_language_candidates(language):
        matching_keys = [key for key in captions if key == candidate or key.startswith(f"{candidate}-")]
        for key in matching_keys:
            tracks = captions.get(key) or []
            track = preferred_track(tracks)
            if track:
                return key, track

    for key, tracks in captions.items():
        track = preferred_track(tracks)
        if track:
            return key, track

    return None, None


def _youtube_json3_to_transcript(data):
    segments = []
    parts = []
    for event in data.get("events", []):
        text = "".join(segment.get("utf8", "") for segment in event.get("segs") or [])
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        if not text:
            continue

        start = float(event.get("tStartMs") or 0) / 1000
        duration = float(event.get("dDurationMs") or 0) / 1000
        segments.append({"start": start, "end": start + duration, "text": text})
        parts.append(text)

    transcript_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if not transcript_text:
        return None

    return {"text": transcript_text, "segments": segments}


def _text_caption_to_transcript(caption_text):
    lines = []
    for line in caption_text.splitlines():
        line = line.strip()
        if (
            not line
            or line == "WEBVTT"
            or line.isdigit()
            or "-->" in line
            or line.startswith(("Kind:", "Language:"))
        ):
            continue
        line = html.unescape(re.sub(r"<[^>]+>", "", line)).strip()
        if line:
            lines.append(line)

    transcript_text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not transcript_text:
        return None
    return {"text": transcript_text, "segments": []}


def fetch_youtube_caption_transcript(metadata, language="en"):
    """Fetch a transcript from YouTube captions present in yt-dlp metadata."""
    extractor = (metadata.get("extractor_key") or metadata.get("extractor") or "").lower()
    if extractor != "youtube":
        return None

    for caption_kind in ("subtitles", "automatic_captions"):
        selected_language, track = _select_caption_track(metadata.get(caption_kind), language)
        if not track:
            continue

        response = requests.get(
            track["url"],
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        if track.get("ext") == "json3":
            transcript = _youtube_json3_to_transcript(response.json())
        else:
            transcript = _text_caption_to_transcript(response.text)
        if transcript:
            transcript["language"] = selected_language
            transcript["source"] = f"youtube_{caption_kind}"
            logger.info(f"Using YouTube {caption_kind} transcript ({selected_language})")
            return transcript

    subtitles = len(metadata.get("subtitles") or {})
    automatic = len(metadata.get("automatic_captions") or {})
    logger.info(f"No usable YouTube captions found in metadata (subtitles={subtitles}, automatic={automatic})")
    return None


def download_url_audio(url, output_dir=None):
    """Download audio from a single media URL via yt-dlp."""
    if output_dir is None:
        output_dir = AUDIO_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading audio from URL: {url}")

    output_template = str(output_dir / "%(title).100s [%(id)s].%(ext)s")
    cmd = _yt_dlp_cmd() + [
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        output_template,
        "--print",
        "after_move:filepath",
        url,
    ]
    result = _run_yt_dlp(cmd)

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if output_lines:
        output_path = Path(output_lines[-1])
        if output_path.exists():
            logger.info(f"Audio downloaded: {output_path.name}")
            return output_path

    files = sorted(output_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        logger.info(f"Audio downloaded: {files[0].name}")
        return files[0]

    raise Exception("Download completed but output file not found")
