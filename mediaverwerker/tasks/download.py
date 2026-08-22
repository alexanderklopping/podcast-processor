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
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import feedparser
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
from ..util import retry_with_backoff, sanitize_filename, validate_url

logger = logging.getLogger("mediaverwerker")
_YTDLP_COOKIES_TEMP_FILE = None
SPOTIFY_OEMBED_URL = "https://open.spotify.com/oembed"
SPOTIFY_EPISODE_RE = re.compile(r"(?:^|/)episode/([^/?#]+)")
SUBSTACK_AUDIO_HOST = "api.substack.com"
SUBSTACK_FEED_AUDIO_RE = re.compile(r"^/feed/podcast/\d+/[^/]+\.mp3$")


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


def _is_substack_feed_audio_url(url):
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == SUBSTACK_AUDIO_HOST
        and bool(SUBSTACK_FEED_AUDIO_RE.match(parsed.path))
    )


def _extract_substack_podcast_url(page_html):
    patterns = (
        r'\\"podcast_url\\":\\"((?:\\\\.|[^"\\])+)\\"',
        r'"podcast_url"\s*:\s*"((?:\\.|[^"\\])+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, page_html)
        if not match:
            continue
        try:
            value = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            continue
        parsed = urlparse(value)
        if (
            parsed.scheme == "https"
            and parsed.netloc == SUBSTACK_AUDIO_HOST
            and parsed.path.startswith("/api/v1/audio/upload/")
        ):
            return value
    return None


def _resolve_substack_audio_url(source_url):
    """Resolve a blocked Substack feed enclosure through its public post page."""
    parsed_source = urlparse(source_url or "")
    if parsed_source.scheme != "https" or not parsed_source.netloc:
        return None
    validate_url(source_url)

    response = requests.get(
        source_url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    response.raise_for_status()
    podcast_url = _extract_substack_podcast_url(response.text)
    if not podcast_url:
        return None

    parsed_audio = urlparse(podcast_url)
    fallback_url = urlunparse(parsed_audio._replace(netloc=parsed_source.netloc))
    validate_url(fallback_url)
    return fallback_url


def _download_response(episode):
    audio_url = episode["audio_url"]
    source_url = episode.get("source_url")
    headers = {
        "User-Agent": "Mediaverwerker/1.0",
        "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.8",
    }
    if source_url:
        headers["Referer"] = source_url

    response = requests.get(audio_url, stream=True, timeout=300, headers=headers)
    if response.status_code != 403 or not source_url or not _is_substack_feed_audio_url(audio_url):
        return response

    fallback_url = _resolve_substack_audio_url(source_url)
    if not fallback_url:
        return response

    response.close()
    logger.info("Substack feed enclosure returned 403; retrying via the publication audio endpoint")
    return requests.get(fallback_url, stream=True, timeout=300, headers=headers)


def _metadata_to_episode(metadata, url):
    source_url = metadata.get("webpage_url") or metadata.get("original_url") or metadata.get("url") or url
    audio_url = metadata.get("audio_url") or source_url
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
        "audio_url": audio_url,
        "description": metadata.get("description") or "",
        "podcast_name": source_name,
        "language": language or "en",
        "source_type": "individual_url",
        "source_url": source_url,
        "feed_storage_key": INDIVIDUAL_FEED_SLUG,
    }


def _is_spotify_episode_url(url):
    parsed = urlparse(url)
    return parsed.netloc.endswith("open.spotify.com") and SPOTIFY_EPISODE_RE.search(parsed.path) is not None


def _spotify_episode_id(url):
    match = SPOTIFY_EPISODE_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def _split_spotify_title(title):
    """Split Spotify oEmbed title into episode title and podcast query."""
    title = (title or "").strip()
    for separator in (" - ", " – ", " — ", " | "):
        if separator in title:
            episode_title, podcast_name = title.rsplit(separator, 1)
            return episode_title.strip(), podcast_name.strip()
    return title, title


def _normalize_match_text(value):
    value = html.unescape(value or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _title_match_score(target_title, candidate_title):
    target = _normalize_match_text(target_title)
    candidate = _normalize_match_text(candidate_title)
    if not target or not candidate:
        return 0.0
    if target == candidate:
        return 1.0
    if target in candidate or candidate in target:
        return 0.95
    return SequenceMatcher(None, target, candidate).ratio()


def _entry_get(entry, key, default=None):
    if hasattr(entry, "get"):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _entry_audio_url(entry):
    for enclosure in _entry_get(entry, "enclosures", []) or []:
        href = enclosure.get("href") if hasattr(enclosure, "get") else getattr(enclosure, "href", None)
        media_type = enclosure.get("type", "") if hasattr(enclosure, "get") else getattr(enclosure, "type", "")
        if href and (not media_type or str(media_type).startswith("audio/")):
            return href

    for link in _entry_get(entry, "links", []) or []:
        rel = link.get("rel") if hasattr(link, "get") else getattr(link, "rel", None)
        href = link.get("href") if hasattr(link, "get") else getattr(link, "href", None)
        media_type = link.get("type", "") if hasattr(link, "get") else getattr(link, "type", "")
        if href and rel == "enclosure" and (not media_type or str(media_type).startswith("audio/")):
            return href

    return None


def _entry_release_date(entry):
    published = _entry_get(entry, "published") or _entry_get(entry, "updated") or ""
    if not published:
        return ""
    try:
        return parsedate_to_datetime(published).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        if re.match(r"^\d{4}-\d{2}-\d{2}", published):
            return published[:10]
        return published


def _search_apple_podcast_episode(query, target_title):
    """Find an exact podcast episode in Apple Search and return its enclosure."""
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "media": "podcast",
                "entity": "podcastEpisode",
                "limit": 20,
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as error:
        logger.warning(f"Apple episode search failed: {error}")
        return None

    best_result = None
    best_score = 0.0
    for result in results:
        if not result.get("episodeUrl"):
            continue
        score = _title_match_score(target_title, result.get("trackName", ""))
        if score > best_score:
            best_result = result
            best_score = score

    if best_result is None or best_score < 0.72:
        return None
    return best_result


def _resolve_spotify_episode_metadata(url):
    """Resolve a Spotify episode page to the original podcast RSS enclosure."""
    episode_id = _spotify_episode_id(url)
    if not episode_id:
        raise YtDlpError("Spotify episode URL does not contain an episode id")

    response = requests.get(
        SPOTIFY_OEMBED_URL,
        params={"url": url},
        timeout=10,
        headers={"User-Agent": "Mediaverwerker/1.0"},
    )
    response.raise_for_status()
    spotify_title = response.json().get("title", "")
    episode_title, podcast_query = _split_spotify_title(spotify_title)
    if not episode_title:
        raise YtDlpError("Spotify oEmbed did not return an episode title")

    podcast = search_podcast(podcast_query) or search_podcast(episode_title)
    if not podcast:
        raise YtDlpError(f"Could not find original podcast feed for Spotify episode: {podcast_query}")

    feed = feedparser.parse(podcast["url"])
    if getattr(feed, "bozo", False):
        logger.warning(f"Spotify resolver feed parsing issue: {feed.bozo_exception}")

    best_entry = None
    best_score = 0.0
    for entry in getattr(feed, "entries", [])[:50]:
        score = _title_match_score(episode_title, _entry_get(entry, "title", ""))
        if score > best_score:
            best_entry = entry
            best_score = score

    if best_entry is None or best_score < 0.72:
        apple_episode = _search_apple_podcast_episode(spotify_title, episode_title)
        if not apple_episode:
            raise YtDlpError(f"Could not match Spotify episode title in podcast RSS feed: {episode_title}")
        return {
            "id": episode_id,
            "extractor_key": "SpotifyPodcast",
            "title": apple_episode.get("trackName", episode_title),
            "description": apple_episode.get("description", ""),
            "channel": apple_episode.get("collectionName", podcast_query),
            "language": "nl" if apple_episode.get("country") == "NLD" else "en",
            "release_date": (apple_episode.get("releaseDate") or "")[:10],
            "webpage_url": url,
            "original_url": url,
            "audio_url": apple_episode["episodeUrl"],
            "resolved_from": "spotify",
        }

    audio_url = _entry_audio_url(best_entry)
    if not audio_url:
        raise YtDlpError(f"Matched Spotify episode has no audio enclosure: {episode_title}")

    return {
        "id": episode_id,
        "extractor_key": "SpotifyPodcast",
        "title": _entry_get(best_entry, "title", episode_title),
        "description": _entry_get(best_entry, "summary", "") or _entry_get(best_entry, "description", ""),
        "channel": podcast.get("name") or podcast_query,
        "language": podcast.get("language") or "en",
        "release_date": _entry_release_date(best_entry),
        "webpage_url": url,
        "original_url": url,
        "audio_url": audio_url,
        "resolved_from": "spotify",
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

    response = _download_response(episode)
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

    if _is_spotify_episode_url(url):
        metadata = _resolve_spotify_episode_metadata(url)
    else:
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
        if not line or line == "WEBVTT" or line.isdigit() or "-->" in line or line.startswith(("Kind:", "Language:")):
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
