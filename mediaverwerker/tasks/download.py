"""Audio and video download tasks."""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..config import (
    AUDIO_DIR,
    INDIVIDUAL_FEED_SLUG,
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_FROM_BROWSER,
    YTDLP_IMPERSONATE,
    YTDLP_REMOTE_COMPONENTS,
)
from ..util import retry_with_backoff, sanitize_filename

logger = logging.getLogger("mediaverwerker")


def _yt_dlp_cmd():
    """Build a yt-dlp command using the active Python environment."""
    cmd = [sys.executable, "-m", "yt_dlp"]
    if YTDLP_REMOTE_COMPONENTS:
        cmd.extend(["--remote-components", YTDLP_REMOTE_COMPONENTS])
    if YTDLP_IMPERSONATE:
        cmd.extend(["--impersonate", YTDLP_IMPERSONATE])
    if YTDLP_COOKIES_FROM_BROWSER:
        cmd.extend(["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER])
    elif YTDLP_COOKIES_FILE:
        cmd.extend(["--cookies", YTDLP_COOKIES_FILE])
    return cmd


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
    temp_filepath = filepath.with_suffix('.mp3.tmp')

    if filepath.exists():
        if filepath.stat().st_size > 1000:
            logger.info(f"Audio already exists: {filename}")
            return filepath
        else:
            logger.warning(f"Existing file too small, re-downloading: {filename}")
            filepath.unlink()

    logger.info(f"Downloading: {episode['title']}")

    response = requests.get(
        episode["audio_url"],
        stream=True,
        timeout=300,
        headers={"User-Agent": "Mediaverwerker/1.0"}
    )
    response.raise_for_status()

    expected_size = int(response.headers.get('content-length', 0))

    downloaded_size = 0
    with open(temp_filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded_size += len(chunk)

    if expected_size > 0 and downloaded_size < expected_size * 0.95:
        temp_filepath.unlink()
        raise Exception(f"Incomplete download: {downloaded_size}/{expected_size} bytes")

    temp_filepath.replace(filepath)
    logger.info(f"Downloaded: {filename} ({downloaded_size / (1024*1024):.1f}MB)")
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
        "--print", "filename",
        "-o", "%(title)s.%(ext)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    expected_filename = result.stdout.strip()

    # Download
    cmd = _yt_dlp_cmd() + [
        "-o", str(output_dir / "%(title)s.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    output_path = output_dir / expected_filename
    if output_path.exists():
        logger.info(f"Video downloaded: {output_path.name}")
        return output_path

    # Fallback: find the most recent file in output_dir
    files = sorted(output_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        logger.info(f"Video downloaded: {files[0].name}")
        return files[0]

    raise Exception(f"Download completed but output file not found")


def fetch_url_metadata(url):
    """Fetch metadata for a single media URL via yt-dlp."""
    logger.info(f"Fetching URL metadata: {url}")

    cmd = _yt_dlp_cmd() + [
        "--dump-single-json",
        "--no-download",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)

    source_url = (
        metadata.get("webpage_url")
        or metadata.get("original_url")
        or metadata.get("url")
        or url
    )
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

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
