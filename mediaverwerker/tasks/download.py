"""Audio and video download tasks."""

import logging
import subprocess
from pathlib import Path

import requests

from ..config import AUDIO_DIR
from ..util import retry_with_backoff, sanitize_filename

logger = logging.getLogger("mediaverwerker")


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
    cmd = [
        "yt-dlp",
        "--print", "filename",
        "-o", "%(title)s.%(ext)s",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    expected_filename = result.stdout.strip()

    # Download
    cmd = [
        "yt-dlp",
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
