"""State management: processed episodes, failed episodes, podcast config."""

import json
import logging
from datetime import datetime

from .config import PROCESSED_FILE, FAILED_FILE, PODCASTS_FILE, BASE_DIR

logger = logging.getLogger("mediaverwerker")


def load_processed_episodes():
    """Load list of already processed episode GUIDs."""
    if PROCESSED_FILE.exists():
        try:
            with open(PROCESSED_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error reading processed episodes file: {e}")
            backup_path = PROCESSED_FILE.with_suffix('.json.bak')
            PROCESSED_FILE.rename(backup_path)
            logger.info(f"Backed up corrupted file to {backup_path}")
    return []


def save_processed_episodes(processed):
    """Save list of processed episode GUIDs with atomic write."""
    temp_file = PROCESSED_FILE.with_suffix('.json.tmp')
    try:
        with open(temp_file, "w") as f:
            json.dump(processed, f, indent=2)
        temp_file.replace(PROCESSED_FILE)
    except Exception as e:
        logger.error(f"Error saving processed episodes: {e}")
        if temp_file.exists():
            temp_file.unlink()
        raise


def load_failed_episodes():
    """Load list of failed episodes for retry."""
    if FAILED_FILE.exists():
        try:
            with open(FAILED_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_failed_episode(episode, error_msg):
    """Save failed episode for later retry."""
    failed = load_failed_episodes()
    failed[episode["guid"]] = {
        "title": episode["title"],
        "audio_url": episode["audio_url"],
        "error": str(error_msg),
        "failed_at": datetime.now().isoformat(),
        "retry_count": failed.get(episode["guid"], {}).get("retry_count", 0) + 1,
        "source_type": episode.get("source_type"),
        "source_url": episode.get("source_url"),
        "podcast_name": episode.get("podcast_name"),
        "language": episode.get("language"),
        "description": episode.get("description"),
        "published": episode.get("published"),
        "feed_storage_key": episode.get("feed_storage_key"),
    }
    with open(FAILED_FILE, "w") as f:
        json.dump(failed, f, indent=2)
    logger.info(f"Saved failed episode for retry: {episode['title']}")


def remove_failed_episode(guid):
    """Remove episode from failed list after successful processing."""
    failed = load_failed_episodes()
    if guid in failed:
        del failed[guid]
        with open(FAILED_FILE, "w") as f:
            json.dump(failed, f, indent=2)


def load_podcasts():
    """Load podcast configurations from podcasts.json."""
    if PODCASTS_FILE.exists():
        with open(PODCASTS_FILE, "r") as f:
            return json.load(f)
    return []


def write_status_file(success_count, total_count, errors):
    """Write a status file for monitoring."""
    status = {
        "last_run": datetime.now().isoformat(),
        "success_count": success_count,
        "total_count": total_count,
        "errors": errors,
        "status": "ok" if success_count == total_count else "partial" if success_count > 0 else "failed"
    }

    status_file = BASE_DIR / "status.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
