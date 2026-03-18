"""Pipeline orchestration and parallel execution."""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import feedparser

from .config import init, validate_environment, IS_CLOUD
from .state import (
    load_processed_episodes, save_processed_episodes,
    load_failed_episodes, save_failed_episode, remove_failed_episode,
    load_podcasts, write_status_file,
)
from .tasks.download import (
    download_episode,
    search_podcast,
    fetch_url_metadata,
    download_url_audio,
)
from .tasks.transcribe import transcribe_audio, save_transcript
from .tasks.article import create_article, save_article
from .tasks.segment import find_segment
from .tasks.clip import clip_media, generate_srt
from .tasks.feeds import (
    update_all_rss_feeds,
    update_individual_rss_feed,
    push_feeds_to_github,
    setup_feeds_repo_for_cloud,
)
from .util import retry_with_backoff, sanitize_filename

logger = logging.getLogger("mediaverwerker")


def _mark_episode_processed(guid):
    """Persist a GUID once without duplicates."""
    processed = load_processed_episodes()
    if guid not in processed:
        processed.append(guid)
        save_processed_episodes(processed)


def _entry_guid(entry):
    """Build a stable GUID from the fields available in a feed entry."""
    return (
        entry.get("id")
        or entry.get("guid")
        or entry.get("link")
        or entry.get("title")
        or entry.get("published")
        or "unknown-entry"
    )


@retry_with_backoff()
def fetch_rss_feed(url):
    """Fetch and parse RSS feed with retry logic."""
    logger.info(f"Fetching RSS feed: {url}")
    feed = feedparser.parse(url)
    if feed.bozo:
        logger.warning(f"Feed parsing issue: {feed.bozo_exception}")
    if not feed.entries:
        raise Exception("No entries found in RSS feed")
    return feed


def get_new_episodes_for_podcast(podcast):
    """Fetch RSS feed and return new episodes for a single podcast (max 2)."""
    feed = fetch_rss_feed(podcast["url"])
    processed = load_processed_episodes()
    new_episodes = []

    for entry in feed.entries[:2]:
        guid = _entry_guid(entry)
        if guid not in processed:
            audio_url = None
            for enclosure in entry.get("enclosures", []):
                if enclosure.get("type", "").startswith("audio/"):
                    audio_url = enclosure.get("href") or enclosure.get("url")
                    break

            if audio_url:
                new_episodes.append({
                    "guid": guid,
                    "title": entry.title,
                    "published": entry.get("published", ""),
                    "audio_url": audio_url,
                    "description": entry.get("summary", ""),
                    "podcast_name": podcast["name"],
                    "language": podcast.get("language", "en"),
                })

    logger.info(f"Found {len(new_episodes)} new episode(s) for {podcast['name']}")
    return new_episodes


def find_episode_by_name_and_date(podcast_name, date_str=None):
    """Find a specific episode by podcast name and optional date.

    Args:
        podcast_name: Name or partial name of the podcast.
        date_str: Date string like "yesterday", "2026-03-12", or None for latest.

    Returns:
        Episode dict or None.
    """
    podcasts = load_podcasts()

    # Fuzzy match podcast name
    matched = None
    for p in podcasts:
        if podcast_name.lower() in p["name"].lower() or p["name"].lower() in podcast_name.lower():
            matched = p
            break

    if not matched:
        # Try broader matching
        for p in podcasts:
            if any(word.lower() in p["name"].lower() for word in podcast_name.split()):
                matched = p
                break

    if not matched:
        logger.error(f"Podcast not found: {podcast_name}")
        return None

    feed = fetch_rss_feed(matched["url"])

    # Parse target date
    from datetime import timedelta
    from dateutil import parser as dateparser
    import time

    target_date = None
    if date_str:
        today = datetime.now().date()
        if date_str.lower() in ("gisteren", "yesterday"):
            target_date = today - timedelta(days=1)
        elif date_str.lower() in ("vandaag", "today"):
            target_date = today
        elif date_str.lower() in ("eergisteren", "day before yesterday"):
            target_date = today - timedelta(days=2)
        else:
            try:
                target_date = dateparser.parse(date_str).date()
            except Exception:
                logger.warning(f"Could not parse date: {date_str}")

    for entry in feed.entries[:10]:
        # Check date match if specified
        if target_date:
            pub_date_str = entry.get("published", "")
            if pub_date_str:
                try:
                    entry_date = dateparser.parse(pub_date_str).date()
                    if entry_date != target_date:
                        continue
                except Exception:
                    pass

        audio_url = None
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href") or enclosure.get("url")
                break

        if audio_url:
            return {
                "guid": _entry_guid(entry),
                "title": entry.title,
                "published": entry.get("published", ""),
                "audio_url": audio_url,
                "description": entry.get("summary", ""),
                "podcast_name": matched["name"],
                "language": matched.get("language", "en"),
            }

    # If no date match, return the latest
    if target_date:
        logger.warning(f"No episode found for date {target_date}, returning latest")
    if feed.entries:
        entry = feed.entries[0]
        audio_url = None
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href") or enclosure.get("url")
                break
        if audio_url:
            return {
                "guid": _entry_guid(entry),
                "title": entry.title,
                "published": entry.get("published", ""),
                "audio_url": audio_url,
                "description": entry.get("summary", ""),
                "podcast_name": matched["name"],
                "language": matched.get("language", "en"),
            }

    return None


def get_all_new_episodes():
    """Fetch new episodes from all configured podcasts."""
    podcasts = load_podcasts()
    all_new = []
    for podcast in podcasts:
        try:
            all_new.extend(get_new_episodes_for_podcast(podcast))
        except Exception as e:
            logger.error(f"Error fetching {podcast['name']}: {e}")
    return all_new


def process_episode(episode):
    """Process a single episode: download, transcribe, create article."""
    podcast_name = episode.get("podcast_name", "unknown")
    language = episode.get("language", "en")
    logger.info(f"Processing [{podcast_name}]: {episode['title']}")

    try:
        audio_path = download_episode(episode)
        transcript = transcribe_audio(audio_path, language)
        save_transcript(episode, transcript)
        article = create_article(episode, transcript)
        save_article(episode, article)

        _mark_episode_processed(episode["guid"])
        remove_failed_episode(episode["guid"])

        logger.info(f"Successfully processed: {episode['title']}")
        return True

    except Exception as e:
        logger.error(f"Error processing {episode['title']}: {e}", exc_info=True)
        save_failed_episode(episode, e)
        return False


def process_and_extract(podcast_name, date=None, topic=None, output_format="transcript"):
    """Download a specific episode, transcribe, and optionally find a topic segment.

    Args:
        podcast_name: Name of the podcast.
        date: Date string (e.g., "yesterday", "2026-03-12").
        topic: Topic to search for in the transcript.
        output_format: "transcript", "article", or "clip".

    Returns:
        dict with results.
    """
    episode = find_episode_by_name_and_date(podcast_name, date)
    if not episode:
        return {"error": f"Episode not found for {podcast_name}"}

    logger.info(f"Found episode: {episode['title']}")
    audio_path = download_episode(episode)

    # Transcribe with timestamps if we need to find a segment
    need_timestamps = topic is not None or output_format == "clip"
    transcript = transcribe_audio(audio_path, episode.get("language", "en"), timestamps=need_timestamps)
    save_transcript(episode, transcript)

    result = {
        "episode": episode,
        "audio_path": str(audio_path),
    }

    if topic and need_timestamps:
        segment = find_segment(transcript["segments"], topic)
        if segment:
            result["segment"] = segment
            result["segment_text"] = segment["text"]
            logger.info(f"\n--- Segment over '{topic}' ---\n{segment['text']}\n---")
        else:
            result["segment"] = None
            result["segment_text"] = None
            logger.info(f"Topic '{topic}' not found in transcript")
    else:
        text = transcript if isinstance(transcript, str) else transcript.get("text", "")
        result["transcript"] = text

    if output_format == "article":
        article = create_article(episode, transcript)
        save_article(episode, article)
        result["article"] = article

    return result


def process_individual_url(url, topic=None, output_format="article", output_dir=None, publish_to_feed=True):
    """Download and process a single media URL into the individual episodes feed."""
    import shutil

    try:
        episode = fetch_url_metadata(url)
        logger.info(f"Processing [individual URL]: {episode['title']}")

        if episode["guid"] in load_processed_episodes():
            logger.info(f"Already processed: {episode['title']}")
            result = {
                "episode": episode,
                "already_processed": True,
            }
            if publish_to_feed:
                result["feed_url"] = "https://alexanderklopping.github.io/podcast-feeds/individuele-afleveringen.xml"
            return result

        audio_path = download_url_audio(url)
        need_timestamps = topic is not None
        transcript = transcribe_audio(audio_path, episode["language"], timestamps=need_timestamps)
        transcript_path = save_transcript(episode, transcript)

        result = {
            "episode": episode,
            "audio_path": str(audio_path),
            "transcript_path": str(transcript_path),
        }

        if topic and need_timestamps:
            segment = find_segment(transcript["segments"], topic)
            if segment:
                result["segment_text"] = segment["text"]
                logger.info(f"\n--- Segment over '{topic}' ---\n{segment['text']}\n---")
            else:
                result["segment_text"] = None
                logger.info(f"Topic '{topic}' niet gevonden in transcript")

        article = None
        if output_format == "article" or publish_to_feed:
            article = create_article(episode, transcript)
            article_path = save_article(episode, article)
            result["article_path"] = str(article_path)
            result["article"] = article

        if output_dir:
            out_dir = Path(output_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            dest_audio = out_dir / audio_path.name
            shutil.copy2(audio_path, dest_audio)

            text = transcript if isinstance(transcript, str) else transcript.get("text", "")
            dest_transcript = out_dir / f"{sanitize_filename(episode['title'])}_transcript.txt"
            with open(dest_transcript, "w", encoding="utf-8") as f:
                f.write(f"# {episode['title']}\n")
                f.write(f"Bron: {episode['podcast_name']}\n")
                f.write(f"Gepubliceerd: {episode['published']}\n")
                f.write(f"URL: {episode['source_url']}\n\n---\n\n")
                f.write(text)

            result["output_dir"] = str(out_dir)
            result["output_files"] = [str(dest_audio), str(dest_transcript)]
            if article:
                dest_article = out_dir / Path(result["article_path"]).name
                shutil.copy2(result["article_path"], dest_article)
                result["output_files"].append(str(dest_article))

        _mark_episode_processed(episode["guid"])
        remove_failed_episode(episode["guid"])

        if publish_to_feed and article is not None:
            if IS_CLOUD and not setup_feeds_repo_for_cloud():
                return {"error": "Failed to setup feeds repository", "episode": episode}
            update_individual_rss_feed()
            push_feeds_to_github()
            result["feed_url"] = "https://alexanderklopping.github.io/podcast-feeds/individuele-afleveringen.xml"

        return result

    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}", exc_info=True)
        if "episode" in locals():
            save_failed_episode(episode, e)
            return {"error": str(e), "episode": episode}
        return {"error": str(e)}


def process_adhoc_episode(podcast_query, date=None, topic=None, output_format="transcript", output_dir=None):
    """Download and process an episode from any podcast (not just configured ones).

    Searches for the podcast via iTunes, finds the episode, downloads and transcribes it.

    Args:
        podcast_query: Podcast name to search for (e.g., "Hard Fork").
        date: Optional date string.
        topic: Optional topic to find in transcript.
        output_format: "transcript" or "article".
        output_dir: Where to save files (e.g., "~/Desktop"). Defaults to AUDIO_DIR.

    Returns:
        dict with results.
    """
    from .config import AUDIO_DIR, TRANSCRIPTS_DIR
    import shutil

    # First check if it's a configured podcast
    podcasts = load_podcasts()
    matched_config = None
    for p in podcasts:
        if podcast_query.lower() in p["name"].lower() or p["name"].lower() in podcast_query.lower():
            matched_config = p
            break

    if matched_config:
        logger.info(f"Found in configured podcasts: {matched_config['name']}")
        podcast = matched_config
    else:
        # Search via iTunes
        podcast = search_podcast(podcast_query)
        if not podcast:
            return {"error": f"Podcast '{podcast_query}' niet gevonden"}

    # Resolve output directory
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Find the episode
    feed = fetch_rss_feed(podcast["url"])

    from datetime import timedelta
    from dateutil import parser as dateparser

    target_date = None
    if date:
        today = datetime.now().date()
        if date.lower() in ("gisteren", "yesterday"):
            target_date = today - timedelta(days=1)
        elif date.lower() in ("vandaag", "today"):
            target_date = today
        elif date.lower() in ("eergisteren", "day before yesterday"):
            target_date = today - timedelta(days=2)
        else:
            try:
                target_date = dateparser.parse(date).date()
            except Exception:
                pass

    # Find matching entry
    selected_entry = None
    for entry in feed.entries[:10]:
        if target_date:
            pub_date_str = entry.get("published", "")
            if pub_date_str:
                try:
                    entry_date = dateparser.parse(pub_date_str).date()
                    if entry_date != target_date:
                        continue
                except Exception:
                    pass
        selected_entry = entry
        break

    if not selected_entry and feed.entries:
        selected_entry = feed.entries[0]

    if not selected_entry:
        return {"error": "Geen aflevering gevonden"}

    # Extract audio URL
    audio_url = None
    for enclosure in selected_entry.get("enclosures", []):
        if enclosure.get("type", "").startswith("audio/"):
            audio_url = enclosure.get("href") or enclosure.get("url")
            break

    # Some feeds use link directly
    if not audio_url:
        for link in selected_entry.get("links", []):
            if link.get("type", "").startswith("audio/"):
                audio_url = link.get("href")
                break

    if not audio_url:
        return {"error": f"Geen audio URL gevonden voor: {selected_entry.title}"}

    episode = {
        "guid": _entry_guid(selected_entry),
        "title": selected_entry.title,
        "published": selected_entry.get("published", ""),
        "audio_url": audio_url,
        "description": selected_entry.get("summary", ""),
        "podcast_name": podcast["name"],
        "language": podcast.get("language", "en"),
    }

    logger.info(f"Episode: {episode['title']}")

    # Download
    audio_path = download_episode(episode)

    # Transcribe
    need_timestamps = topic is not None
    transcript = transcribe_audio(audio_path, episode["language"], timestamps=need_timestamps)
    transcript_path = save_transcript(episode, transcript)

    result = {
        "episode": episode,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
    }

    # Topic search
    if topic and need_timestamps:
        segment = find_segment(transcript["segments"], topic)
        if segment:
            result["segment_text"] = segment["text"]
            logger.info(f"\n--- Segment over '{topic}' ---\n{segment['text']}\n---")
        else:
            result["segment_text"] = None
            logger.info(f"Topic '{topic}' niet gevonden in transcript")

    # Article
    if output_format == "article":
        article = create_article(episode, transcript)
        article_path = save_article(episode, article)
        result["article_path"] = str(article_path)

    # Copy to output dir if specified
    if out_dir:
        text = transcript if isinstance(transcript, str) else transcript.get("text", "")
        # Copy audio
        dest_audio = out_dir / audio_path.name
        shutil.copy2(audio_path, dest_audio)
        logger.info(f"Audio gekopieerd naar: {dest_audio}")

        # Save transcript to output dir
        dest_transcript = out_dir / f"{sanitize_filename(episode['title'])}_transcript.txt"
        with open(dest_transcript, "w", encoding="utf-8") as f:
            f.write(f"# {episode['title']}\n")
            f.write(f"Podcast: {episode['podcast_name']}\n")
            f.write(f"Gepubliceerd: {episode['published']}\n\n---\n\n")
            f.write(text)
        logger.info(f"Transcript opgeslagen: {dest_transcript}")

        result["output_dir"] = str(out_dir)
        result["output_files"] = [str(dest_audio), str(dest_transcript)]

    return result


def batch_process(episodes, max_workers=3):
    """Process multiple episodes in parallel."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_episode, ep): ep for ep in episodes}
        for future in as_completed(futures):
            episode = futures[future]
            try:
                success = future.result()
                results.append({"episode": episode["title"], "success": success})
            except Exception as e:
                logger.error(f"Parallel processing error for {episode['title']}: {e}")
                results.append({"episode": episode["title"], "success": False, "error": str(e)})

    return results


def retry_failed_episodes():
    """Retry previously failed episodes."""
    failed = load_failed_episodes()
    if not failed:
        return

    retryable = {
        guid: info for guid, info in failed.items()
        if info.get("retry_count", 0) < 5
    }

    if not retryable:
        logger.info("No failed episodes to retry")
        return

    logger.info(f"Retrying {len(retryable)} failed episode(s)")

    for guid, info in retryable.items():
        if info.get("source_type") == "individual_url" and info.get("source_url"):
            process_individual_url(info["source_url"])
            continue

        episode = {
            "guid": guid,
            "title": info["title"],
            "audio_url": info["audio_url"],
            "published": info.get("published", ""),
            "description": info.get("description", ""),
            "podcast_name": info.get("podcast_name", "unknown"),
            "language": info.get("language", "en"),
            "feed_storage_key": info.get("feed_storage_key"),
        }
        process_episode(episode)


def run_full_pipeline():
    """Run the complete processing pipeline (equivalent to old main())."""
    init()

    logger.info("=" * 60)
    logger.info(f"Mediaverwerker - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Running in {'CLOUD' if IS_CLOUD else 'LOCAL'} mode")
    logger.info("=" * 60)

    if not validate_environment():
        write_status_file(0, 0, ["Environment validation failed"])
        sys.exit(1)

    if IS_CLOUD:
        if not setup_feeds_repo_for_cloud():
            write_status_file(0, 0, ["Failed to setup feeds repository"])
            sys.exit(1)

    errors = []

    try:
        retry_failed_episodes()

        new_episodes = get_all_new_episodes()
        if not new_episodes:
            logger.info("No new episodes to process.")
            write_status_file(0, 0, [])
            return

        if len(new_episodes) > 2:
            results = batch_process(new_episodes)
            success_count = sum(1 for r in results if r["success"])
            errors = [r["episode"] for r in results if not r["success"]]
        else:
            success_count = 0
            for episode in new_episodes:
                if process_episode(episode):
                    success_count += 1
                else:
                    errors.append(f"Failed: {episode['title']}")

        logger.info("=" * 60)
        logger.info(f"Completed: {success_count}/{len(new_episodes)} episodes processed")
        logger.info("=" * 60)

        update_all_rss_feeds()
        push_feeds_to_github()
        write_status_file(success_count, len(new_episodes), errors)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        write_status_file(0, 0, [str(e)])
        sys.exit(1)
