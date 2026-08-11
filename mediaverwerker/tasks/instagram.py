"""Recurring Instagram profile discovery and transcript feed publishing."""

import base64
import json
import logging
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import AUDIO_DIR, FEEDS_DIR, INSTAGRAM_COOKIES_B64, INSTAGRAM_FEEDS_FILE
from .article import save_article
from .feeds import generate_rss_feed
from .transcribe import save_transcript, transcribe_audio

logger = logging.getLogger("mediaverwerker")

VIDEO_EXTENSIONS = {"mp4", "m4v", "mov", "webm"}
PROFILE_RE = re.compile(r"^https://www\.instagram\.com/([A-Za-z0-9._]+)/$")
STATUS_FILENAME = "instagram_status.json"
MAX_DISCOVERY_POSTS = 50


class InstagramError(RuntimeError):
    """Raised when Instagram discovery or media processing cannot continue."""


def load_instagram_feeds():
    """Load the version-controlled recurring Instagram feed registry."""
    if not INSTAGRAM_FEEDS_FILE.exists():
        return []
    with open(INSTAGRAM_FEEDS_FILE, "r", encoding="utf-8") as handle:
        feeds = json.load(handle)
    if not isinstance(feeds, list):
        raise InstagramError("instagram_feeds.json must contain a list")
    return feeds


def _status_path():
    return FEEDS_DIR / STATUS_FILENAME


def load_instagram_status():
    """Load persistent discovery and health state from podcast-feeds."""
    path = _status_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InstagramError(f"Invalid {STATUS_FILENAME}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def save_instagram_status(status):
    """Atomically persist Instagram state in the cloned feeds repository."""
    FEEDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _status_path()
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


@contextmanager
def instagram_cookies_file():
    """Materialize the cookie secret only for the duration of a sync."""
    if not INSTAGRAM_COOKIES_B64:
        raise InstagramError("INSTAGRAM_COOKIES_B64 is not configured")
    try:
        cookie_bytes = base64.b64decode(INSTAGRAM_COOKIES_B64, validate=True)
    except Exception as exc:
        raise InstagramError("INSTAGRAM_COOKIES_B64 is not valid base64") from exc

    handle = tempfile.NamedTemporaryFile("wb", delete=False, prefix="instagram-cookies-", suffix=".txt")
    path = Path(handle.name)
    try:
        with handle:
            handle.write(cookie_bytes)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _gallery_dl_base(cookies_path):
    return [
        sys.executable,
        "-m",
        "gallery_dl",
        "--config-ignore",
        "--cookies",
        str(cookies_path),
        "-o",
        "extractor.instagram.include=posts,reels",
        "-o",
        f"extractor.instagram.max-posts={MAX_DISCOVERY_POSTS}",
        "-o",
        "extractor.instagram.sleep-request=6-12",
        "-o",
        "output.jsonl=true",
    ]


def _run_gallery_dl(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "gallery-dl failed").strip()
        raise InstagramError(detail)
    return result.stdout


def _json_objects(output):
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            yield from (item for item in value if isinstance(item, dict))


def _nested(data, *paths):
    for path in paths:
        current = data
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current not in (None, ""):
            return current
    return None


def _is_video(metadata):
    extension = str(metadata.get("extension") or metadata.get("ext") or "").lower()
    media_type = str(metadata.get("media_type") or metadata.get("typename") or metadata.get("type") or "").lower()
    return extension in VIDEO_EXTENSIONS or "video" in media_type or bool(metadata.get("video_url"))


def _post_id(metadata):
    value = _nested(
        metadata,
        ("post_id",),
        ("shortcode",),
        ("post_shortcode",),
        ("id",),
        ("pk",),
    )
    return str(value) if value is not None else None


def _shortcode(metadata, post_id):
    value = metadata.get("shortcode") or metadata.get("post_shortcode")
    return str(value or post_id).split("_", 1)[0]


def _published_at(metadata):
    value = metadata.get("date") or metadata.get("timestamp") or metadata.get("taken_at")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value.replace("Z", "+00:00")
    return datetime.now(timezone.utc).isoformat()


def discover_profile(profile, cookies_path):
    """Return profile metadata and posts grouped across carousel media files."""
    url = profile["url"]
    match = PROFILE_RE.match(url)
    if not match or match.group(1).lower() != profile["username"].lower():
        raise InstagramError(f"Invalid canonical Instagram profile URL: {url}")

    discovery_url = f"{url}posts/"
    output = _run_gallery_dl(_gallery_dl_base(cookies_path) + ["--dump-json", discovery_url])
    grouped = {}
    display_name = None
    for metadata in _json_objects(output):
        post_id = _post_id(metadata)
        if not post_id:
            continue
        shortcode = _shortcode(metadata, post_id)
        display_name = display_name or _nested(
            metadata,
            ("fullname",),
            ("full_name",),
            ("user", "full_name"),
            ("owner", "full_name"),
        )
        post = grouped.setdefault(
            post_id,
            {
                "id": post_id,
                "shortcode": shortcode,
                "permalink": str(metadata.get("post_url") or f"https://www.instagram.com/p/{shortcode}/"),
                "caption": str(metadata.get("description") or metadata.get("caption") or "").strip(),
                "publishedAt": _published_at(metadata),
                "hasVideo": False,
            },
        )
        post["hasVideo"] = post["hasVideo"] or _is_video(metadata)
        if not post["caption"]:
            post["caption"] = str(metadata.get("description") or metadata.get("caption") or "").strip()

    posts = sorted(grouped.values(), key=lambda item: item["publishedAt"], reverse=True)
    return {"displayName": str(display_name).strip() if display_name else None, "posts": posts}


def _download_post_videos(post, cookies_path, target_dir):
    cmd = _gallery_dl_base(cookies_path) + [
        "--destination",
        str(target_dir),
        "--filter",
        "extension in ('mp4', 'm4v', 'mov', 'webm')",
        post["permalink"],
    ]
    _run_gallery_dl(cmd)
    return sorted(
        path for path in target_dir.rglob("*") if path.is_file() and path.suffix.lower().lstrip(".") in VIDEO_EXTENSIONS
    )


def _ffmpeg(args, error_prefix):
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise InstagramError(f"{error_prefix}: {result.stderr.strip()}")


def _concat_file_line(path):
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _extract_audio(video_paths, target_dir):
    audio_parts = []
    for index, video_path in enumerate(video_paths):
        audio_path = target_dir / f"audio-{index:02d}.mp3"
        _ffmpeg(
            ["-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k", str(audio_path)],
            f"Could not extract audio from {video_path.name}",
        )
        audio_parts.append(audio_path)

    if len(audio_parts) == 1:
        return audio_parts[0]

    concat_file = target_dir / "concat.txt"
    concat_file.write_text("".join(_concat_file_line(path) for path in audio_parts), encoding="utf-8")
    combined = target_dir / "combined.mp3"
    _ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(combined)],
        "Could not combine carousel audio",
    )
    return combined


def _title_for_post(post, username):
    caption_line = next((line.strip() for line in post["caption"].splitlines() if line.strip()), "")
    if caption_line:
        return caption_line[:140]
    return f"Instagram-video van @{username} - {post['publishedAt'][:10]}"


def _article_markdown(post, transcript_text, title):
    parts = [f"# {title}"]
    if post["caption"]:
        parts.append(post["caption"])
    parts.extend(["## Transcript", transcript_text])
    return "\n\n".join(parts).strip() + "\n"


def process_instagram_post(profile, post, display_name, cookies_path):
    """Download, transcribe, and persist one Instagram post."""
    username = profile["username"]
    episode = {
        "guid": f"instagram:{username.lower()}:{post['shortcode']}",
        "title": _title_for_post(post, username),
        "published": post["publishedAt"],
        "published_at": post["publishedAt"],
        "description": post["caption"],
        "podcast_name": display_name or f"@{username}",
        "author": f"@{username}",
        "language": profile.get("language", "auto"),
        "source_type": "instagram",
        "source_url": post["permalink"],
        "feed_storage_key": f"instagram-{username.lower()}",
    }

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"instagram-{username}-", dir=AUDIO_DIR) as temp_dir:
        target_dir = Path(temp_dir)
        videos = _download_post_videos(post, cookies_path, target_dir)
        if not videos:
            raise InstagramError(f"No downloadable video found for {post['permalink']}")
        audio_path = _extract_audio(videos, target_dir)
        transcript = transcribe_audio(audio_path, episode["language"])
        text = transcript if isinstance(transcript, str) else transcript.get("text", "")
        text = text.strip()
        if not text:
            raise InstagramError(f"No intelligible speech found for {post['permalink']}")
        save_transcript(episode, text)
        save_article(episode, _article_markdown(post, text, episode["title"]))
    return episode


def _profile_status(status, username):
    return status.setdefault(
        username,
        {
            "displayName": None,
            "feedFilename": f"instagram-{username.lower()}.xml",
            "lastCheckedAt": None,
            "lastSuccessAt": None,
            "lastSeenPostId": None,
            "seenPostIds": [],
            "error": None,
        },
    )


def generate_instagram_rss_feed(profile, profile_status):
    username = profile["username"]
    display_name = profile_status.get("displayName") or f"@{username}"
    return generate_rss_feed(
        display_name,
        feed_storage_key=f"instagram-{username.lower()}",
        feed_filename=f"instagram-{username.lower()}.xml",
        description=f"Transcripties van Instagram-video's van @{username}.",
    )


def _store_seen_posts(profile_status, posts, seen):
    profile_status["seenPostIds"] = [post["id"] for post in posts if post["id"] in seen][-500:]


def _record_cookie_error(profiles, status, message):
    errors = []
    checked_at = datetime.now(timezone.utc).isoformat()
    for profile in profiles:
        username = profile["username"]
        profile_status = _profile_status(status, username)
        profile_status["lastCheckedAt"] = checked_at
        profile_status["error"] = message
        errors.append(f"@{username}: {message}")
    save_instagram_status(status)
    return {"processed": 0, "errors": errors}


def _cleanup_unsubscribed_profiles(profiles, status):
    configured = {profile["username"].lower() for profile in profiles}
    changed = False

    for username in list(status):
        if username.lower() in configured:
            continue
        filename = status[username].get("feedFilename") or f"instagram-{username.lower()}.xml"
        (FEEDS_DIR / filename).unlink(missing_ok=True)
        del status[username]
        changed = True

    for feed_path in FEEDS_DIR.glob("instagram-*.xml"):
        username = feed_path.stem.removeprefix("instagram-").lower()
        if username not in configured:
            feed_path.unlink()
            changed = True

    if changed:
        save_instagram_status(status)


def sync_instagram_feeds():
    """Process the initial latest video and all subsequently discovered videos."""
    profiles = load_instagram_feeds()
    status = load_instagram_status()
    _cleanup_unsubscribed_profiles(profiles, status)
    if not profiles:
        return {"processed": 0, "errors": []}

    processed_count = 0
    errors = []
    try:
        cookie_context = instagram_cookies_file()
        cookies_path = cookie_context.__enter__()
    except Exception as exc:
        return _record_cookie_error(profiles, status, str(exc))

    try:
        for profile in profiles:
            username = profile["username"]
            profile_status = _profile_status(status, username)
            profile_status["lastCheckedAt"] = datetime.now(timezone.utc).isoformat()
            try:
                discovery = discover_profile(profile, cookies_path)
                posts = discovery["posts"]
                display_name = discovery["displayName"] or profile_status.get("displayName") or f"@{username}"
                profile_status["displayName"] = display_name
                seen = set(profile_status.get("seenPostIds") or [])
                first_run = not profile_status.get("lastSeenPostId") and not seen

                if first_run:
                    eligible = next((post for post in posts if post["hasVideo"]), None)
                    candidates = [eligible] if eligible else []
                    seen.update(post["id"] for post in posts if not eligible or post["id"] != eligible["id"])
                else:
                    candidates = [post for post in reversed(posts) if post["hasVideo"] and post["id"] not in seen]
                    seen.update(post["id"] for post in posts if not post["hasVideo"])

                for post in candidates:
                    process_instagram_post(profile, post, display_name, cookies_path)
                    seen.add(post["id"])
                    profile_status["lastSeenPostId"] = post["id"]
                    _store_seen_posts(profile_status, posts, seen)
                    save_instagram_status(status)
                    processed_count += 1

                if posts and not profile_status.get("lastSeenPostId"):
                    profile_status["lastSeenPostId"] = posts[0]["id"]
                _store_seen_posts(profile_status, posts, seen)
                profile_status["lastSuccessAt"] = datetime.now(timezone.utc).isoformat()
                profile_status["error"] = None
                generate_instagram_rss_feed(profile, profile_status)
            except Exception as exc:
                message = str(exc)
                logger.error(f"Instagram @{username}: {message}", exc_info=True)
                profile_status["error"] = message
                errors.append(f"@{username}: {message}")
            finally:
                save_instagram_status(status)
    finally:
        cookie_context.__exit__(None, None, None)

    return {"processed": processed_count, "errors": errors}


def update_instagram_rss_feeds():
    """Regenerate configured Instagram feeds without touching discovery state."""
    status = load_instagram_status()
    for profile in load_instagram_feeds():
        try:
            generate_instagram_rss_feed(profile, _profile_status(status, profile["username"]))
        except Exception as exc:
            logger.error(f"Error generating Instagram feed for @{profile['username']}: {exc}")
