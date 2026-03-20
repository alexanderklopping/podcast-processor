"""Shared utilities: retry, sanitize, logging, ffmpeg helpers."""

import ipaddress
import logging
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse

from .config import LOGS_DIR, MAX_RETRIES, RETRY_DELAY_SECONDS


def setup_logging():
    """Configure logging to both file and console."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_filename = LOGS_DIR / f"processor_{datetime.now().strftime('%Y-%m-%d')}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger("mediaverwerker")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def retry_with_backoff(max_retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS, backoff_factor=2):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("mediaverwerker")
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")

            raise last_exception

        return wrapper

    return decorator


def sanitize_filename(title):
    """Create a safe filename from a title string."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    return safe[:100].strip()


def extract_urls(text):
    """Extract HTTP(S) URLs from free text."""
    return re.findall(r'https?://[^\s<>"\')]+', text)


def is_url(text):
    """Return True if the input is a standalone HTTP(S) URL."""
    return bool(re.fullmatch(r'https?://[^\s<>"\')]+', text.strip()))


def validate_url(url):
    """Validate a URL for safety. Raises ValueError for invalid or dangerous URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")

    # Block private/loopback IPs (SSRF protection)
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
            raise ValueError(f"URL points to non-public address: {parsed.hostname}")
    except ValueError as e:
        if "non-public" in str(e):
            raise
        # hostname is not an IP literal — resolve it to check
        try:
            resolved = socket.getaddrinfo(parsed.hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for _, _, _, _, addr in resolved:
                ip = ipaddress.ip_address(addr[0])
                if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                    raise ValueError(f"URL hostname {parsed.hostname} resolves to non-public address: {addr[0]}")
        except socket.gaierror:
            pass  # DNS resolution failed — let downstream handle it

    return url


def get_audio_duration(audio_path):
    """Get duration of audio/video file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def split_audio(audio_path, chunk_duration_seconds=600):
    """Split audio file into chunks using ffmpeg."""
    logger = logging.getLogger("mediaverwerker")
    chunks_dir = audio_path.parent / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    total_duration = get_audio_duration(audio_path)
    if total_duration is None:
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        total_duration = file_size_mb * 60

    chunk_paths = []
    chunk_index = 0
    start_time = 0

    while start_time < total_duration:
        chunk_filename = f"{audio_path.stem}_chunk{chunk_index:03d}.mp3"
        chunk_path = chunks_dir / chunk_filename

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start_time),
            "-t",
            str(chunk_duration_seconds),
            "-acodec",
            "libmp3lame",
            "-ab",
            "64k",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(chunk_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunk_paths.append(chunk_path)
                logger.debug(f"Created chunk {chunk_index + 1}: {chunk_filename}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to create chunk {chunk_index}: {e}")

        start_time += chunk_duration_seconds
        chunk_index += 1

    return chunk_paths


def format_timestamp(seconds):
    """Format seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_srt_timestamp(seconds):
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
