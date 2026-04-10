"""Configuration, paths, and environment loading."""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

# Paths - BASE_DIR is the project root (parent of mediaverwerker package)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Load environment variables from project root
_env_config = dotenv_values(BASE_DIR / ".env")

# API keys - prefer .env file, fall back to environment
OPENAI_API_KEY = _env_config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _env_config.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN = _env_config.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("FEEDS_GITHUB_TOKEN")
YTDLP_COOKIES_FROM_BROWSER = _env_config.get("YTDLP_COOKIES_FROM_BROWSER") or os.getenv("YTDLP_COOKIES_FROM_BROWSER")
YTDLP_COOKIES_FILE = _env_config.get("YTDLP_COOKIES_FILE") or os.getenv("YTDLP_COOKIES_FILE")
YTDLP_IMPERSONATE = _env_config.get("YTDLP_IMPERSONATE") or os.getenv("YTDLP_IMPERSONATE")
YTDLP_REMOTE_COMPONENTS = _env_config.get("YTDLP_REMOTE_COMPONENTS") or os.getenv("YTDLP_REMOTE_COMPONENTS")
GROQ_API_KEY = _env_config.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# Transcription provider: "groq" (default, cheapest) or "openai"
TRANSCRIPTION_PROVIDER = _env_config.get("TRANSCRIPTION_PROVIDER") or os.getenv("TRANSCRIPTION_PROVIDER") or "groq"

# Cloud mode detection
IS_CLOUD = os.getenv("RENDER") is not None or os.getenv("IS_CLOUD") == "true"

# Directories
PODCASTS_FILE = BASE_DIR / "podcasts.json"
AUDIO_DIR = BASE_DIR / "audio"
TRANSCRIPTS_DIR = BASE_DIR / "transcripten"
ARTICLES_DIR = BASE_DIR / "artikelen"
LOGS_DIR = BASE_DIR / "logs"
FEEDS_DIR = BASE_DIR / "feeds"
PROCESSED_FILE = BASE_DIR / "processed_episodes.json"
FAILED_FILE = BASE_DIR / "failed_episodes.json"
INDIVIDUAL_FEED_NAME = "Individuele Afleveringen"
INDIVIDUAL_FEED_SLUG = "individuele-afleveringen"

# Whisper API limits (Groq paid tier supports 100MB, OpenAI 25MB)
MAX_WHISPER_SIZE = 100 * 1024 * 1024 if TRANSCRIPTION_PROVIDER == "groq" else 25 * 1024 * 1024

# Retry defaults
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def init():
    """Create required directories."""
    for d in [AUDIO_DIR, TRANSCRIPTS_DIR, ARTICLES_DIR, LOGS_DIR, FEEDS_DIR]:
        d.mkdir(exist_ok=True)


def validate_environment():
    """Validate that all required environment variables and tools are set."""
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    missing = []

    if TRANSCRIPTION_PROVIDER == "groq" and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if TRANSCRIPTION_PROVIDER == "openai" and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False

    if not PODCASTS_FILE.exists():
        logger.error(f"Podcasts configuration file not found: {PODCASTS_FILE}")
        return False

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not installed or not in PATH")
        return False

    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("yt_dlp is not installed in the active Python environment")
        return False

    logger.info("Environment validation passed")
    return True
