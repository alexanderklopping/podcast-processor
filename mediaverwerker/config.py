"""Configuration, paths, and environment loading."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

# Paths - BASE_DIR is the project root (parent of mediaverwerker package)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Load environment variables from project root
_env_config = dotenv_values(BASE_DIR / ".env")

# Cloud mode detection
IS_CLOUD = os.getenv("RENDER") is not None or os.getenv("IS_CLOUD") == "true"

_ONEPASSWORD_NOTES = []


def _first_nonempty(*values):
    for value in values:
        if value:
            return value
    return None


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _add_1password_note(message):
    if message not in _ONEPASSWORD_NOTES:
        _ONEPASSWORD_NOTES.append(message)


def _1password_enabled():
    disable_flag = _first_nonempty(
        _env_config.get("MEDIAVERWERKER_DISABLE_1PASSWORD"),
        os.getenv("MEDIAVERWERKER_DISABLE_1PASSWORD"),
    )
    return not IS_CLOUD and not _is_truthy(disable_flag)


def _1password_vault():
    return _first_nonempty(
        _env_config.get("MEDIAVERWERKER_1PASSWORD_VAULT"),
        os.getenv("MEDIAVERWERKER_1PASSWORD_VAULT"),
        "Private",
    )


def _1password_item():
    return _first_nonempty(
        _env_config.get("MEDIAVERWERKER_1PASSWORD_ITEM"),
        os.getenv("MEDIAVERWERKER_1PASSWORD_ITEM"),
        "podcast-processor",
    )


def _find_op_binary():
    configured = _first_nonempty(_env_config.get("OP_BIN"), os.getenv("OP_BIN"))
    if configured:
        return configured if Path(configured).exists() else shutil.which(configured)

    if shutil.which("op"):
        return shutil.which("op")

    fallback = Path("/opt/homebrew/bin/op")
    if fallback.exists():
        return str(fallback)

    return None


def _1password_reference(secret_name):
    override_key = f"MEDIAVERWERKER_1PASSWORD_{secret_name}_REF"
    return _first_nonempty(
        _env_config.get(override_key),
        os.getenv(override_key),
        f"op://{_1password_vault()}/{_1password_item()}/{secret_name}",
    )


def _read_1password_secret(secret_name):
    if not _1password_enabled():
        return None

    op_binary = _find_op_binary()
    if not op_binary:
        _add_1password_note("1Password CLI not found. Install `op` to load local project secrets.")
        return None

    reference = _1password_reference(secret_name)
    try:
        result = subprocess.run(
            [op_binary, "read", reference],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or "unable to read secret from 1Password"
        _add_1password_note(f"{secret_name}: {detail}")
        return None

    value = result.stdout.strip()
    if not value:
        _add_1password_note(f"{secret_name}: empty value returned from 1Password")
        return None

    os.environ[secret_name] = value
    return value


def _resolve_secret(secret_name, *, use_1password=True):
    value = _first_nonempty(_env_config.get(secret_name), os.getenv(secret_name))
    if value:
        return value

    if use_1password:
        return _read_1password_secret(secret_name)

    return None


# API keys - prefer .env file, fall back to environment
TRANSCRIPTION_PROVIDER = _first_nonempty(
    _env_config.get("TRANSCRIPTION_PROVIDER"),
    os.getenv("TRANSCRIPTION_PROVIDER"),
    "groq",
)

OPENAI_API_KEY = _resolve_secret(
    "OPENAI_API_KEY",
    use_1password=TRANSCRIPTION_PROVIDER == "openai",
)
ANTHROPIC_API_KEY = _resolve_secret("ANTHROPIC_API_KEY")
GITHUB_TOKEN = _first_nonempty(
    _env_config.get("GITHUB_TOKEN"),
    os.getenv("GITHUB_TOKEN"),
    os.getenv("FEEDS_GITHUB_TOKEN"),
)
YTDLP_COOKIES_FROM_BROWSER = _env_config.get("YTDLP_COOKIES_FROM_BROWSER") or os.getenv("YTDLP_COOKIES_FROM_BROWSER")
YTDLP_COOKIES_FILE = _env_config.get("YTDLP_COOKIES_FILE") or os.getenv("YTDLP_COOKIES_FILE")
YTDLP_IMPERSONATE = _env_config.get("YTDLP_IMPERSONATE") or os.getenv("YTDLP_IMPERSONATE")
YTDLP_REMOTE_COMPONENTS = _env_config.get("YTDLP_REMOTE_COMPONENTS") or os.getenv("YTDLP_REMOTE_COMPONENTS")
GROQ_API_KEY = _resolve_secret("GROQ_API_KEY", use_1password=TRANSCRIPTION_PROVIDER == "groq")

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
        if _1password_enabled():
            for note in _ONEPASSWORD_NOTES:
                logger.info(f"1Password: {note}")
            logger.info(
                "Local secret lookup expects a 1Password item named "
                f"'{_1password_item()}' in vault '{_1password_vault()}', with field names matching the env vars."
            )
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
