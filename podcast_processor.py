#!/usr/bin/env python3
"""
Podcast Processor - Downloads, transcribes, and converts podcast episodes to articles.
Runs daily at 09:00 to check for new episodes from VSR podcast.
"""

import logging
import os
import json
import subprocess
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import feedparser
import re
import requests
from fpdf import FPDF
from openai import OpenAI
from anthropic import Anthropic
from dotenv import dotenv_values

# Maximum file size for Whisper API (25MB)
MAX_WHISPER_SIZE = 25 * 1024 * 1024

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Paths - define first so we can load .env from correct location
BASE_DIR = Path(__file__).parent.resolve()

# Load environment variables from script directory
_env_config = dotenv_values(BASE_DIR / ".env")

# Configuration - prefer .env file, fall back to environment
OPENAI_API_KEY = _env_config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _env_config.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN = _env_config.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")

# Cloud mode detection
IS_CLOUD = os.getenv("RENDER") is not None or os.getenv("IS_CLOUD") == "true"

# Paths
PODCASTS_FILE = BASE_DIR / "podcasts.json"
AUDIO_DIR = BASE_DIR / "audio"
TRANSCRIPTS_DIR = BASE_DIR / "transcripten"
ARTICLES_DIR = BASE_DIR / "artikelen"
LOGS_DIR = BASE_DIR / "logs"
FEEDS_DIR = BASE_DIR / "feeds"
PROCESSED_FILE = BASE_DIR / "processed_episodes.json"
FAILED_FILE = BASE_DIR / "failed_episodes.json"

# Create directories
AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
ARTICLES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
FEEDS_DIR.mkdir(exist_ok=True)

# Setup logging
def setup_logging():
    """Configure logging to both file and console."""
    log_filename = LOGS_DIR / f"processor_{datetime.now().strftime('%Y-%m-%d')}.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def validate_environment():
    """Validate that all required environment variables are set."""
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please check your .env file")
        return False

    # Verify podcasts.json exists
    if not os.path.exists(PODCASTS_FILE):
        logger.error(f"Podcasts configuration file not found: {PODCASTS_FILE}")
        return False

    # Verify ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not installed or not in PATH")
        logger.error("Install with: brew install ffmpeg")
        return False

    logger.info("Environment validation passed")
    return True


def retry_with_backoff(max_retries=MAX_RETRIES, delay=RETRY_DELAY_SECONDS, backoff_factor=2):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")

            raise last_exception
        return wrapper
    return decorator


def load_processed_episodes():
    """Load list of already processed episode GUIDs."""
    if PROCESSED_FILE.exists():
        try:
            with open(PROCESSED_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error reading processed episodes file: {e}")
            # Backup corrupted file
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
        "retry_count": failed.get(episode["guid"], {}).get("retry_count", 0) + 1
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
    """Fetch RSS feed and return new episodes for a single podcast."""
    feed = fetch_rss_feed(podcast["url"])
    processed = load_processed_episodes()
    new_episodes = []

    for entry in feed.entries:
        guid = entry.get("id", entry.get("guid", entry.link))
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


def get_all_new_episodes():
    """Fetch new episodes from all configured podcasts."""
    podcasts = load_podcasts()
    all_new_episodes = []

    for podcast in podcasts:
        try:
            new_episodes = get_new_episodes_for_podcast(podcast)
            all_new_episodes.extend(new_episodes)
        except Exception as e:
            logger.error(f"Error fetching {podcast['name']}: {e}")

    return all_new_episodes


def sanitize_filename(title):
    """Create a safe filename from episode title."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    return safe[:100].strip()


@retry_with_backoff(max_retries=3, delay=10)
def download_episode(episode):
    """Download podcast audio file with retry logic."""
    filename = sanitize_filename(episode["title"]) + ".mp3"
    filepath = AUDIO_DIR / filename
    temp_filepath = filepath.with_suffix('.mp3.tmp')

    if filepath.exists():
        # Verify file is not corrupted (has reasonable size)
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
        timeout=300,  # 5 minute timeout
        headers={"User-Agent": "PodcastProcessor/1.0"}
    )
    response.raise_for_status()

    # Get expected file size
    expected_size = int(response.headers.get('content-length', 0))

    # Download to temp file first
    downloaded_size = 0
    with open(temp_filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded_size += len(chunk)

    # Verify download
    if expected_size > 0 and downloaded_size < expected_size * 0.95:
        temp_filepath.unlink()
        raise Exception(f"Incomplete download: {downloaded_size}/{expected_size} bytes")

    # Move to final location
    temp_filepath.replace(filepath)
    logger.info(f"Downloaded: {filename} ({downloaded_size / (1024*1024):.1f}MB)")
    return filepath


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
            ],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not get duration: {e}")
        return None


def split_audio(audio_path, chunk_duration_seconds=600):
    """Split audio file into chunks using ffmpeg."""
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
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(chunk_duration_seconds),
            "-acodec", "libmp3lame",
            "-ab", "64k",
            "-ar", "16000",
            "-ac", "1",
            str(chunk_path)
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


@retry_with_backoff(max_retries=3, delay=5)
def transcribe_single_file(client, audio_path, language="en"):
    """Transcribe a single audio file with retry logic."""
    with open(audio_path, "rb") as audio_file:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="text"
        )


def transcribe_audio(audio_path, language="en"):
    """Transcribe audio using OpenAI Whisper API."""
    logger.info(f"Transcribing: {audio_path.name} (language: {language})")

    file_size = audio_path.stat().st_size
    client = OpenAI(api_key=OPENAI_API_KEY)

    if file_size > MAX_WHISPER_SIZE:
        logger.info(f"File size ({file_size / (1024*1024):.1f}MB) exceeds 25MB limit, splitting...")

        chunk_paths = split_audio(audio_path)

        if not chunk_paths:
            raise Exception("Failed to split audio file into chunks")

        transcripts = []
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}...")
            try:
                transcript = transcribe_single_file(client, chunk_path, language)
                transcripts.append(transcript)
                # Rate limiting: wait between API calls
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i + 1}: {e}")
                raise
            finally:
                try:
                    chunk_path.unlink()
                except:
                    pass

        # Clean up chunks directory
        chunks_dir = audio_path.parent / "chunks"
        try:
            if chunks_dir.exists() and not any(chunks_dir.iterdir()):
                chunks_dir.rmdir()
        except:
            pass

        full_transcript = " ".join(transcripts)
        logger.info(f"Combined {len(transcripts)} chunks into full transcript")
    else:
        full_transcript = transcribe_single_file(client, audio_path, language)

    logger.info("Transcription complete")
    return full_transcript


def save_transcript(episode, transcript):
    """Save transcript to file."""
    podcast_name = episode.get("podcast_name", "unknown")
    filename = f"{podcast_name}_{sanitize_filename(episode['title'])}.txt"
    filepath = TRANSCRIPTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {episode['title']}\n")
        f.write(f"Gepubliceerd: {episode['published']}\n\n")
        f.write("---\n\n")
        f.write(transcript)

    logger.info(f"Transcript saved: {filename}")
    return filepath


PODCAST_TO_ARTICLE_SYSTEM_PROMPT = """# Podcast to Article

**Your Role & Mission**

You are my executive assistant helping me transform a long podcast episode into a compelling written chapter for a physical book I'm creating. Think of yourself as a skilled ghostwriter who listens to the entire episode and crafts it into something that reads like a chapter from a classic business biography.

**Style & Voice**

Write like a chapter from a great business biography--think _Shoe Dog_, _The Everything Store_, or _Hatching Twitter_. This means:

- Rich narrative
- Key turning points treated as pivotal scenes
- Quotes woven in to let the protagonists speak for themselves
- The reader should feel like they're watching history unfold, not reading a summary
- Analytical insight layered into the storytelling, not separated from it

**Taal & Stijl: Nederlands**

Schrijf als een Nederlandse native speaker die literaire non-fictie schrijft:

- **Natuurlijk Nederlands**: Schrijf zoals een ervaren Nederlandse auteur—geen vertalingen, geen anglicismen, geen stijve constructies. De tekst moet voelen alsof hij in het Nederlands is bedacht, niet vertaald.
- **Literaire cadans**: Varieer zinslengte bewust. Korte zinnen voor impact. Langere, vloeiende zinnen om de lezer mee te nemen in het verhaal.
- **Actieve stem**: "Zuckerberg verwierp het bod" niet "Het bod werd door Zuckerberg verworpen"
- **Concrete details**: Geen abstracties maar tastbare momenten—de kleur van de Post-it, de stilte in de vergaderzaal, het tikken van vingers op tafel
- **Geen opsmuk**: Vermijd clichés, containerbegrippen en managementjargon. Elk woord moet zijn plek verdienen.

**Length**

As long as the story warrants. Use your judgment based on the episode's length and insight density. Quality and completeness over brevity. This is meant to be a satisfying read, not a skim.

**Step 1: Understand the Arc**

Listen to/read the full episode and identify:

- The central narrative: What is the story being told? What's the dramatic question?
- The key characters: Who are the protagonists, antagonists, and supporting players?
- The turning points: What are the 3-5 moments where everything changed?
- The stakes: What was at risk? What could have gone wrong?

Great business stories have narrative shape--a beginning that sets the stage, rising tension, pivotal decisions, and resolution (or ongoing cliffhanger). Find that shape.

**Step 2: Map the Characters**

Episodes often feature many players, which can be disorienting. Solve this for the reader by:

- Introducing each character clearly on first appearance with a brief identifying detail (role, relationship to the central figure, why they matter)
- Re-anchoring the reader when a character reappears after a gap (e.g., "Sculley--the Pepsi executive Jobs had personally recruited--now faced an impossible choice")
- Keeping the focus on the 3-5 most important figures; mention minor characters only when necessary and don't let them clutter the narrative
- Using consistent identifiers (if you call someone "the young engineer" once, don't switch to "the Stanford grad" later without reason)

The reader should never have to stop and ask "Wait, who is this again?"

**Step 3: Identify and Explain "Blocker" Concepts**

Scan for business, technical, or industry-specific concepts that are essential to understanding the story. These are "blocker" concepts--if the reader doesn't understand them, they'll be lost.

For each blocker concept:

- Explain it in plain language using an analogy or real-world example
- Keep explanations to 1-2 sentences maximum
- Weave these explanations naturally into the narrative the first time the concept appears

The target reader is someone who is generally intelligent and curious--they read business books and follow tech news, but they may not know the specifics of every industry. Think of someone with a liberal arts degree who's interested in how great companies are built.

**Step 4: Harvest the Best Quotes**

Episodes are rich with quotes preserve them:

- Their sharpest analytical insights and observations
- Memorable one-liners or turns of phrase
- Moments where they reveal something surprising or counterintuitive
- Quotes from founders, executives, journalists, biographers
- Historical documents, memos, interviews they reference
- These are gold--they let the protagonists speak for themselves
- Attribute clearly (e.g., "As Jobs later recalled..." or "In a memo to the board, Hastings wrote...")

When cleaning up quotes:

- Remove filler words (um, uh, like, you know)
- Fix grammatical mistakes from natural speech
- Keep the speaker's authentic voice and meaning intact
- Longer quotes are fine if they're powerful--this isn't a tight summary

**Step 5: Build the Narrative**

Structure the piece like a chapter from a biography:

**Opening:** Start with a scene, a tension, or a question that pulls the reader in. Drop them into a pivotal moment, then zoom out to set the stage. Avoid "This episode covers..." framing--just begin the story.

**Middle:** Move through the narrative chronologically or thematically, depending on what serves the story best. Treat major turning points as scenes--slow down, add detail, let the reader feel the weight of the moment. Use quotes to let key players speak at crucial junctures.

**Closing:** End with resonance--what happened next, what it meant, what lesson or question lingers. The reader should close the chapter feeling like they understand something important about business, strategy, or human nature.

**Step 6: Weave It Together**

Combine narrative, analysis, and quotes into one flowing piece that:

- Reads like a chapter from a great business book, not a podcast summary
- Has no section headers, bullet points, or artificial breaks (a line break between major sections is fine)
- Includes a compelling title in the style of a book chapter
- At the beginning, includes a short paragraph capturing the essence of the story and why it matters
- Makes complete sense to someone who has never heard the podcast
- Is designed to be printed--no links or screen-dependent elements
- Balances storytelling with insight: the reader should be both entertained and educated

**Quality Check:**

Before you finish, ask yourself:

1. "Does this read like a chapter from a business book I'd actually want to read?"
2. "Does the opening pull me in immediately, like a great first page?"
3. "Have I preserved the best quotes from both the hosts and the primary sources they cite?"
4. "Do the turning points land with dramatic weight, or did I rush past them?"
5. "Would someone who knows nothing about this company walk away understanding the story and why it matters?"
6. "Can the reader keep track of who's who throughout the piece?"
7. "Would this print beautifully in a physical book?"
8. "Voelt dit als oorspronkelijk Nederlands, geschreven door een native speaker?"

If yes to all, you've succeeded."""


@retry_with_backoff(max_retries=3, delay=5)
def create_article(episode, transcript):
    """Convert transcript to article using Claude with the podcast-to-article skill."""
    logger.info(f"Creating article for: {episode['title']}")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Always write articles in Dutch, regardless of podcast language
    user_prompt = f"""Hier is een podcast aflevering om te transformeren naar een geschreven hoofdstuk:

**Titel:** {episode['title']}
**Gepubliceerd:** {episode['published']}
**Beschrijving:** {episode['description']}

---

**TRANSCRIPT:**

{transcript}

---

Transformeer dit transcript naar een compelling geschreven hoofdstuk volgens de instructies. Schrijf ALTIJD in het Nederlands, ook als het transcript in het Engels is."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=PODCAST_TO_ARTICLE_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    article = message.content[0].text
    logger.info("Article created")
    return article


class ArticlePDF(FPDF):
    """Custom PDF class for article formatting."""

    def __init__(self):
        super().__init__()
        # Use different font paths for cloud vs local
        if IS_CLOUD:
            # In Docker, use built-in fonts (no custom Unicode font)
            self._use_builtin_fonts = True
        else:
            # On macOS, use Arial Unicode for full character support
            self._use_builtin_fonts = False
            try:
                self.add_font('DejaVu', '', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
                self.add_font('DejaVu', 'B', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
                self.add_font('DejaVu', 'I', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
            except Exception:
                self._use_builtin_fonts = True

    def _set_font_safe(self, style='', size=11):
        """Set font with fallback for cloud environment."""
        if self._use_builtin_fonts:
            # Use Helvetica (built-in) - limited Unicode but works everywhere
            self.set_font('Helvetica', style, size)
        else:
            self.set_font('DejaVu', style, size)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self._set_font_safe('', 9)
        self.set_text_color(128)
        self.cell(0, 10, str(self.page_no()), align='C')


def markdown_to_pdf(markdown_text, pdf_path, title):
    """Convert markdown text to a nicely formatted PDF."""
    pdf = ArticlePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(25, 20, 25)

    lines = markdown_text.split('\n')
    first_para = True

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue

        # H1 title
        if line.startswith('# '):
            pdf._set_font_safe('B', 20)
            pdf.set_text_color(0)
            text = line[2:]
            pdf.multi_cell(0, 10, text)
            pdf.ln(5)

        # H2 subtitle
        elif line.startswith('## '):
            pdf.ln(5)
            pdf._set_font_safe('B', 14)
            pdf.set_text_color(50)
            text = line[3:]
            pdf.multi_cell(0, 8, text)
            pdf.ln(3)

        # Italic intro (starts with *)
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            pdf._set_font_safe('I', 11)
            pdf.set_text_color(60)
            text = line.strip('*')
            pdf.multi_cell(0, 6, text)
            pdf.ln(5)
            first_para = True

        # Regular paragraph
        else:
            pdf._set_font_safe('', 11)
            pdf.set_text_color(30)
            # Remove markdown formatting
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', line)  # bold
            text = re.sub(r'\*(.+?)\*', r'\1', text)  # italic
            text = re.sub(r'_(.+?)_', r'\1', text)  # italic underscore
            pdf.multi_cell(0, 6, text)
            first_para = False

    pdf.output(pdf_path)


def save_article(episode, article):
    """Save article as both Markdown and PDF."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    podcast_name = episode.get("podcast_name", "unknown")
    base_filename = f"{date_str}_{podcast_name}_{sanitize_filename(episode['title'])}"

    # Save Markdown
    md_filepath = ARTICLES_DIR / f"{base_filename}.md"
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(article)
    logger.info(f"Markdown saved: {base_filename}.md")

    # Convert to PDF
    try:
        pdf_filepath = ARTICLES_DIR / f"{base_filename}.pdf"
        markdown_to_pdf(article, str(pdf_filepath), episode['title'])
        logger.info(f"PDF saved: {base_filename}.pdf")
    except Exception as e:
        logger.warning(f"Failed to create PDF: {e}")

    return md_filepath


def process_episode(episode):
    """Process a single episode: download, transcribe, create article."""
    podcast_name = episode.get("podcast_name", "unknown")
    language = episode.get("language", "en")
    logger.info(f"Processing [{podcast_name}]: {episode['title']}")

    try:
        # Download audio
        audio_path = download_episode(episode)

        # Transcribe with correct language
        transcript = transcribe_audio(audio_path, language)

        # Save transcript
        save_transcript(episode, transcript)

        # Create and save article
        article = create_article(episode, transcript)
        save_article(episode, article)

        # Mark as processed
        processed = load_processed_episodes()
        processed.append(episode["guid"])
        save_processed_episodes(processed)

        # Remove from failed list if it was there
        remove_failed_episode(episode["guid"])

        logger.info(f"Successfully processed: {episode['title']}")
        return True

    except Exception as e:
        logger.error(f"Error processing {episode['title']}: {e}", exc_info=True)
        save_failed_episode(episode, e)
        return False


def retry_failed_episodes():
    """Retry previously failed episodes."""
    failed = load_failed_episodes()

    if not failed:
        return

    # Only retry episodes that haven't failed too many times
    retryable = {
        guid: info for guid, info in failed.items()
        if info.get("retry_count", 0) < 5
    }

    if not retryable:
        logger.info("No failed episodes to retry")
        return

    logger.info(f"Retrying {len(retryable)} failed episode(s)")

    for guid, info in retryable.items():
        episode = {
            "guid": guid,
            "title": info["title"],
            "audio_url": info["audio_url"],
            "published": "",
            "description": "",
        }
        process_episode(episode)


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


def markdown_to_html(markdown_text):
    """Convert markdown to basic HTML."""
    html = markdown_text

    # Convert headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

    # Convert bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Convert paragraphs (double newlines)
    paragraphs = html.split('\n\n')
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<h'):
            # Check if it's already wrapped in a tag
            if not p.startswith('<'):
                p = f'<p>{p}</p>'
        html_parts.append(p)
    html = '\n'.join(html_parts)

    # Convert single newlines within paragraphs to <br>
    html = re.sub(r'(?<!</p>)\n(?!<)', '<br>\n', html)

    return html


def extract_title_from_markdown(markdown_text):
    """Extract the first H1 title from markdown."""
    match = re.search(r'^# (.+)$', markdown_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Untitled"


def extract_description_from_markdown(markdown_text):
    """Extract the first paragraph after the title as description."""
    # Remove the title line
    text = re.sub(r'^# .+$', '', markdown_text, count=1, flags=re.MULTILINE).strip()
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE).strip()
    # Remove italic intro if present
    text = re.sub(r'^\*[^*]+\*\s*', '', text).strip()

    # Get first paragraph
    paragraphs = text.split('\n\n')
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('#') and not p.startswith('*'):
            # Limit to ~300 chars
            if len(p) > 300:
                p = p[:297] + '...'
            return p
    return ""


def generate_rss_feed(podcast_name):
    """Generate an RSS feed for a specific podcast."""
    logger.info(f"Generating RSS feed for: {podcast_name}")

    # Find all markdown articles for this podcast
    articles = []

    # Get all markdown files
    all_md_files = list(ARTICLES_DIR.glob("*.md"))

    for md_file in all_md_files:
        filename = md_file.stem
        # Check if this file belongs to this podcast
        # Pattern: date_podcastname_title.md OR date_title.md (old VSR format)
        if f"_{podcast_name}_" in filename:
            pass  # Matches new format
        elif podcast_name == "VSR":
            # For VSR, also include old format files (date_title.md without podcast name)
            # These are files that don't have any podcast name prefix
            parts = filename.split('_')
            if len(parts) >= 2:
                # Check if second part is NOT a known podcast name
                known_podcasts = [p['name'] for p in load_podcasts()]
                if parts[1] not in known_podcasts:
                    pass  # This is an old VSR article
                else:
                    continue  # Skip, belongs to another podcast
            else:
                continue
        else:
            continue  # Skip files not matching this podcast
        try:
            # Parse date from filename (format: 2026-01-13_Podcast_Title.md)
            filename = md_file.stem
            date_match = re.match(r'^(\d{4}-\d{2}-\d{2})_', filename)
            if date_match:
                date_str = date_match.group(1)
                pub_date = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                pub_date = datetime.fromtimestamp(md_file.stat().st_mtime)

            # Read content
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            title = extract_title_from_markdown(content)
            description = extract_description_from_markdown(content)
            html_content = markdown_to_html(content)

            articles.append({
                'title': title,
                'description': description,
                'content': html_content,
                'pub_date': pub_date,
                'guid': filename,
            })
        except Exception as e:
            logger.warning(f"Error processing {md_file}: {e}")

    # Sort by date, newest first
    articles.sort(key=lambda x: x['pub_date'], reverse=True)

    # Build RSS XML
    rss_items = []
    for article in articles:
        # Format date for RSS (RFC 822)
        rfc822_date = article['pub_date'].strftime("%a, %d %b %Y %H:%M:%S +0000")

        # Escape content for CDATA
        content_escaped = article['content'].replace(']]>', ']]]]><![CDATA[>')
        description_escaped = article['description'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Escape title for XML
        title_escaped = article['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        item = f"""    <item>
      <title>{title_escaped}</title>
      <link>https://alexanderklopping.github.io/podcast-feeds/{podcast_name}.xml#{article['guid']}</link>
      <description>{description_escaped}</description>
      <pubDate>{rfc822_date}</pubDate>
      <guid isPermaLink="false">{article['guid']}</guid>
      <content:encoded><![CDATA[{content_escaped}]]></content:encoded>
    </item>"""
        rss_items.append(item)

    # Build complete feed
    now_rfc822 = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    feed_url = f"https://alexanderklopping.github.io/podcast-feeds/{podcast_name}.xml"
    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{podcast_name} - Podcast Artikelen</title>
    <link>{feed_url}</link>
    <description>Artikelen gegenereerd van {podcast_name} podcast afleveringen</description>
    <language>nl</language>
    <lastBuildDate>{now_rfc822}</lastBuildDate>
    <atom:link href="{feed_url}" rel="self" type="application/rss+xml"/>
{chr(10).join(rss_items)}
  </channel>
</rss>"""

    # Save feed
    feed_path = FEEDS_DIR / f"{podcast_name}.xml"
    with open(feed_path, 'w', encoding='utf-8') as f:
        f.write(feed)

    logger.info(f"RSS feed saved: {feed_path} ({len(articles)} articles)")
    return feed_path


def update_all_rss_feeds():
    """Update RSS feeds for all configured podcasts."""
    logger.info("Updating all RSS feeds...")

    podcasts = load_podcasts()
    for podcast in podcasts:
        try:
            generate_rss_feed(podcast['name'])
        except Exception as e:
            logger.error(f"Error generating feed for {podcast['name']}: {e}")


def setup_feeds_repo_for_cloud():
    """Clone the feeds repo in cloud environment."""
    if not IS_CLOUD:
        return True

    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set - cannot push to GitHub in cloud mode")
        return False

    # Clone the repo if it doesn't exist
    if not (FEEDS_DIR / ".git").exists():
        logger.info("Cloning podcast-feeds repository...")
        FEEDS_DIR.mkdir(exist_ok=True)

        repo_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/alexanderklopping/podcast-feeds.git"
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(FEEDS_DIR)],
                check=True,
                capture_output=True
            )
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return False

    # Load processed_episodes.json from feeds repo if it exists there
    feeds_processed_file = FEEDS_DIR / "processed_episodes.json"
    if feeds_processed_file.exists() and not PROCESSED_FILE.exists():
        import shutil
        shutil.copy(feeds_processed_file, PROCESSED_FILE)
        logger.info("Loaded processed_episodes.json from feeds repo")

    return True


def push_feeds_to_github():
    """Commit and push feeds to GitHub repository."""
    try:
        # Check if feeds directory is a git repo
        if not (FEEDS_DIR / ".git").exists():
            logger.warning("Feeds directory is not a git repository. Skipping push.")
            return False

        # In cloud mode, configure git and set remote URL with token
        if IS_CLOUD and GITHUB_TOKEN:
            subprocess.run(
                ["git", "config", "user.email", "podcast-processor@render.com"],
                cwd=FEEDS_DIR,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Podcast Processor"],
                cwd=FEEDS_DIR,
                check=True,
                capture_output=True
            )
            # Set remote URL with token for authentication
            repo_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/alexanderklopping/podcast-feeds.git"
            subprocess.run(
                ["git", "remote", "set-url", "origin", repo_url],
                cwd=FEEDS_DIR,
                check=True,
                capture_output=True
            )

            # Copy processed_episodes.json to feeds repo for persistence
            feeds_processed_file = FEEDS_DIR / "processed_episodes.json"
            if PROCESSED_FILE.exists():
                import shutil
                shutil.copy(PROCESSED_FILE, feeds_processed_file)

        # Git add, commit, push
        subprocess.run(
            ["git", "add", "."],
            cwd=FEEDS_DIR,
            check=True,
            capture_output=True
        )

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=FEEDS_DIR,
            capture_output=True,
            text=True
        )

        if not result.stdout.strip():
            logger.info("No changes to commit in feeds")
            return True

        subprocess.run(
            ["git", "commit", "-m", f"Update feeds - {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
            cwd=FEEDS_DIR,
            check=True,
            capture_output=True
        )

        subprocess.run(
            ["git", "push"],
            cwd=FEEDS_DIR,
            check=True,
            capture_output=True
        )

        logger.info("Feeds pushed to GitHub successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False


def main():
    """Main function to process new podcast episodes."""
    logger.info("=" * 60)
    logger.info(f"Podcast Processor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Running in {'CLOUD' if IS_CLOUD else 'LOCAL'} mode")
    logger.info("=" * 60)

    # Validate environment
    if not validate_environment():
        write_status_file(0, 0, ["Environment validation failed"])
        sys.exit(1)

    # Setup feeds repo in cloud mode
    if IS_CLOUD:
        if not setup_feeds_repo_for_cloud():
            write_status_file(0, 0, ["Failed to setup feeds repository"])
            sys.exit(1)

    errors = []

    try:
        # Retry any previously failed episodes first
        retry_failed_episodes()

        # Get new episodes from all podcasts
        new_episodes = get_all_new_episodes()

        if not new_episodes:
            logger.info("No new episodes to process.")
            write_status_file(0, 0, [])
            return

        # Process each new episode
        success_count = 0
        for episode in new_episodes:
            if process_episode(episode):
                success_count += 1
            else:
                errors.append(f"Failed: {episode['title']}")

        logger.info("=" * 60)
        logger.info(f"Completed: {success_count}/{len(new_episodes)} episodes processed")
        logger.info("=" * 60)

        # Update RSS feeds
        update_all_rss_feeds()
        push_feeds_to_github()

        write_status_file(success_count, len(new_episodes), errors)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        write_status_file(0, 0, [str(e)])
        sys.exit(1)


if __name__ == "__main__":
    main()
