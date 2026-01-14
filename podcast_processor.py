#!/usr/bin/env python3
"""
Podcast Processor - Cloud Version
Downloads, transcribes, and converts podcast episodes to articles.
Designed to run on Render as a cron job.
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
from openai import OpenAI
from anthropic import Anthropic

# Maximum file size for Whisper API (25MB)
MAX_WHISPER_SIZE = 25 * 1024 * 1024

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "alexanderklopping/podcast-feeds"

# Paths - all in /app for containerized environment
BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
PODCASTS_FILE = BASE_DIR / "podcasts.json"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripten"
ARTICLES_DIR = DATA_DIR / "artikelen"
FEEDS_DIR = DATA_DIR / "feeds"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
ARTICLES_DIR.mkdir(exist_ok=True)
FEEDS_DIR.mkdir(exist_ok=True)


def setup_logging():
    """Configure logging to console only (cloud environment)."""
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False

    # Verify podcasts.json exists
    if not PODCASTS_FILE.exists():
        logger.error(f"Podcasts configuration file not found: {PODCASTS_FILE}")
        return False

    # Verify ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not installed or not in PATH")
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


def setup_github_repo():
    """Clone the podcast-feeds repo to get processed_episodes.json and set up for push."""
    repo_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

    if (FEEDS_DIR / ".git").exists():
        # Already cloned, just pull latest
        logger.info("Pulling latest from podcast-feeds repo...")
        subprocess.run(["git", "pull"], cwd=FEEDS_DIR, capture_output=True)
    else:
        # Clone the repo
        logger.info("Cloning podcast-feeds repo...")
        # Remove existing feeds dir content first
        for f in FEEDS_DIR.iterdir():
            if f.is_file():
                f.unlink()

        subprocess.run(
            ["git", "clone", repo_url, "."],
            cwd=FEEDS_DIR,
            capture_output=True,
            check=True
        )

    # Configure git user for commits
    subprocess.run(["git", "config", "user.email", "bot@podcast-processor.local"], cwd=FEEDS_DIR, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Podcast Processor Bot"], cwd=FEEDS_DIR, capture_output=True)

    logger.info("GitHub repo setup complete")


def load_processed_episodes():
    """Load list of already processed episode GUIDs from the feeds repo."""
    processed_file = FEEDS_DIR / "processed_episodes.json"
    if processed_file.exists():
        try:
            with open(processed_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error reading processed episodes file: {e}")
    return []


def save_processed_episodes(processed):
    """Save list of processed episode GUIDs to the feeds repo."""
    processed_file = FEEDS_DIR / "processed_episodes.json"
    with open(processed_file, "w") as f:
        json.dump(processed, f, indent=2)


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
        headers={"User-Agent": "PodcastProcessor/1.0"}
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
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i + 1}: {e}")
                raise
            finally:
                try:
                    chunk_path.unlink()
                except:
                    pass

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


def save_article(episode, article):
    """Save article as Markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    podcast_name = episode.get("podcast_name", "unknown")
    base_filename = f"{date_str}_{podcast_name}_{sanitize_filename(episode['title'])}"

    md_filepath = ARTICLES_DIR / f"{base_filename}.md"
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(article)
    logger.info(f"Markdown saved: {base_filename}.md")

    return md_filepath


def process_episode(episode):
    """Process a single episode: download, transcribe, create article."""
    podcast_name = episode.get("podcast_name", "unknown")
    language = episode.get("language", "en")
    logger.info(f"Processing [{podcast_name}]: {episode['title']}")

    try:
        audio_path = download_episode(episode)
        transcript = transcribe_audio(audio_path, language)
        article = create_article(episode, transcript)
        save_article(episode, article)

        # Mark as processed
        processed = load_processed_episodes()
        processed.append(episode["guid"])
        save_processed_episodes(processed)

        # Clean up audio file to save space
        try:
            audio_path.unlink()
            logger.info(f"Cleaned up audio file: {audio_path.name}")
        except:
            pass

        logger.info(f"Successfully processed: {episode['title']}")
        return True

    except Exception as e:
        logger.error(f"Error processing {episode['title']}: {e}", exc_info=True)
        return False


def markdown_to_html(markdown_text):
    """Convert markdown to basic HTML."""
    html = markdown_text

    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    paragraphs = html.split('\n\n')
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<h'):
            if not p.startswith('<'):
                p = f'<p>{p}</p>'
        html_parts.append(p)
    html = '\n'.join(html_parts)

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
    text = re.sub(r'^# .+$', '', markdown_text, count=1, flags=re.MULTILINE).strip()
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'^\*[^*]+\*\s*', '', text).strip()

    paragraphs = text.split('\n\n')
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('#') and not p.startswith('*'):
            if len(p) > 300:
                p = p[:297] + '...'
            return p
    return ""


def generate_rss_feed(podcast_name):
    """Generate an RSS feed for a specific podcast."""
    logger.info(f"Generating RSS feed for: {podcast_name}")

    articles = []
    all_md_files = list(ARTICLES_DIR.glob("*.md"))

    for md_file in all_md_files:
        filename = md_file.stem
        if f"_{podcast_name}_" in filename:
            pass
        elif podcast_name == "VSR":
            parts = filename.split('_')
            if len(parts) >= 2:
                known_podcasts = [p['name'] for p in load_podcasts()]
                if parts[1] not in known_podcasts:
                    pass
                else:
                    continue
            else:
                continue
        else:
            continue

        try:
            date_match = re.match(r'^(\d{4}-\d{2}-\d{2})_', filename)
            if date_match:
                date_str = date_match.group(1)
                pub_date = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                pub_date = datetime.fromtimestamp(md_file.stat().st_mtime)

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

    articles.sort(key=lambda x: x['pub_date'], reverse=True)

    rss_items = []
    for article in articles:
        rfc822_date = article['pub_date'].strftime("%a, %d %b %Y %H:%M:%S +0000")
        content_escaped = article['content'].replace(']]>', ']]]]><![CDATA[>')
        description_escaped = article['description'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
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


def push_to_github():
    """Commit and push all changes to GitHub."""
    try:
        subprocess.run(["git", "add", "."], cwd=FEEDS_DIR, check=True, capture_output=True)

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=FEEDS_DIR,
            capture_output=True,
            text=True
        )

        if not result.stdout.strip():
            logger.info("No changes to commit")
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

        logger.info("Changes pushed to GitHub successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False


def main():
    """Main function to process new podcast episodes."""
    logger.info("=" * 60)
    logger.info(f"Podcast Processor (Cloud) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    if not validate_environment():
        sys.exit(1)

    # Setup GitHub repo (clone/pull to get processed_episodes.json)
    try:
        setup_github_repo()
    except Exception as e:
        logger.error(f"Failed to setup GitHub repo: {e}")
        sys.exit(1)

    try:
        new_episodes = get_all_new_episodes()

        if not new_episodes:
            logger.info("No new episodes to process.")
            # Still update feeds in case there are local articles
            update_all_rss_feeds()
            push_to_github()
            return

        success_count = 0
        for episode in new_episodes:
            if process_episode(episode):
                success_count += 1

        logger.info("=" * 60)
        logger.info(f"Completed: {success_count}/{len(new_episodes)} episodes processed")
        logger.info("=" * 60)

        # Update RSS feeds and push everything to GitHub
        update_all_rss_feeds()
        push_to_github()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
