"""RSS feed generation and GitHub publishing."""

import logging
import re
import shutil
import subprocess
from datetime import datetime

from ..config import FEEDS_DIR, ARTICLES_DIR, PROCESSED_FILE, IS_CLOUD, GITHUB_TOKEN
from ..state import load_podcasts

logger = logging.getLogger("mediaverwerker")


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
        if p and not p.startswith('<h') and not p.startswith('<'):
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
    known_podcasts = [p['name'] for p in load_podcasts()]

    for md_file in all_md_files:
        filename = md_file.stem
        if f"_{podcast_name}_" in filename:
            pass
        elif podcast_name == "VSR":
            parts = filename.split('_')
            if len(parts) >= 2 and parts[1] not in known_podcasts:
                pass
            else:
                continue
        else:
            continue

        try:
            date_match = re.match(r'^(\d{4}-\d{2}-\d{2})_', filename)
            if date_match:
                pub_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            else:
                pub_date = datetime.fromtimestamp(md_file.stat().st_mtime)

            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            articles.append({
                'title': extract_title_from_markdown(content),
                'description': extract_description_from_markdown(content),
                'content': markdown_to_html(content),
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
    for podcast in load_podcasts():
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

    if not (FEEDS_DIR / ".git").exists():
        logger.info("Cloning podcast-feeds repository...")
        FEEDS_DIR.mkdir(exist_ok=True)
        repo_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/alexanderklopping/podcast-feeds.git"
        try:
            subprocess.run(["git", "clone", repo_url, str(FEEDS_DIR)], check=True, capture_output=True)
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return False

    feeds_processed_file = FEEDS_DIR / "processed_episodes.json"
    if feeds_processed_file.exists() and not PROCESSED_FILE.exists():
        shutil.copy(feeds_processed_file, PROCESSED_FILE)
        logger.info("Loaded processed_episodes.json from feeds repo")

    return True


def push_feeds_to_github():
    """Commit and push feeds to GitHub repository."""
    try:
        if not (FEEDS_DIR / ".git").exists():
            logger.warning("Feeds directory is not a git repository. Skipping push.")
            return False

        if IS_CLOUD and GITHUB_TOKEN:
            subprocess.run(["git", "config", "user.email", "mediaverwerker@github.com"], cwd=FEEDS_DIR, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Mediaverwerker"], cwd=FEEDS_DIR, check=True, capture_output=True)
            repo_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/alexanderklopping/podcast-feeds.git"
            subprocess.run(["git", "remote", "set-url", "origin", repo_url], cwd=FEEDS_DIR, check=True, capture_output=True)

            if PROCESSED_FILE.exists():
                shutil.copy(PROCESSED_FILE, FEEDS_DIR / "processed_episodes.json")

        subprocess.run(["git", "add", "."], cwd=FEEDS_DIR, check=True, capture_output=True)

        result = subprocess.run(["git", "status", "--porcelain"], cwd=FEEDS_DIR, capture_output=True, text=True)
        if not result.stdout.strip():
            logger.info("No changes to commit in feeds")
            return True

        subprocess.run(
            ["git", "commit", "-m", f"Update feeds - {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
            cwd=FEEDS_DIR, check=True, capture_output=True
        )
        subprocess.run(["git", "push"], cwd=FEEDS_DIR, check=True, capture_output=True)

        logger.info("Feeds pushed to GitHub successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False
