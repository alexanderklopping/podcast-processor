"""Tests for mediaverwerker.tasks.feeds helper functions."""

from mediaverwerker.tasks.feeds import (
    extract_description_from_markdown,
    extract_embedded_metadata,
    extract_title_from_markdown,
    generate_rss_feed,
    markdown_to_html,
    strip_embedded_metadata,
)

SAMPLE_METADATA = """<!--
title: Test Episode
podcast: My Podcast
source_url: https://example.com
-->

# The Episode Title

Some content here.
"""


class TestStripEmbeddedMetadata:
    def test_strips_comment_block(self):
        result = strip_embedded_metadata(SAMPLE_METADATA)
        assert "<!--" not in result
        assert "# The Episode Title" in result

    def test_no_metadata(self):
        text = "# Just a title\n\nSome text."
        assert strip_embedded_metadata(text) == text


class TestExtractEmbeddedMetadata:
    def test_parses_fields(self):
        meta = extract_embedded_metadata(SAMPLE_METADATA)
        assert meta["title"] == "Test Episode"
        assert meta["podcast"] == "My Podcast"
        assert meta["source_url"] == "https://example.com"

    def test_no_metadata(self):
        assert extract_embedded_metadata("# Title\nContent") == {}


class TestExtractTitle:
    def test_extracts_h1(self):
        assert extract_title_from_markdown(SAMPLE_METADATA) == "The Episode Title"

    def test_no_title(self):
        assert extract_title_from_markdown("No heading here") == "Untitled"

    def test_skips_metadata(self):
        md = "<!--\ntitle: Meta Title\n-->\n# Actual Title"
        assert extract_title_from_markdown(md) == "Actual Title"


class TestExtractDescription:
    def test_extracts_first_paragraph(self):
        desc = extract_description_from_markdown(SAMPLE_METADATA)
        assert desc == "Some content here."

    def test_truncates_long(self):
        md = "# Title\n\n" + "A" * 400
        desc = extract_description_from_markdown(md)
        assert len(desc) <= 300
        assert desc.endswith("...")


class TestMarkdownToHtml:
    def test_h1(self):
        assert "<h1>Title</h1>" in markdown_to_html("# Title")

    def test_h2(self):
        assert "<h2>Section</h2>" in markdown_to_html("## Section")

    def test_bold(self):
        assert "<strong>bold</strong>" in markdown_to_html("**bold**")

    def test_italic(self):
        assert "<em>italic</em>" in markdown_to_html("*italic*")

    def test_paragraph_wrapping(self):
        html = markdown_to_html("Some text\n\nMore text")
        assert "<p>Some text</p>" in html
        assert "<p>More text</p>" in html


def test_generate_rss_feed_preserves_existing_feed_when_no_articles(tmp_path, monkeypatch):
    """Do not wipe a published feed just because this run processed another podcast."""
    from mediaverwerker.tasks import feeds

    articles_dir = tmp_path / "articles"
    feeds_dir = tmp_path / "feeds"
    articles_dir.mkdir()
    feeds_dir.mkdir()

    existing_feed = feeds_dir / "VSR.xml"
    existing_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>VSR - Podcast Artikelen</title>
    <item>
      <title>Existing VSR episode</title>
      <guid isPermaLink="false">existing-vsr</guid>
    </item>
  </channel>
</rss>"""
    existing_feed.write_text(existing_xml, encoding="utf-8")

    monkeypatch.setattr(feeds, "ARTICLES_DIR", articles_dir)
    monkeypatch.setattr(feeds, "FEEDS_DIR", feeds_dir)
    monkeypatch.setattr(feeds, "load_podcasts", lambda: [{"name": "VSR"}])

    assert generate_rss_feed("VSR") == existing_feed
    assert existing_feed.read_text(encoding="utf-8") == existing_xml
