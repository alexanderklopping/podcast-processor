"""Tests for podcast discovery pipeline behavior."""

from click.exceptions import Exit
import pytest

from mediaverwerker import cli, pipeline


class Entry(dict):
    """Small feedparser-like entry for tests."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_get_new_episodes_limits_to_latest_per_podcast(monkeypatch):
    feed = type(
        "Feed",
        (),
        {
            "entries": [
                Entry(
                    id="episode-1",
                    title="Newest episode",
                    published="2026-06-10",
                    summary="Latest",
                    enclosures=[{"type": "audio/mpeg", "href": "https://example.com/1.mp3"}],
                ),
                Entry(
                    id="episode-2",
                    title="Older episode",
                    published="2026-06-09",
                    summary="Older",
                    enclosures=[{"type": "audio/mpeg", "href": "https://example.com/2.mp3"}],
                ),
            ]
        },
    )()
    monkeypatch.setattr(pipeline, "fetch_rss_feed", lambda _url: feed)
    monkeypatch.setattr(pipeline, "load_processed_episodes", lambda: [])

    episodes = pipeline.get_new_episodes_for_podcast(
        {"name": "VSR", "url": "https://example.com/feed.xml", "language": "nl"}
    )

    assert [episode["title"] for episode in episodes] == ["Newest episode"]


def test_batch_process_publishes_after_each_success(monkeypatch):
    calls = []

    def fake_process_episode(episode):
        return episode["title"] != "Failed episode"

    monkeypatch.setattr(pipeline, "process_episode", fake_process_episode)
    monkeypatch.setattr(pipeline, "update_all_rss_feeds", lambda: calls.append("update"))
    monkeypatch.setattr(pipeline, "push_feeds_to_github", lambda: calls.append("push"))

    results = pipeline.batch_process(
        [
            {"title": "First success"},
            {"title": "Failed episode"},
            {"title": "Second success"},
        ],
        max_workers=1,
    )

    assert results == [
        {"episode": "First success", "success": True},
        {"episode": "Failed episode", "success": False},
        {"episode": "Second success", "success": True},
    ]
    assert calls == ["update", "push", "update", "push"]


def test_dispatch_actions_exits_when_individual_url_processing_fails(monkeypatch):
    monkeypatch.setattr(
        cli,
        "process_individual_url",
        lambda **_kwargs: {"error": "metadata lookup failed"},
    )

    with pytest.raises(Exit) as exc:
        cli._dispatch_actions(
            {
                "actions": [
                    {
                        "type": "process_url",
                        "url": "https://www.youtube.com/watch?v=broken",
                        "output": "article",
                    }
                ]
            }
        )

    assert exc.value.exit_code == 1


def test_process_individual_url_uses_youtube_captions_without_audio_download(monkeypatch):
    episode = {
        "guid": "url:youtube:abc123",
        "title": "Captioned video",
        "published": "2026-06-10",
        "audio_url": "https://www.youtube.com/watch?v=abc123",
        "description": "",
        "podcast_name": "YouTube",
        "language": "en",
        "source_type": "individual_url",
        "source_url": "https://www.youtube.com/watch?v=abc123",
        "feed_storage_key": "individuele-afleveringen",
    }

    monkeypatch.setattr(pipeline, "fetch_url_metadata", lambda _url, return_raw=False: (episode, {"extractor_key": "Youtube"}))
    monkeypatch.setattr(
        pipeline,
        "fetch_youtube_caption_transcript",
        lambda _metadata, _language: {"text": "caption transcript", "segments": []},
    )
    monkeypatch.setattr(pipeline, "download_url_audio", lambda _url: pytest.fail("audio download should be skipped"))
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *_args, **_kwargs: pytest.fail("transcription should be skipped"))
    monkeypatch.setattr(pipeline, "load_processed_episodes", lambda: [])
    monkeypatch.setattr(pipeline, "save_transcript", lambda _episode, _transcript: "/tmp/transcript.txt")
    monkeypatch.setattr(pipeline, "create_article", lambda _episode, _transcript: "# Article")
    monkeypatch.setattr(pipeline, "save_article", lambda _episode, _article: "/tmp/article.md")
    monkeypatch.setattr(pipeline, "_mark_episode_processed", lambda _guid: None)
    monkeypatch.setattr(pipeline, "remove_failed_episode", lambda _guid: None)

    result = pipeline.process_individual_url(
        "https://www.youtube.com/watch?v=abc123",
        publish_to_feed=False,
    )

    assert result["transcript_path"] == "/tmp/transcript.txt"
    assert "audio_path" not in result
