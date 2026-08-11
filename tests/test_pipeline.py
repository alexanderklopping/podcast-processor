"""Tests for podcast discovery pipeline behavior."""

import pytest
import typer

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
                    link="https://example.com/newest-episode",
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
    assert episodes[0]["source_url"] == "https://example.com/newest-episode"


def test_full_pipeline_runs_instagram_when_there_are_no_new_podcasts(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "validate_environment", lambda: True)
    monkeypatch.setattr(pipeline, "IS_CLOUD", False)
    monkeypatch.setattr(pipeline, "retry_failed_episodes", lambda: None)
    monkeypatch.setattr(pipeline, "get_all_new_episodes", lambda: [])
    monkeypatch.setattr(
        pipeline,
        "sync_instagram_feeds",
        lambda: calls.append("instagram") or {"processed": 0, "errors": []},
    )
    monkeypatch.setattr(pipeline, "update_all_rss_feeds", lambda: calls.append("feeds"))
    monkeypatch.setattr(pipeline, "push_feeds_to_github", lambda: calls.append("push"))
    monkeypatch.setattr(pipeline, "write_status_file", lambda *_args: None)

    assert pipeline.execute_pipeline_once() == 0
    assert calls == ["instagram", "feeds", "push"]


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


def test_execute_pipeline_returns_failure_for_partial_batch(monkeypatch):
    statuses = []
    monkeypatch.setattr(pipeline, "init", lambda: None)
    monkeypatch.setattr(pipeline, "validate_environment", lambda: True)
    monkeypatch.setattr(pipeline, "IS_CLOUD", False)
    monkeypatch.setattr(pipeline, "retry_failed_episodes", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "get_all_new_episodes",
        lambda: [{"title": "Success"}, {"title": "Failure"}],
    )
    monkeypatch.setattr(
        pipeline,
        "process_episode",
        lambda episode: episode["title"] == "Success",
    )
    monkeypatch.setattr(pipeline, "publish_current_feeds", lambda: None)
    monkeypatch.setattr(pipeline, "update_all_rss_feeds", lambda: None)
    monkeypatch.setattr(pipeline, "push_feeds_to_github", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "sync_instagram_feeds",
        lambda: {"processed": 0, "errors": []},
    )
    monkeypatch.setattr(
        pipeline,
        "write_status_file",
        lambda success, total, errors: statuses.append((success, total, errors)),
    )

    exit_code = pipeline.execute_pipeline_once()

    assert exit_code == 1
    assert statuses == [(1, 2, ["Failed: Failure"])]


def test_dispatch_actions_exits_when_individual_url_processing_fails(monkeypatch):
    monkeypatch.setattr(
        cli,
        "process_individual_url",
        lambda **_kwargs: {"error": "metadata lookup failed"},
    )

    with pytest.raises(typer.Exit) as exc:
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

    monkeypatch.setattr(
        pipeline,
        "fetch_url_metadata",
        lambda _url, return_raw=False: (episode, {"extractor_key": "Youtube"}),
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_youtube_caption_transcript",
        lambda _metadata, _language: {"text": "caption transcript", "segments": []},
    )
    monkeypatch.setattr(
        pipeline,
        "download_url_audio",
        lambda _url: pytest.fail("audio download should be skipped"),
    )
    monkeypatch.setattr(
        pipeline,
        "transcribe_audio",
        lambda *_args, **_kwargs: pytest.fail("transcription should be skipped"),
    )
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


def test_process_individual_url_downloads_resolved_audio_url(monkeypatch):
    episode = {
        "guid": "url:spotifypodcast:abc123",
        "title": "Resolved podcast episode",
        "published": "2026-06-10",
        "audio_url": "https://audio.example.com/resolved.mp3",
        "description": "",
        "podcast_name": "Example Podcast",
        "language": "en",
        "source_type": "individual_url",
        "source_url": "https://open.spotify.com/episode/abc123",
        "feed_storage_key": "individuele-afleveringen",
    }
    downloaded_urls = []

    monkeypatch.setattr(
        pipeline,
        "fetch_url_metadata",
        lambda _url, return_raw=False: (episode, {"extractor_key": "SpotifyPodcast"}),
    )
    monkeypatch.setattr(pipeline, "fetch_youtube_caption_transcript", lambda _metadata, _language: None)
    monkeypatch.setattr(
        pipeline,
        "download_url_audio",
        lambda url: downloaded_urls.append(url) or "/tmp/resolved.mp3",
    )
    monkeypatch.setattr(
        pipeline,
        "transcribe_audio",
        lambda _audio_path, _language, timestamps=False: {"text": "transcript", "segments": []},
    )
    monkeypatch.setattr(pipeline, "load_processed_episodes", lambda: [])
    monkeypatch.setattr(pipeline, "save_transcript", lambda _episode, _transcript: "/tmp/transcript.txt")
    monkeypatch.setattr(pipeline, "_mark_episode_processed", lambda _guid: None)
    monkeypatch.setattr(pipeline, "remove_failed_episode", lambda _guid: None)

    result = pipeline.process_individual_url(
        "https://open.spotify.com/episode/abc123",
        output_format="transcript",
        publish_to_feed=False,
    )

    assert downloaded_urls == ["https://audio.example.com/resolved.mp3"]
    assert result["audio_path"] == "/tmp/resolved.mp3"
