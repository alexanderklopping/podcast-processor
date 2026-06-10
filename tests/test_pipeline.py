"""Tests for podcast discovery pipeline behavior."""

from mediaverwerker import pipeline


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
