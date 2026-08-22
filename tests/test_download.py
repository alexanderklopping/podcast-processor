"""Tests for yt-dlp command construction."""

import subprocess
from pathlib import Path

import pytest

from mediaverwerker.tasks import download


class Entry(dict):
    """Small feedparser-like entry for tests."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_yt_dlp_cmd_includes_configured_js_runtime_and_remote_components(monkeypatch):
    monkeypatch.setattr(download, "YTDLP_JS_RUNTIMES", "node", raising=False)
    monkeypatch.setattr(download, "YTDLP_REMOTE_COMPONENTS", "ejs:github")
    monkeypatch.setattr(download, "YTDLP_EXTRACTOR_ARGS", "youtube:player_skip=webpage;player_client=tv_embedded")
    monkeypatch.setattr(download, "YTDLP_POT_SERVER_HOME", None)
    monkeypatch.setattr(download, "YTDLP_IMPERSONATE", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_FROM_BROWSER", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_FILE", None)

    cmd = download._yt_dlp_cmd()

    assert "--js-runtimes" in cmd
    assert cmd[cmd.index("--js-runtimes") + 1] == "node"
    assert "--remote-components" in cmd
    assert cmd[cmd.index("--remote-components") + 1] == "ejs:github"
    assert "--extractor-args" in cmd
    assert cmd[cmd.index("--extractor-args") + 1] == "youtube:player_skip=webpage;player_client=tv_embedded"


def test_yt_dlp_cmd_uses_cookie_secret_without_youtube_workaround(monkeypatch):
    monkeypatch.setattr(download, "YTDLP_JS_RUNTIMES", "node", raising=False)
    monkeypatch.setattr(download, "YTDLP_REMOTE_COMPONENTS", "ejs:github")
    monkeypatch.setattr(download, "YTDLP_EXTRACTOR_ARGS", "youtube:player_skip=webpage;player_client=tv_embedded")
    monkeypatch.setattr(download, "YTDLP_POT_SERVER_HOME", None)
    monkeypatch.setattr(download, "YTDLP_IMPERSONATE", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_FROM_BROWSER", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_FILE", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_B64", "I05ldHNjYXBlIENvb2tpZSBGaWxlCg==")
    monkeypatch.setattr(download, "_YTDLP_COOKIES_TEMP_FILE", None)

    cmd = download._yt_dlp_cmd()

    assert "--cookies" in cmd
    cookies_path = Path(cmd[cmd.index("--cookies") + 1])
    assert cookies_path.read_text(encoding="utf-8") == "#Netscape Cookie File\n"
    assert "--extractor-args" not in cmd


def test_yt_dlp_cmd_uses_po_token_provider_with_cookie_secret(monkeypatch):
    monkeypatch.setattr(download, "YTDLP_JS_RUNTIMES", "node", raising=False)
    monkeypatch.setattr(download, "YTDLP_REMOTE_COMPONENTS", "ejs:github")
    monkeypatch.setattr(download, "YTDLP_EXTRACTOR_ARGS", None)
    monkeypatch.setattr(download, "YTDLP_POT_SERVER_HOME", "/runner/bgutil/server")
    monkeypatch.setattr(download, "YTDLP_IMPERSONATE", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_FROM_BROWSER", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_FILE", None)
    monkeypatch.setattr(download, "YTDLP_COOKIES_B64", "I05ldHNjYXBlIENvb2tpZSBGaWxlCg==")
    monkeypatch.setattr(download, "_YTDLP_COOKIES_TEMP_FILE", None)

    cmd = download._yt_dlp_cmd()

    assert "--cookies" in cmd
    assert "--extractor-args" in cmd
    assert cmd[cmd.index("--extractor-args") + 1] == ("youtubepot-bgutilscript:server_home=/runner/bgutil/server")


def test_run_yt_dlp_error_includes_stderr(monkeypatch):
    def fake_run(_cmd, capture_output, text, env):
        assert capture_output is True
        assert text is True
        assert "OPENAI_API_KEY" not in env
        assert "BLOB_READ_WRITE_TOKEN" not in env
        return subprocess.CompletedProcess(
            args=["yt-dlp"],
            returncode=1,
            stdout="",
            stderr="ERROR: Sign in to confirm you are not a bot",
        )

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    with pytest.raises(download.YtDlpError) as exc:
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
        monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "must-not-leak")
        download._run_yt_dlp(["yt-dlp", "--dump-single-json", "https://example.com/video"])

    assert "Sign in to confirm" in str(exc.value)


def test_extract_substack_podcast_url_from_escaped_page_state():
    page_html = (
        r"{\"post\":{\"podcast_url\":"
        r"\"https://api.substack.com/api/v1/audio/upload/audio-id/src\"}}"
    )

    assert (
        download._extract_substack_podcast_url(page_html) == "https://api.substack.com/api/v1/audio/upload/audio-id/src"
    )


def test_substack_403_retries_via_publication_audio_endpoint(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.closed = False

        def close(self):
            self.closed = True

    blocked = FakeResponse(403)
    success = FakeResponse(200)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return blocked if len(calls) == 1 else success

    monkeypatch.setattr(download.requests, "get", fake_get)
    monkeypatch.setattr(
        download,
        "_resolve_substack_audio_url",
        lambda source_url: "https://www.dwarkesh.com/api/v1/audio/upload/audio-id/src",
    )

    response = download._download_response(
        {
            "audio_url": "https://api.substack.com/feed/podcast/123/audio.mp3",
            "source_url": "https://www.dwarkesh.com/p/example",
        }
    )

    assert response is success
    assert blocked.closed is True
    assert [call[0] for call in calls] == [
        "https://api.substack.com/feed/podcast/123/audio.mp3",
        "https://www.dwarkesh.com/api/v1/audio/upload/audio-id/src",
    ]
    assert calls[1][1]["headers"]["Referer"] == "https://www.dwarkesh.com/p/example"


def test_fetch_url_metadata_allows_caption_only_youtube_metadata(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"id":"abc123","extractor_key":"Youtube","title":"A video","webpage_url":"https://youtube.com/watch?v=abc123"}',
            stderr="",
        )

    monkeypatch.setattr(download, "_run_yt_dlp", fake_run)

    episode, metadata = download.fetch_url_metadata("https://youtube.com/watch?v=abc123", return_raw=True)

    assert "--ignore-no-formats-error" in calls[0]
    assert episode["guid"] == "url:youtube:abc123"
    assert metadata["title"] == "A video"


def test_fetch_url_metadata_resolves_spotify_episode_to_original_feed_audio(monkeypatch):
    spotify_url = "https://open.spotify.com/episode/spotify-episode-id?si=abc"

    class FakeOembedResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"title": "NPR News: 06-10-2026 12PM EDT - NPR News Now"}

    monkeypatch.setattr(
        download.requests,
        "get",
        lambda url, **_kwargs: FakeOembedResponse() if url == "https://open.spotify.com/oembed" else None,
    )
    monkeypatch.setattr(
        download,
        "search_podcast",
        lambda query: {"name": query, "url": "https://feeds.example.com/npr-news-now.xml", "language": "en"},
    )
    monkeypatch.setattr(
        download.feedparser,
        "parse",
        lambda _url: type(
            "Feed",
            (),
            {
                "entries": [
                    Entry(
                        title="NPR News: 06-10-2026 12PM EDT",
                        summary="Hourly news update.",
                        published="Wed, 10 Jun 2026 16:11:33 +0000",
                        enclosures=[{"type": "audio/mpeg", "href": "https://audio.example.com/npr-12pm.mp3"}],
                    )
                ],
                "bozo": False,
            },
        )(),
    )

    episode, metadata = download.fetch_url_metadata(spotify_url, return_raw=True)

    assert episode["guid"] == "url:spotifypodcast:spotify-episode-id"
    assert episode["title"] == "NPR News: 06-10-2026 12PM EDT"
    assert episode["podcast_name"] == "NPR News Now"
    assert episode["source_url"] == spotify_url
    assert episode["audio_url"] == "https://audio.example.com/npr-12pm.mp3"
    assert metadata["resolved_from"] == "spotify"


def test_spotify_resolver_falls_back_to_exact_apple_episode(monkeypatch):
    spotify_url = "https://open.spotify.com/episode/spotify-episode-id"

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    spotify_title = "Mark Zuckerberg on Meta's AGI Vision — 12-Minute Interview Digest"

    def fake_get(url, **_kwargs):
        if url == "https://open.spotify.com/oembed":
            return FakeResponse({"title": spotify_title})
        if url == "https://itunes.apple.com/search":
            return FakeResponse(
                {
                    "results": [
                        {
                            "trackName": spotify_title,
                            "collectionName": "Brilliant Minds' Digest",
                            "episodeUrl": "https://audio.example.com/zuckerberg.mp3",
                            "description": "Interview digest.",
                            "releaseDate": "2025-05-02T08:00:00Z",
                            "country": "USA",
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(download.requests, "get", fake_get)
    monkeypatch.setattr(
        download,
        "search_podcast",
        lambda query: {"name": query, "url": "https://feeds.example.com/wrong.xml", "language": "en"},
    )
    monkeypatch.setattr(
        download.feedparser,
        "parse",
        lambda _url: type(
            "Feed",
            (),
            {"entries": [Entry(title="Unrelated episode", enclosures=[])], "bozo": False},
        )(),
    )

    episode, metadata = download.fetch_url_metadata(spotify_url, return_raw=True)

    assert episode["title"] == spotify_title
    assert episode["podcast_name"] == "Brilliant Minds' Digest"
    assert episode["audio_url"] == "https://audio.example.com/zuckerberg.mp3"
    assert episode["published"] == "2025-05-02"
    assert metadata["resolved_from"] == "spotify"


def test_fetch_youtube_caption_transcript_uses_json3_track(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "events": [
                    {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Hello"}, {"utf8": " world"}]},
                    {"tStartMs": 3500, "dDurationMs": 1500, "segs": [{"utf8": "Stay hungry"}]},
                ]
            }

    def fake_get(url, timeout, headers):
        assert url == "https://example.com/captions.json3"
        assert timeout == 60
        assert headers["User-Agent"] == "Mozilla/5.0"
        return FakeResponse()

    monkeypatch.setattr(download.requests, "get", fake_get)

    transcript = download.fetch_youtube_caption_transcript(
        {
            "extractor_key": "Youtube",
            "subtitles": {},
            "automatic_captions": {
                "en": [{"ext": "json3", "url": "https://example.com/captions.json3"}],
            },
        },
        "en",
    )

    assert transcript["text"] == "Hello world Stay hungry"
    assert transcript["segments"][0] == {"start": 1.0, "end": 3.0, "text": "Hello world"}
    assert transcript["source"] == "youtube_automatic_captions"


def test_fetch_youtube_caption_transcript_can_parse_vtt_track(monkeypatch):
    class FakeResponse:
        text = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello <c>world</c>

00:00:03.500 --> 00:00:05.000
Stay hungry
"""

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download.requests, "get", lambda *_args, **_kwargs: FakeResponse())

    transcript = download.fetch_youtube_caption_transcript(
        {
            "extractor_key": "Youtube",
            "subtitles": {
                "en": [{"ext": "vtt", "url": "https://example.com/captions.vtt"}],
            },
            "automatic_captions": {},
        },
        "en",
    )

    assert transcript["text"] == "Hello world Stay hungry"
    assert transcript["segments"] == []
    assert transcript["source"] == "youtube_subtitles"
