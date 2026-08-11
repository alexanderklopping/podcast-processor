import base64
import json

from mediaverwerker.tasks import article, feeds, instagram, transcribe

PROFILE = {
    "username": "yuanunpackschina",
    "url": "https://www.instagram.com/yuanunpackschina/",
    "addedAt": "2026-08-11T10:00:00+00:00",
    "initialImport": "latest_one",
    "language": "auto",
}


def post(post_id, published_at, *, video=True, caption="Caption"):
    return {
        "id": post_id,
        "shortcode": post_id,
        "permalink": f"https://www.instagram.com/p/{post_id}/",
        "caption": caption,
        "publishedAt": published_at,
        "hasVideo": video,
    }


def configure_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "instagram_feeds.json"
    config_path.write_text(json.dumps([PROFILE]), encoding="utf-8")
    output_dir = tmp_path / "feeds"
    articles_dir = tmp_path / "articles"
    transcripts_dir = tmp_path / "transcripts"
    audio_dir = tmp_path / "audio"
    for directory in (output_dir, articles_dir, transcripts_dir, audio_dir):
        directory.mkdir()

    monkeypatch.setattr(instagram, "INSTAGRAM_FEEDS_FILE", config_path)
    monkeypatch.setattr(instagram, "FEEDS_DIR", output_dir)
    monkeypatch.setattr(instagram, "AUDIO_DIR", audio_dir)
    monkeypatch.setattr(instagram, "INSTAGRAM_COOKIES_B64", base64.b64encode(b"# Netscape HTTP Cookie File\n").decode())
    monkeypatch.setattr(feeds, "FEEDS_DIR", output_dir)
    monkeypatch.setattr(feeds, "ARTICLES_DIR", articles_dir)
    monkeypatch.setattr(article, "ARTICLES_DIR", articles_dir)
    monkeypatch.setattr(transcribe, "TRANSCRIPTS_DIR", transcripts_dir)
    return output_dir, articles_dir, transcripts_dir, audio_dir


def test_discover_profile_groups_carousel_files_and_ignores_images(monkeypatch, tmp_path):
    output = "\n".join(
        json.dumps(item)
        for item in [
            [
                1,
                {
                    "post_id": "NEWEST",
                    "post_shortcode": "NEWEST",
                    "post_url": "https://www.instagram.com/reel/NEWEST/",
                    "description": "Carousel caption",
                    "date": "2026-08-11T09:00:00+00:00",
                    "fullname": "Yuan Yang",
                },
            ],
            [
                3,
                "https://cdn.example/video.mp4",
                {
                    "post_id": "NEWEST",
                    "post_shortcode": "NEWEST",
                    "post_url": "https://www.instagram.com/reel/NEWEST/",
                    "extension": "mp4",
                    "date": "2026-08-11T09:00:00+00:00",
                },
            ],
            [
                3,
                "https://cdn.example/photo.jpg",
                {
                    "post_id": "PHOTO",
                    "post_shortcode": "PHOTO",
                    "extension": "jpg",
                    "date": "2026-08-10T09:00:00+00:00",
                },
            ],
        ]
    )
    monkeypatch.setattr(instagram, "_run_gallery_dl", lambda _args: output)

    result = instagram.discover_profile(PROFILE, tmp_path / "cookies.txt")

    assert result["displayName"] == "Yuan Yang"
    assert result["posts"][0]["id"] == "NEWEST"
    assert result["posts"][0]["permalink"] == "https://www.instagram.com/reel/NEWEST/"
    assert result["posts"][0]["hasVideo"] is True
    assert result["posts"][1]["hasVideo"] is False


def test_first_run_processes_latest_video_once_and_baselines_older_posts(monkeypatch, tmp_path):
    output_dir, *_ = configure_paths(monkeypatch, tmp_path)
    discovered = [
        post("PHOTO", "2026-08-11T10:00:00+00:00", video=False),
        post("LATEST", "2026-08-11T09:00:00+00:00"),
        post("OLDER", "2026-08-10T09:00:00+00:00"),
    ]
    monkeypatch.setattr(
        instagram,
        "discover_profile",
        lambda _profile, _cookies: {"displayName": "Yuan Yang", "posts": discovered},
    )
    processed = []
    monkeypatch.setattr(
        instagram,
        "process_instagram_post",
        lambda _profile, item, _display_name, _cookies: processed.append(item["id"]),
    )
    monkeypatch.setattr(instagram, "generate_instagram_rss_feed", lambda *_args: None)

    first = instagram.sync_instagram_feeds()
    second = instagram.sync_instagram_feeds()
    status = json.loads((output_dir / "instagram_status.json").read_text(encoding="utf-8"))

    assert first == {"processed": 1, "errors": []}
    assert second == {"processed": 0, "errors": []}
    assert processed == ["LATEST"]
    assert set(status[PROFILE["username"]]["seenPostIds"]) == {"PHOTO", "LATEST", "OLDER"}


def test_later_run_processes_all_new_videos_oldest_first(monkeypatch, tmp_path):
    output_dir, *_ = configure_paths(monkeypatch, tmp_path)
    (output_dir / "instagram_status.json").write_text(
        json.dumps(
            {
                PROFILE["username"]: {
                    "displayName": "Yuan Yang",
                    "feedFilename": "instagram-yuanunpackschina.xml",
                    "lastCheckedAt": None,
                    "lastSuccessAt": "2026-08-10T10:00:00+00:00",
                    "lastSeenPostId": "KNOWN",
                    "seenPostIds": ["KNOWN"],
                    "error": None,
                }
            }
        ),
        encoding="utf-8",
    )
    discovered = [
        post("NEWEST", "2026-08-12T10:00:00+00:00"),
        post("MIDDLE", "2026-08-12T09:00:00+00:00"),
        post("KNOWN", "2026-08-10T09:00:00+00:00"),
    ]
    monkeypatch.setattr(
        instagram,
        "discover_profile",
        lambda _profile, _cookies: {"displayName": "Yuan Yang", "posts": discovered},
    )
    processed = []
    monkeypatch.setattr(
        instagram,
        "process_instagram_post",
        lambda _profile, item, _display_name, _cookies: processed.append(item["id"]),
    )
    monkeypatch.setattr(instagram, "generate_instagram_rss_feed", lambda *_args: None)

    result = instagram.sync_instagram_feeds()

    assert result == {"processed": 2, "errors": []}
    assert processed == ["MIDDLE", "NEWEST"]


def test_partial_run_persists_each_success_before_a_later_post_fails(monkeypatch, tmp_path):
    output_dir, *_ = configure_paths(monkeypatch, tmp_path)
    (output_dir / "instagram_status.json").write_text(
        json.dumps(
            {
                PROFILE["username"]: {
                    "lastSeenPostId": "KNOWN",
                    "seenPostIds": ["KNOWN"],
                    "error": None,
                }
            }
        ),
        encoding="utf-8",
    )
    discovered = [
        post("NEWEST", "2026-08-12T10:00:00+00:00"),
        post("MIDDLE", "2026-08-12T09:00:00+00:00"),
        post("KNOWN", "2026-08-10T09:00:00+00:00"),
    ]
    monkeypatch.setattr(
        instagram,
        "discover_profile",
        lambda _profile, _cookies: {"displayName": "Yuan Yang", "posts": discovered},
    )
    processed = []

    def process(_profile, item, _display_name, _cookies):
        processed.append(item["id"])
        if item["id"] == "NEWEST":
            raise instagram.InstagramError("download failed")

    monkeypatch.setattr(instagram, "process_instagram_post", process)
    monkeypatch.setattr(instagram, "generate_instagram_rss_feed", lambda *_args: None)

    first = instagram.sync_instagram_feeds()
    second = instagram.sync_instagram_feeds()

    assert first["processed"] == 1
    assert first["errors"] == ["@yuanunpackschina: download failed"]
    assert second["processed"] == 0
    assert processed == ["MIDDLE", "NEWEST", "NEWEST"]


def test_missing_cookie_secret_sets_error_for_each_profile(monkeypatch, tmp_path):
    output_dir, *_ = configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(instagram, "INSTAGRAM_COOKIES_B64", None)

    result = instagram.sync_instagram_feeds()
    status = json.loads((output_dir / "instagram_status.json").read_text(encoding="utf-8"))

    assert result["errors"] == ["@yuanunpackschina: INSTAGRAM_COOKIES_B64 is not configured"]
    assert status[PROFILE["username"]]["error"] == "INSTAGRAM_COOKIES_B64 is not configured"


def test_discovery_failure_sets_visible_status_and_preserves_existing_feed(monkeypatch, tmp_path):
    output_dir, *_ = configure_paths(monkeypatch, tmp_path)
    feed_path = output_dir / "instagram-yuanunpackschina.xml"
    feed_path.write_text("existing feed", encoding="utf-8")
    monkeypatch.setattr(
        instagram,
        "discover_profile",
        lambda *_args: (_ for _ in ()).throw(instagram.InstagramError("Login required")),
    )

    result = instagram.sync_instagram_feeds()
    status = json.loads((output_dir / "instagram_status.json").read_text(encoding="utf-8"))

    assert result["processed"] == 0
    assert result["errors"] == ["@yuanunpackschina: Login required"]
    assert status[PROFILE["username"]]["error"] == "Login required"
    assert feed_path.read_text(encoding="utf-8") == "existing feed"


def test_process_and_feed_contract_uses_stable_guid_timestamp_caption_and_transcript(monkeypatch, tmp_path):
    output_dir, articles_dir, transcripts_dir, audio_dir = configure_paths(monkeypatch, tmp_path)
    video = audio_dir / "video.mp4"
    video.write_bytes(b"video")
    audio = audio_dir / "audio.mp3"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(instagram, "_download_post_videos", lambda *_args: [video])
    monkeypatch.setattr(instagram, "_extract_audio", lambda *_args: audio)
    monkeypatch.setattr(instagram, "transcribe_audio", lambda _path, language: "Transcript text")

    item = post("ABC123", "2026-08-11T09:30:00+00:00", caption="The caption")
    instagram.process_instagram_post(PROFILE, item, "Yuan Yang", tmp_path / "cookies.txt")
    second = post("DEF456", "2026-08-11T10:30:00+00:00", caption="The caption")
    instagram.process_instagram_post(PROFILE, second, "Yuan Yang", tmp_path / "cookies.txt")
    instagram.generate_instagram_rss_feed(PROFILE, {"displayName": "Yuan Yang"})
    xml = (output_dir / "instagram-yuanunpackschina.xml").read_text(encoding="utf-8")

    assert "instagram:yuanunpackschina:ABC123" in xml
    assert "instagram:yuanunpackschina:DEF456" in xml
    assert "Tue, 11 Aug 2026 09:30:00 +0000" in xml
    assert "https://www.instagram.com/p/ABC123/" in xml
    assert "The caption" in xml
    assert "Transcript text" in xml
    assert "@yuanunpackschina" in xml
    assert len(list(articles_dir.glob("*.md"))) == 2
    assert len(list(transcripts_dir.glob("*.txt"))) == 2
