"""Tests for mediaverwerker.util pure functions."""

import pytest

from mediaverwerker.util import (
    extract_urls,
    format_srt_timestamp,
    format_timestamp,
    is_url,
    sanitize_filename,
    validate_url,
)


class TestSanitizeFilename:
    def test_basic(self):
        assert sanitize_filename("Hello World") == "Hello World"

    def test_special_chars(self):
        assert sanitize_filename("Episode #1: The Beginning!") == "Episode _1_ The Beginning_"

    def test_truncation(self):
        long_title = "A" * 200
        assert len(sanitize_filename(long_title)) <= 100

    def test_empty(self):
        assert sanitize_filename("") == ""

    def test_preserves_hyphens_underscores(self):
        assert sanitize_filename("my-podcast_ep01") == "my-podcast_ep01"


class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com")

    def test_https(self):
        assert is_url("https://youtube.com/watch?v=abc123")

    def test_not_url(self):
        assert not is_url("just some text")

    def test_not_url_partial(self):
        assert not is_url("go to https://example.com please")

    def test_stripped(self):
        assert is_url("  https://example.com  ")


class TestExtractUrls:
    def test_single(self):
        assert extract_urls("check https://example.com out") == ["https://example.com"]

    def test_multiple(self):
        urls = extract_urls("see http://a.com and https://b.com")
        assert len(urls) == 2

    def test_none(self):
        assert extract_urls("no urls here") == []


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00"

    def test_seconds(self):
        assert format_timestamp(45) == "00:00:45"

    def test_minutes(self):
        assert format_timestamp(125) == "00:02:05"

    def test_hours(self):
        assert format_timestamp(3661) == "01:01:01"


class TestFormatSrtTimestamp:
    def test_zero(self):
        assert format_srt_timestamp(0) == "00:00:00,000"

    def test_with_millis(self):
        assert format_srt_timestamp(1.5) == "00:00:01,500"

    def test_complex(self):
        assert format_srt_timestamp(3723.25) == "01:02:03,250"


class TestValidateUrl:
    def test_valid_https(self):
        assert validate_url("https://youtube.com/watch?v=abc") == "https://youtube.com/watch?v=abc"

    def test_valid_http(self):
        assert validate_url("http://example.com") == "http://example.com"

    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("ftp://example.com/file")

    def test_rejects_file(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("file:///etc/passwd")

    def test_rejects_no_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("example.com")

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="no hostname"):
            validate_url("http://")

    def test_rejects_loopback_ip(self):
        with pytest.raises(ValueError, match="non-public"):
            validate_url("http://127.0.0.1/admin")

    def test_rejects_private_ip(self):
        with pytest.raises(ValueError, match="non-public"):
            validate_url("http://192.168.1.1/")

    def test_rejects_link_local(self):
        with pytest.raises(ValueError, match="non-public"):
            validate_url("http://169.254.169.254/latest/meta-data/")
