"""Tests for mediaverwerker.state functions."""

import json
from unittest.mock import patch

from mediaverwerker.state import load_processed_episodes, save_processed_episodes


class TestProcessedEpisodes:
    def test_load_empty(self, tmp_path):
        """Returns empty list when file doesn't exist."""
        fake_path = tmp_path / "processed.json"
        with patch("mediaverwerker.state.PROCESSED_FILE", fake_path):
            assert load_processed_episodes() == []

    def test_save_and_load(self, tmp_path):
        """Round-trip save and load."""
        fake_path = tmp_path / "processed.json"
        episodes = ["guid-1", "guid-2", "guid-3"]
        with patch("mediaverwerker.state.PROCESSED_FILE", fake_path):
            save_processed_episodes(episodes)
            loaded = load_processed_episodes()
        assert loaded == episodes

    def test_load_corrupted(self, tmp_path):
        """Returns empty list and backs up corrupted file."""
        fake_path = tmp_path / "processed.json"
        fake_path.write_text("not json{{{")
        with patch("mediaverwerker.state.PROCESSED_FILE", fake_path):
            result = load_processed_episodes()
        assert result == []
        assert (tmp_path / "processed.json.bak").exists()

    def test_save_atomic(self, tmp_path):
        """Temp file is cleaned up on success."""
        fake_path = tmp_path / "processed.json"
        with patch("mediaverwerker.state.PROCESSED_FILE", fake_path):
            save_processed_episodes(["a"])
        assert not (tmp_path / "processed.json.tmp").exists()
        assert json.loads(fake_path.read_text()) == ["a"]
