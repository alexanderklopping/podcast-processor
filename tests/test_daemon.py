"""Tests for mediaverwerker.daemon helpers."""

from mediaverwerker.daemon import ServiceState, parse_positive_int


class TestParsePositiveInt:
    def test_missing_value_uses_default(self):
        assert parse_positive_int(None, 60) == 60

    def test_invalid_value_uses_default(self):
        assert parse_positive_int("abc", 60) == 60

    def test_non_positive_value_uses_default(self):
        assert parse_positive_int("0", 60) == 60

    def test_valid_value_is_returned(self):
        assert parse_positive_int("15", 60) == 15


class TestServiceState:
    def test_snapshot_before_first_run_is_healthy(self):
        state = ServiceState()
        snapshot = state.snapshot()
        assert snapshot["healthy"] is True
        assert snapshot["run_count"] == 0
        assert snapshot["last_exit_code"] is None

    def test_failed_run_marks_state_unhealthy(self):
        state = ServiceState()
        state.begin_run()
        state.finish_run(1)
        snapshot = state.snapshot()
        assert snapshot["healthy"] is False
        assert snapshot["running"] is False
        assert snapshot["run_count"] == 1
        assert snapshot["last_exit_code"] == 1
