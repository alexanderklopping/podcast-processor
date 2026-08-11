"""Integration-style tests for task-to-model routing."""

from mediaverwerker.tasks import segment, transcribe


def test_transcript_cleanup_routes_polish_and_score(monkeypatch):
    calls = []
    source = " ".join(["woord"] * 600)

    def fake_text(**kwargs):
        calls.append(("text", kwargs))
        return source

    def fake_json(**kwargs):
        calls.append(("json", kwargs))
        return {
            "scores": {"eigennamen": 9, "vloeiendheid": 9, "schoonheid": 9, "volledigheid": 9},
            "gemiddelde": 9,
            "problemen": [],
        }

    monkeypatch.setattr(transcribe, "generate_text", fake_text)
    monkeypatch.setattr(transcribe, "generate_json", fake_json)

    assert transcribe._cleanup_transcript(source) == source
    assert calls[0][1]["role"] == "polish"
    assert calls[0][1]["task"] == "transcript_cleanup"
    assert calls[1][1]["task"] == "transcript_cleanup_score"


def test_segment_selection_preserves_timestamp_boundaries(monkeypatch):
    transcript_segments = [
        {"start": index * 300.0, "end": (index + 1) * 300.0, "text": f"segment {index}"} for index in range(4)
    ]

    monkeypatch.setattr(
        segment,
        "generate_json",
        lambda **_kwargs: {
            "start_index": 0,
            "end_index": 2,
            "confidence": 0.9,
            "alternative_queries": [],
        },
    )

    result = segment.find_segment(transcript_segments, "AI", margin=5)

    assert result["start"] == 0
    assert result["end"] == 905
    assert result["segments"] == transcript_segments[:3]
