from pathlib import Path

import pytest

from mediaverwerker import interviews


def _raw_turns():
    return [
        {"id": "0", "speaker": "A", "start": 0.0, "end": 3.0, "text": "Welkom bij dit gesprek."},
        {"id": "1", "speaker": "B", "start": 10.0, "end": 14.0, "text": "Dank je, fijn om er te zijn."},
        {"id": "2", "speaker": "A", "start": 20.0, "end": 24.0, "text": "Laten we beginnen."},
        {"id": "3", "speaker": "B", "start": 30.0, "end": 35.0, "text": "Dit is mijn antwoord."},
        {"id": "4", "speaker": "A", "start": 40.0, "end": 44.0, "text": "Een volgende vraag."},
        {"id": "5", "speaker": "B", "start": 50.0, "end": 55.0, "text": "Een volgend antwoord."},
    ]


def test_structure_uses_safe_speaker_fallbacks_and_exact_chapter_turns(monkeypatch):
    monkeypatch.setattr(
        interviews,
        "generate_json",
        lambda **_kwargs: {
            "title": "Gast — onderwerp",
            "language": "nl",
            "speakerMappings": [
                {
                    "speaker": "A",
                    "label": "Onzekere Naam",
                    "confidence": 0.8,
                    "role": "interviewer",
                    "roleConfidence": 0.95,
                },
                {
                    "speaker": "B",
                    "label": "Echte Gast",
                    "confidence": 0.95,
                    "role": "guest",
                    "roleConfidence": 0.95,
                },
            ],
            "chapters": [{"title": f"Deel {index + 1}", "startTime": index * 10.0} for index in range(6)],
        },
    )

    result = interviews.apply_interview_structure(_raw_turns(), {"title": "Bron"})

    assert result["speakers"] == [
        {"id": "A", "label": "Interviewer", "role": "interviewer", "confidence": 0.8},
        {"id": "B", "label": "Echte Gast", "role": "guest", "confidence": 0.95},
    ]
    assert [chapter["turnIndex"] for chapter in result["chapters"]] == list(range(6))
    assert result["turns"][0]["speakerLabel"] == "Interviewer"
    assert result["title"] == "Gast — Deel 1"
    assert result["language"] == "nl"


def test_structure_does_not_guess_roles_from_speaker_order(monkeypatch):
    monkeypatch.setattr(
        interviews,
        "generate_json",
        lambda **_kwargs: {
            "title": "Guest — distributed systems",
            "language": "English",
            "speakerMappings": [
                {
                    "speaker": speaker,
                    "label": "Speaker" if speaker == "A" else "Onzeker",
                    "confidence": 0.99 if speaker == "A" else 0.2,
                    "role": "speaker",
                    "roleConfidence": 0.3,
                }
                for speaker in ["A", "B"]
            ],
            "chapters": [{"title": f"Part {index + 1}", "startTime": index * 10.0} for index in range(6)],
        },
    )

    result = interviews.apply_interview_structure(_raw_turns(), {"title": "Bron"})

    assert [speaker["label"] for speaker in result["speakers"]] == ["Speaker 1", "Speaker 2"]
    assert [speaker["role"] for speaker in result["speakers"]] == ["speaker", "speaker"]
    assert result["language"] == "en"


def test_clean_transcript_blocks_large_content_loss(monkeypatch):
    monkeypatch.setattr(
        interviews,
        "generate_json",
        lambda **_kwargs: {"turns": [{"id": turn["id"], "text": "kort"} for turn in _raw_turns()]},
    )

    with pytest.raises(RuntimeError, match="Volledigheidscontrole"):
        interviews.clean_transcript(_raw_turns())


def test_clean_transcript_keeps_original_when_editing_empties_a_turn(monkeypatch):
    raw_turns = _raw_turns()
    monkeypatch.setattr(
        interviews,
        "generate_json",
        lambda **_kwargs: {
            "turns": [
                {"id": turn["id"], "text": "" if index == 2 else turn["text"]} for index, turn in enumerate(raw_turns)
            ]
        },
    )

    cleaned = interviews.clean_transcript(raw_turns)

    assert cleaned[2]["text"] == raw_turns[2]["text"]
    assert all(turn["text"] for turn in cleaned)


def test_clean_transcript_restores_model_reordered_ids(monkeypatch):
    raw_turns = _raw_turns()
    edited = [{"id": turn["id"], "text": f"Bewerkt {turn['text']}"} for turn in reversed(raw_turns)]
    monkeypatch.setattr(interviews, "generate_json", lambda **_kwargs: {"turns": edited})

    cleaned = interviews.clean_transcript(raw_turns)

    assert [turn["id"] for turn in cleaned] == [turn["id"] for turn in raw_turns]
    assert cleaned[0]["text"] == f"Bewerkt {raw_turns[0]['text']}"


def test_clean_transcript_keeps_original_batch_when_model_changes_ids(monkeypatch):
    raw_turns = _raw_turns()
    monkeypatch.setattr(
        interviews,
        "generate_json",
        lambda **_kwargs: {
            "turns": [{"id": f"changed-{index}", "text": turn["text"]} for index, turn in enumerate(raw_turns)]
        },
    )

    cleaned = interviews.clean_transcript(raw_turns)

    assert cleaned == raw_turns


def test_process_interview_always_downloads_audio_and_reports_fixed_statuses(monkeypatch, tmp_path):
    audio = tmp_path / "normalized.mp3"
    audio.write_bytes(b"mp3-data")
    calls = []

    class Callback:
        def __init__(self, *_args):
            pass

        def send(self, status, **kwargs):
            calls.append((status, kwargs))

    structured = {
        "title": "Gast — onderwerp",
        "language": "nl",
        "speakers": [{"id": "A", "label": "Interviewer", "role": "interviewer", "confidence": 0.5}],
        "turns": [{"speakerId": "A", "speakerLabel": "Interviewer", "startSec": 0, "endSec": 1, "text": "Hallo"}],
        "chapters": [
            {"id": f"hoofdstuk-{index}", "title": f"Deel {index}", "startSec": index, "turnIndex": 0}
            for index in range(1, 7)
        ],
    }
    monkeypatch.setattr(interviews, "InterviewCallback", Callback)
    monkeypatch.setattr(
        interviews,
        "fetch_url_metadata",
        lambda *_args, **_kwargs: (
            {
                "title": "Bron",
                "podcast_name": "Podcast",
                "audio_url": "https://audio.example/test.mp3",
                "published": "",
            },
            {"automatic_captions": {"nl": []}},
        ),
    )
    monkeypatch.setattr(
        interviews,
        "download_url_audio",
        lambda url: calls.append(("download-called", {"url": url})) or Path("downloaded.mp3"),
    )
    monkeypatch.setattr(interviews, "normalize_audio", lambda *_args: audio)
    monkeypatch.setattr(interviews, "get_audio_duration", lambda *_args: 61.5)
    monkeypatch.setattr(interviews, "transcribe_diarized", lambda *_args: _raw_turns())
    monkeypatch.setattr(interviews, "clean_transcript", lambda turns: turns)
    monkeypatch.setattr(interviews, "apply_interview_structure", lambda *_args: structured)
    monkeypatch.setattr(
        interviews,
        "upload_artifacts",
        lambda *_args: {
            "audio": {"url": "https://blob.example/audio", "pathname": "audio.mp3", "etag": None},
            "manifest": {"url": "https://blob.example/transcript", "pathname": "transcript.json"},
            "chapters": {"url": "https://blob.example/chapters", "pathname": "chapters.json"},
        },
    )

    result = interviews.process_interview("iv-1", "https://youtube.com/watch?v=x", "https://reader/callback")

    statuses = [status for status, _kwargs in calls if status != "download-called"]
    assert statuses == ["downloading", "transcribing", "editing", "publishing", "ready"]
    assert any(status == "download-called" for status, _kwargs in calls)
    assert result["audio"]["durationSec"] == 61.5
    assert "sourcePublishedAt" not in result
