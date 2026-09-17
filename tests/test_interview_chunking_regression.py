"""Regression: long diarization requests time out despite a valid upload size."""

import pytest

from mediaverwerker import interviews


def test_long_small_file_keeps_all_intervals_and_absolute_timestamps(tmp_path, monkeypatch):
    audio = tmp_path / "interview.mp3"
    audio.write_bytes(b"small compressed audio")
    chunks = [tmp_path / f"interview_chunk{i:03d}.mp3" for i in range(3)]
    durations = {audio: 650, chunks[0]: 300.04, chunks[1]: 300.04, chunks[2]: 50.04}
    monkeypatch.setattr(interviews, "get_audio_duration", durations.get)
    monkeypatch.setattr(interviews, "split_audio", lambda *_args, **_kwargs: chunks)
    calls = []

    def transcribe(path, speaker_prefix=""):
        calls.append(path)
        return [{"id": speaker_prefix + "0", "speaker": speaker_prefix + "A", "start": 1, "end": 8, "text": "Speech"}]

    monkeypatch.setattr(interviews, "_transcribe_file", transcribe)
    turns = interviews.transcribe_diarized(audio)
    assert calls == chunks
    assert [t["start"] for t in turns] == [1, 301, 601]
    assert [t["end"] for t in turns] == [8, 308, 608]
    assert len({t["id"] for t in turns}) == 3
    assert len({t["speaker"] for t in turns}) == 3


@pytest.mark.parametrize("problem", ["missing", "wrong-order", "short", "unknown"])
def test_invalid_audio_partition_stops_before_any_provider_call(tmp_path, monkeypatch, problem):
    audio = tmp_path / "interview.mp3"
    audio.write_bytes(b"audio")
    chunks = [tmp_path / f"interview_chunk{i:03d}.mp3" for i in range(2)]
    durations = {audio: 400, chunks[0]: 300, chunks[1]: 100}
    if problem == "missing":
        chunks.pop(0)
    elif problem == "wrong-order":
        chunks.reverse()
    elif problem == "short":
        durations[chunks[1]] = 20
    else:
        durations[audio] = None
    monkeypatch.setattr(interviews, "get_audio_duration", durations.get)
    monkeypatch.setattr(interviews, "split_audio", lambda *_args, **_kwargs: chunks)
    monkeypatch.setattr(interviews, "_transcribe_file", lambda *_a, **_kw: pytest.fail("Incomplete audio sent"))
    with pytest.raises(RuntimeError, match="transcriptie niet gestart"):
        interviews.transcribe_diarized(audio)


def test_short_recording_stays_one_request(tmp_path, monkeypatch):
    audio = tmp_path / "short.mp3"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(interviews, "get_audio_duration", lambda _: 45)
    monkeypatch.setattr(interviews, "split_audio", lambda *_a, **_kw: pytest.fail("Unnecessary split"))
    monkeypatch.setattr(interviews, "_transcribe_file", lambda path: [{"text": str(path)}])
    assert interviews.transcribe_diarized(audio) == [{"text": str(audio)}]
