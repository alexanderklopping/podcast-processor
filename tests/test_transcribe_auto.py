from pathlib import Path

from mediaverwerker.tasks import transcribe


class FakeTranscriptions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return "transcript"


class FakeClient:
    def __init__(self):
        self.audio = type("Audio", (), {"transcriptions": FakeTranscriptions()})()


def test_auto_language_omits_language_hint(tmp_path):
    audio_path = Path(tmp_path) / "audio.mp3"
    audio_path.write_bytes(b"audio")
    client = FakeClient()

    transcribe.transcribe_single_file(client, "whisper", audio_path, language="auto")

    assert "language" not in client.audio.transcriptions.request


def test_explicit_language_is_forwarded(tmp_path):
    audio_path = Path(tmp_path) / "audio.mp3"
    audio_path.write_bytes(b"audio")
    client = FakeClient()

    transcribe.transcribe_single_file(client, "whisper", audio_path, language="nl")

    assert client.audio.transcriptions.request["language"] == "nl"
