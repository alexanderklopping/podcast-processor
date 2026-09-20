"""Do not submit header-only chunks when a source duration overstates its audio."""

import shutil
import subprocess

import pytest

from mediaverwerker import util


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg integration fixture")
def test_split_uses_decoded_audio_eof_not_reported_duration(tmp_path, monkeypatch):
    source = tmp_path / "episode.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.1", str(source)],
        check=True,
    )
    # Dynamic podcast enclosures can report more duration than decodable audio.
    monkeypatch.setattr(util, "get_audio_duration", lambda _: 2.1)
    chunks = util.split_audio(source, chunk_duration_seconds=0.5)
    assert len(chunks) == 3
    decoded_bytes = 0
    for chunk in chunks:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(chunk), "-f", "s16le", "-"],
            capture_output=True,
            check=True,
        )
        decoded_bytes += len(result.stdout)
        assert len(result.stdout) > 320  # Every chunk contains actual audio samples.

    assert decoded_bytes >= 1.1 * 16000 * 2  # Full source duration survives.
    # Retrying a shorter recording must not pick up an old fourth chunk.
    (chunks[0].parent / "episode_chunk003.mp3").write_bytes(b"stale")
    retry = util.split_audio(source, chunk_duration_seconds=0.5)
    assert len(retry) == 3
    assert retry[0].parent != chunks[0].parent


def test_split_failure_never_returns_partial_audio(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(util.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        util.split_audio(tmp_path / "episode.mp3")
