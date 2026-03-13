"""Video/audio clipping and subtitle generation via ffmpeg."""

import logging
import subprocess
from pathlib import Path

from ..util import format_timestamp, format_srt_timestamp

logger = logging.getLogger("mediaverwerker")


def clip_media(input_path, output_path, start, end):
    """Clip a segment from a video/audio file with re-encoding for precise cuts.

    Args:
        input_path: Path to input media file.
        output_path: Path for output file.
        start: Start time in seconds.
        end: End time in seconds.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end - start
    logger.info(f"Clipping: {format_timestamp(start)} -> {format_timestamp(end)} ({format_timestamp(duration)})")

    # Detect if input is video or audio
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(input_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    has_video = "video" in probe_result.stdout

    if has_video:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(input_path),
            "-t", str(duration),
            "-acodec", "libmp3lame",
            "-ab", "192k",
            str(output_path),
        ]

    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"Clip saved: {output_path.name}")
    return output_path


def generate_srt(segments, offset, srt_path):
    """Generate an SRT subtitle file from segments, adjusted for clip offset.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        offset: Time offset (start of clip) to subtract from timestamps.
        srt_path: Output path for the SRT file.
    """
    srt_path = Path(srt_path)
    lines = []
    counter = 1

    for seg in segments:
        start = seg["start"] - offset
        end = seg["end"] - offset
        if start < 0:
            continue
        text = seg["text"].strip()
        if not text:
            continue
        lines.append(str(counter))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
        counter += 1

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Subtitles saved: {srt_path.name} ({counter - 1} lines)")
    return srt_path


def burn_subtitles(input_path, srt_path, output_path):
    """Burn SRT subtitles into video."""
    input_path = Path(input_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)

    logger.info("Burning subtitles...")

    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    subtitle_filter = (
        f"subtitles='{srt_escaped}'"
        f":force_style='FontSize=24,FontName=Arial,"
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
        f"Outline=2,Shadow=1'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", subtitle_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"Video with subtitles: {output_path.name}")
    return output_path
