"""Topic-based segment finding via Claude API."""

import json
import logging

from anthropic import Anthropic

from ..config import ANTHROPIC_API_KEY

logger = logging.getLogger("mediaverwerker")


def find_segment(transcript_segments, topic, margin=5.0):
    """Find the segment of a transcript that discusses a specific topic.

    Args:
        transcript_segments: List of dicts with 'start', 'end', 'text' keys.
        topic: Topic description to search for (e.g., "Anthropic", "AI regulation").
        margin: Seconds of margin to add before/after the segment.

    Returns:
        dict with 'start', 'end', 'text' keys for the matched segment.
    """
    logger.info(f"Finding segment about: {topic}")

    # Build a condensed version of the transcript with indices
    numbered_segments = []
    for i, seg in enumerate(transcript_segments):
        numbered_segments.append(f"[{i}] ({seg['start']:.1f}s-{seg['end']:.1f}s) {seg['text'].strip()}")

    transcript_text = "\n".join(numbered_segments)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="""You are a transcript analyzer. Given a timestamped transcript and a topic, identify the segment indices where that topic is discussed.

Return ONLY valid JSON in this exact format:
{"start_index": <first segment index>, "end_index": <last segment index>}

If the topic is not found, return:
{"start_index": null, "end_index": null}""",
        messages=[{
            "role": "user",
            "content": f"Topic: {topic}\n\nTranscript:\n{transcript_text}"
        }]
    )

    try:
        result = json.loads(message.content[0].text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse segment response: {message.content[0].text}")
        return None

    start_idx = result.get("start_index")
    end_idx = result.get("end_index")

    if start_idx is None or end_idx is None:
        logger.info(f"Topic '{topic}' not found in transcript")
        return None

    # Extract the segment
    matched_segments = transcript_segments[start_idx:end_idx + 1]
    if not matched_segments:
        return None

    start_time = max(0, matched_segments[0]["start"] - margin)
    end_time = matched_segments[-1]["end"] + margin
    segment_text = " ".join(seg["text"].strip() for seg in matched_segments)

    logger.info(f"Found segment: {start_time:.1f}s - {end_time:.1f}s ({end_time - start_time:.0f}s)")

    return {
        "start": start_time,
        "end": end_time,
        "text": segment_text,
        "segments": matched_segments,
    }
