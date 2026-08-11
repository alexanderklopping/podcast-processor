"""Topic-based segment finding via shared structured model routing."""

import json
import logging

from ..ai import generate_json

logger = logging.getLogger("mediaverwerker")

SEGMENT_MAX_RETRIES = 5  # Max API calls total (initial + retries)

SEGMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "start_index": {"type": ["integer", "null"]},
        "end_index": {"type": ["integer", "null"]},
        "confidence": {"type": "number"},
        "alternative_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start_index", "end_index", "confidence", "alternative_queries"],
    "additionalProperties": False,
}

# Duration constraints per segment type
DURATION_LIMITS = {
    "segment": (600, 1200),  # 10-20 minutes
    "instart": (20, 370),  # 20s-6min
}


def find_segment(transcript_segments, topic, margin=5.0, segment_type="segment"):
    """Find the segment of a transcript that discusses a specific topic.

    Uses a structured model to find the segment, with duration validation and retry feedback.

    Args:
        transcript_segments: List of dicts with 'start', 'end', 'text' keys.
        topic: Topic description to search for.
        margin: Seconds of margin to add before/after the segment.
        segment_type: "segment" (full studio segment, 12-16 min) or "instart" (pre-recorded insert, 30-360s).

    Returns:
        dict with 'start', 'end', 'text', 'confidence', 'segments' keys, or None.
    """
    logger.info(f"Finding {segment_type} about: {topic}")

    # Build a condensed version of the transcript with indices
    numbered_segments = []
    for i, seg in enumerate(transcript_segments):
        numbered_segments.append(f"[{i}] ({seg['start']:.1f}s-{seg['end']:.1f}s) {seg['text'].strip()}")
    transcript_text = "\n".join(numbered_segments)

    if segment_type == "instart":
        type_guidance = """You are looking for an "instart" — a short pre-recorded video insert filmed OUTSIDE the studio.
Instarts typically last 30-360 seconds. They are NOT the studio discussion — they are the filmed report itself.
Look for the shift from studio to on-location narration, and where studio discussion resumes after.
Do NOT include the studio intro ("we gaan kijken") or the studio reaction after."""
    else:
        type_guidance = """You are looking for a coherent segment where the topic is the PRIMARY subject throughout.
Do NOT include "coming up" teasers/previews in the first ~3 minutes of the show.
Do NOT include other segments before or after. A typical segment is 12-16 minutes."""

    min_dur, max_dur = DURATION_LIMITS.get(segment_type, (0, 99999))

    system_prompt = f"""You are a transcript analyzer for Dutch TV shows. Return ONLY a JSON object.

{type_guidance}

Expected duration: {min_dur}-{max_dur} seconds. Verify your selection fits this range.

Return ONLY this JSON (no markdown, no explanation):
{{"start_index": N, "end_index": N, "confidence": 0.0-1.0, "alternative_queries": []}}

If no match: {{"start_index": null, "end_index": null, "confidence": 0.0, "alternative_queries": ["alt1", "alt2"]}}"""

    input_text = f"Topic: {topic}\n\nTranscript:\n{transcript_text}"

    for attempt in range(SEGMENT_MAX_RETRIES):
        logger.info(f"Segment search attempt {attempt + 1}/{SEGMENT_MAX_RETRIES}")

        result = generate_json(
            task="segment_selection",
            instructions=system_prompt,
            input_text=input_text,
            schema=SEGMENT_RESPONSE_SCHEMA,
            max_output_tokens=1024,
        )
        response_text = json.dumps(result)

        start_idx = result.get("start_index")
        end_idx = result.get("end_index")
        confidence = result.get("confidence", 0.0)

        if start_idx is None or end_idx is None or confidence <= 0.3:
            # Try alternative queries
            alternatives = result.get("alternative_queries", [])
            if alternatives and attempt < SEGMENT_MAX_RETRIES - 1:
                alt = alternatives[0]
                logger.info(f"Not found, trying alternative: '{alt}'")
                input_text = f"Topic: {alt}\n\nTranscript:\n{transcript_text}"
                continue
            logger.info(f"Topic '{topic}' not found after {attempt + 1} attempts")
            return None

        matched_segments = transcript_segments[start_idx : end_idx + 1]
        if not matched_segments:
            continue

        start_time = max(0, matched_segments[0]["start"] - margin)
        end_time = matched_segments[-1]["end"] + margin
        duration = end_time - start_time

        # Duration validation
        if duration > max_dur and attempt < SEGMENT_MAX_RETRIES - 1:
            logger.warning(f"{segment_type} too long ({duration:.0f}s > {max_dur}s), asking to narrow")
            input_text += (
                f"\n\nPrevious selection: {response_text}\n"
                f"That selection is {duration:.0f} seconds ({duration / 60:.1f} min), "
                f"but a {segment_type} should be {min_dur}-{max_dur} seconds. "
                "Narrow the selection."
            )
            continue
        elif duration < min_dur and attempt < SEGMENT_MAX_RETRIES - 1:
            logger.warning(f"{segment_type} too short ({duration:.0f}s < {min_dur}s), asking to widen")
            input_text += (
                f"\n\nPrevious selection: {response_text}\n"
                f"That selection is only {duration:.0f} seconds, "
                f"but a {segment_type} should be {min_dur}-{max_dur} seconds. "
                "Widen the selection."
            )
            continue

        segment_text = " ".join(seg["text"].strip() for seg in matched_segments)

        logger.info(
            f"Found {segment_type}: {start_time:.1f}s - {end_time:.1f}s ({duration:.0f}s, confidence: {confidence:.1%})"
        )

        return {
            "start": start_time,
            "end": end_time,
            "text": segment_text,
            "confidence": confidence,
            "segments": matched_segments,
        }

    logger.info(f"Topic '{topic}' not found in transcript")
    return None


def find_eva_segment(segments, margin=5.0):
    """Find the tech/AI segment (Alexander Klöpping) in an Eva (NPO1) episode.

    Uses semantic model routing to identify the recurring tech segment. More robust than keyword matching
    because the segment topic varies per episode (AI, self-driving cars, crypto, etc.).

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        margin: Seconds of margin to add before/after the segment.

    Returns:
        dict with 'start', 'end', 'text', 'confidence', 'segments' keys, or None.
    """
    logger.info("Searching for Alexander Klöpping tech segment in Eva episode")

    return find_segment(
        segments,
        topic=(
            "Het vaste tech-segment van Alexander Klöpping in de talkshow Eva. "
            "Dit is een terugkerend onderdeel waarin techdeskundige Alexander Klöpping "
            "een technologie-onderwerp bespreekt in de studio, vaak met een instart (vooraf opgenomen reportage). "
            "Het segment begint wanneer het onderwerp in de studio geïntroduceerd wordt (NA de 'coming up') "
            "en eindigt wanneer ze overgaan op een ander onderwerp. "
            "LET OP: de 'coming up' preview in de eerste 3 minuten telt NIET mee."
        ),
        margin=margin,
        segment_type="segment",
    )
