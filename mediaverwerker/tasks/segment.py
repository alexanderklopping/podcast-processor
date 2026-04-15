"""Topic-based segment finding via Claude API."""

import json
import logging

from anthropic import Anthropic

from ..config import ANTHROPIC_API_KEY

logger = logging.getLogger("mediaverwerker")

SEGMENT_MAX_RETRIES = 3  # Max attempts to find a segment


def find_segment(transcript_segments, topic, margin=5.0):
    """Find the segment of a transcript that discusses a specific topic.

    Uses an iterative loop: search → validate → if not found, reformulate query
    with synonyms/broader terms and retry.

    Args:
        transcript_segments: List of dicts with 'start', 'end', 'text' keys.
        topic: Topic description to search for (e.g., "Anthropic", "AI regulation").
        margin: Seconds of margin to add before/after the segment.

    Returns:
        dict with 'start', 'end', 'text', 'confidence', 'segments' keys for the matched segment,
        or None if not found after all attempts.
    """
    logger.info(f"Finding segment about: {topic}")

    # Build a condensed version of the transcript with indices
    numbered_segments = []
    for i, seg in enumerate(transcript_segments):
        numbered_segments.append(f"[{i}] ({seg['start']:.1f}s-{seg['end']:.1f}s) {seg['text'].strip()}")

    transcript_text = "\n".join(numbered_segments)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    search_queries = [topic]  # Start with the original topic

    for attempt in range(SEGMENT_MAX_RETRIES):
        current_query = search_queries[-1]
        logger.info(f"Segment search attempt {attempt + 1}/{SEGMENT_MAX_RETRIES}: '{current_query}'")

        system_prompt = """You are a transcript analyzer. Given a timestamped transcript and a topic, identify the segment indices where that topic is discussed.

Return ONLY valid JSON in this exact format:
{
  "start_index": <first segment index or null>,
  "end_index": <last segment index or null>,
  "confidence": <0.0-1.0 how confident you are this matches the topic>,
  "alternative_queries": ["query1", "query2"]
}

Rules:
- If you find a clear match, return the indices and confidence > 0.7
- If you find a partial/tangential match, return indices with confidence 0.3-0.7
- If no match at all, return null indices, confidence 0.0, and suggest 2 alternative search queries that might find related content (synonyms, broader terms, related concepts)
- The alternative_queries should be meaningfully different from the current query"""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Topic: {current_query}\n\nTranscript:\n{transcript_text}"}],
        )

        try:
            response_text = message.content[0].text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                response_text = "\n".join(lines).strip()
            result = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse segment response: {message.content[0].text}")
            if attempt < SEGMENT_MAX_RETRIES - 1:
                continue
            return None

        start_idx = result.get("start_index")
        end_idx = result.get("end_index")
        confidence = result.get("confidence", 0.0)

        # Found a match with reasonable confidence
        if start_idx is not None and end_idx is not None and confidence > 0.3:
            matched_segments = transcript_segments[start_idx : end_idx + 1]
            if not matched_segments:
                continue

            start_time = max(0, matched_segments[0]["start"] - margin)
            end_time = matched_segments[-1]["end"] + margin
            segment_text = " ".join(seg["text"].strip() for seg in matched_segments)

            logger.info(
                f"Found segment: {start_time:.1f}s - {end_time:.1f}s "
                f"({end_time - start_time:.0f}s, confidence: {confidence:.1%})"
            )

            return {
                "start": start_time,
                "end": end_time,
                "text": segment_text,
                "confidence": confidence,
                "segments": matched_segments,
            }

        # Not found or low confidence - try alternative queries
        alternatives = result.get("alternative_queries", [])
        if alternatives and attempt < SEGMENT_MAX_RETRIES - 1:
            # Pick the first alternative we haven't tried yet
            for alt in alternatives:
                if alt.lower() not in [q.lower() for q in search_queries]:
                    search_queries.append(alt)
                    logger.info(f"Topic '{current_query}' not found, trying alternative: '{alt}'")
                    break
            else:
                # All alternatives already tried
                logger.info(f"Topic '{topic}' not found and no new alternatives to try")
                return None
        elif attempt >= SEGMENT_MAX_RETRIES - 1:
            logger.info(f"Topic '{topic}' not found after {SEGMENT_MAX_RETRIES} attempts")
            return None

    logger.info(f"Topic '{topic}' not found in transcript")
    return None


# Keywords that signal the AI segment in Eva (NPO1)
EVA_AI_KEYWORDS = [
    "kunstmatige intelligentie",
    "artificial intelligence",
    " ai ",
    " ai-",
    "chatgpt",
    "deepseek",
    "openai",
    "algoritme",
    "klöpping",
    "klopping",
    "alexander klöpping",
]

EVA_INTRO_KEYWORDS = [
    "klöpping",
    "klopping",
    "alexander klöpping",
    "alexander klopping",
    "techdeskundige",
    "wereld van ai",
]


def find_eva_segment(segments, margin=5.0):
    """Find the AI segment in an Eva (NPO1) episode using keyword scoring.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        margin: Seconds of margin to add before/after the segment.

    Returns:
        dict with 'start', 'end', 'text', 'confidence', 'segments' keys,
        or None if no AI segment found.
    """
    logger.info("Searching for AI segment in Eva episode (keyword-based)")

    # Score each segment for AI-relevance
    scored = []
    for seg in segments:
        text = seg["text"].lower()
        score = sum(1 for kw in EVA_AI_KEYWORDS if kw in text)
        scored.append((seg["start"], seg["end"], score, seg["text"], seg))

    # Find the introduction (first mention of Klöpping or AI intro)
    intro_start = None
    for start, end, score, text, seg in scored:
        text_lower = text.lower()
        if any(kw in text_lower for kw in EVA_INTRO_KEYWORDS):
            intro_start = start
            break

    if intro_start is None:
        # Fallback: first segment with any AI keyword
        for start, end, score, text, seg in scored:
            if score > 0:
                intro_start = start
                break

    if intro_start is None:
        logger.warning("No AI segment found in Eva episode")
        return None

    # Find the end: last AI keyword occurrence after intro
    last_ai_time = intro_start
    for start, end, score, text, seg in scored:
        if start >= intro_start and score > 0:
            last_ai_time = end

    # Look for a natural ending (up to 30s after last keyword)
    segment_end = last_ai_time
    for start, end, score, text, seg in scored:
        if start > last_ai_time and start <= last_ai_time + 30:
            text_lower = text.lower()
            if any(w in text_lower for w in ["dank", "bedankt", "dankjewel"]):
                segment_end = end
                break
            segment_end = end
        elif start > last_ai_time + 30:
            break

    # Apply margin
    start_time = max(0, intro_start - margin)
    end_time = segment_end + margin

    # Collect matched segments
    matched = [seg for _, _, _, _, seg in scored if seg["start"] >= start_time and seg["end"] <= end_time + 5]
    segment_text = " ".join(seg["text"].strip() for seg in matched)

    duration = end_time - start_time
    logger.info(f"Eva AI segment found: {start_time:.1f}s - {end_time:.1f}s ({duration:.0f}s)")

    return {
        "start": start_time,
        "end": end_time,
        "text": segment_text,
        "confidence": 1.0,
        "segments": matched,
    }
