"""Natural language command parsing via Claude API."""

import json
import logging
from datetime import datetime

from anthropic import Anthropic

from .config import ANTHROPIC_API_KEY
from .state import load_podcasts

logger = logging.getLogger("mediaverwerker")

PARSE_SYSTEM_PROMPT = """You are a command parser for Mediaverwerker, a media processing CLI tool.

Given a natural language command in Dutch or English, parse it into a structured JSON action plan.

Available actions:
- "process_all": Process all new episodes from configured (recurring) podcasts
- "process_episode": Process a specific episode from a CONFIGURED podcast
  - podcast: podcast name (must be from the configured list below)
  - date: date string (ISO format, or "yesterday"/"today")
- "adhoc_episode": Download and process an episode from ANY podcast (not limited to configured ones)
  - podcast_query: the podcast name to search for (e.g., "Hard Fork", "Lex Fridman")
  - date: date string (optional, ISO format or "yesterday"/"today")
  - topic: topic to find in transcript (optional)
  - output: "transcript" (default), "article"
  - output_dir: where to save files (optional, e.g., "~/Desktop")
- "find_segment": Find a segment about a topic in a configured podcast
  - podcast: podcast name
  - date: date string (optional)
  - topic: what to search for
  - output: "transcript" (default), "article", or "clip"
- "clip": Clip a segment from a local file
  - input: file path
  - topic: topic to find (optional)
  - subtitles: boolean
  - burn: boolean
- "feeds_update": Update and push RSS feeds

CONFIGURED (recurring) podcasts:
{podcasts}

IMPORTANT: Any podcast NOT in the configured list should use "adhoc_episode", not "process_episode" or "find_segment". The tool will automatically search for the podcast's RSS feed. You can handle ANY podcast - you are not limited to the list above.

Today's date: {today}

Return ONLY valid JSON in this format:
{{
  "actions": [
    {{"type": "action_type", ...parameters}}
  ],
  "description": "Brief Dutch description of what will happen"
}}

Examples:
- "verwerk alles" -> {{"actions": [{{"type": "process_all"}}], "description": "Alle nieuwe afleveringen verwerken"}}
- "download hard fork van gisteren en transcript over Anthropic" -> {{"actions": [{{"type": "adhoc_episode", "podcast_query": "Hard Fork", "date": "yesterday", "topic": "Anthropic", "output": "transcript"}}], "description": "Hard Fork van gisteren downloaden, segment over Anthropic zoeken"}}
- "download de laatste hardfork en zet de file op m'n desktop incl transcript" -> {{"actions": [{{"type": "adhoc_episode", "podcast_query": "Hard Fork", "output": "transcript", "output_dir": "~/Desktop"}}], "description": "Laatste Hard Fork downloaden naar Desktop met transcript"}}
- "geef me het stuk over AI regulation uit de laatste dwarkesh" -> {{"actions": [{{"type": "find_segment", "podcast": "Dwarkesh", "topic": "AI regulation", "output": "transcript"}}], "description": "AI regulation segment zoeken in laatste Dwarkesh"}}
- "download de laatste lex fridman" -> {{"actions": [{{"type": "adhoc_episode", "podcast_query": "Lex Fridman Podcast"}}], "description": "Laatste Lex Fridman aflevering downloaden"}}"""


def parse_command(user_input):
    """Parse a natural language command into structured actions.

    Args:
        user_input: Natural language command string.

    Returns:
        dict with 'actions' list and 'description' string.
    """
    podcasts = load_podcasts()
    podcasts_str = "\n".join(f"- {p['name']} ({p.get('language', 'en')}): {p['url']}" for p in podcasts)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=PARSE_SYSTEM_PROMPT.format(
            podcasts=podcasts_str,
            today=datetime.now().strftime("%Y-%m-%d"),
        ),
        messages=[{"role": "user", "content": user_input}],
    )

    response_text = message.content[0].text.strip()

    # Strip markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines).strip()

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse NLP response: {response_text}")
        return {"actions": [], "description": "Could not parse command", "error": True}

    # Validate structure
    ALLOWED_ACTIONS = {"process_all", "process_episode", "adhoc_episode", "find_segment", "clip", "feeds_update"}
    if not isinstance(result.get("actions"), list):
        return {"actions": [], "description": "Invalid response structure", "error": True}

    validated_actions = []
    for action in result["actions"]:
        if not isinstance(action, dict):
            continue
        if action.get("type") not in ALLOWED_ACTIONS:
            logger.warning(f"Skipping unknown action type: {action.get('type')}")
            continue
        validated_actions.append(action)

    result["actions"] = validated_actions
    logger.info(f"Parsed command: {result.get('description', '')}")
    return result
