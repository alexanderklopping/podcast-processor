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
- "process_url": Download and process a single media URL into the "Individuele Afleveringen" feed
  - url: media URL (YouTube, direct episode URL, or any yt-dlp supported page)
  - topic: topic to find in transcript (optional)
  - output: "article" (default) or "transcript"
  - output_dir: where to save files (optional, e.g., "~/Desktop")
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
- "process_url": Process a video/audio from a direct URL (YouTube, Twitter/X, etc.)
  - url: the full URL to process
  - language: language code (default "en", use "nl" for Dutch content)

CONFIGURED (recurring) podcasts:
{podcasts}

IMPORTANT: If the input contains a URL (starting with http:// or https://), ALWAYS use "process_url". Do NOT try to interpret a URL as a podcast name.

IMPORTANT: Any podcast NOT in the configured list should use "adhoc_episode", not "process_episode" or "find_segment". The tool will automatically search for the podcast's RSS feed. You can handle ANY podcast - you are not limited to the list above.
IMPORTANT: If the user gives a concrete URL to one specific episode or video, use "process_url" instead of searching by podcast name.

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
- "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -> {{"actions": [{{"type": "process_url", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "output": "article"}}], "description": "Losse URL verwerken en publiceren in de feed met individuele afleveringen"}}
- "verwerk deze url en zet transcript op mijn desktop: https://example.com/episode" -> {{"actions": [{{"type": "process_url", "url": "https://example.com/episode", "output": "transcript", "output_dir": "~/Desktop"}}], "description": "Losse URL verwerken en transcript opslaan op Desktop"}}
- "download hard fork van gisteren en transcript over Anthropic" -> {{"actions": [{{"type": "adhoc_episode", "podcast_query": "Hard Fork", "date": "yesterday", "topic": "Anthropic", "output": "transcript"}}], "description": "Hard Fork van gisteren downloaden, segment over Anthropic zoeken"}}
- "download de laatste hardfork en zet de file op m'n desktop incl transcript" -> {{"actions": [{{"type": "adhoc_episode", "podcast_query": "Hard Fork", "output": "transcript", "output_dir": "~/Desktop"}}], "description": "Laatste Hard Fork downloaden naar Desktop met transcript"}}
- "geef me het stuk over AI regulation uit de laatste dwarkesh" -> {{"actions": [{{"type": "find_segment", "podcast": "Dwarkesh", "topic": "AI regulation", "output": "transcript"}}], "description": "AI regulation segment zoeken in laatste Dwarkesh"}}
- "download de laatste lex fridman"
- "verwerk https://youtube.com/watch?v=abc" -> {{"actions": [{{"type": "process_url", "url": "https://youtube.com/watch?v=abc", "language": "en"}}], "description": "YouTube video verwerken"}}
- "verwerk https://x.com/user/status/123" -> {{"actions": [{{"type": "process_url", "url": "https://x.com/user/status/123", "language": "en"}}], "description": "Twitter video verwerken"}} -> {{"actions": [{{"type": "adhoc_episode", "podcast_query": "Lex Fridman Podcast"}}], "description": "Laatste Lex Fridman aflevering downloaden"}}"""


PARSE_MAX_RETRIES = 3  # Max parsing attempts before giving up

# Validation rules for simulating execution feasibility
ALLOWED_ACTIONS = {
    "process_all",
    "process_episode",
    "process_url",
    "adhoc_episode",
    "find_segment",
    "clip",
    "feeds_update",
}

REQUIRED_FIELDS = {
    "process_url": ["url"],
    "adhoc_episode": ["podcast_query"],
    "find_segment": ["podcast", "topic"],
    "process_episode": ["podcast"],
    "clip": ["file"],
}


def _validate_parsed_result(result, podcasts):
    """Validate a parsed result and return (validated_result, issues).

    Returns tuple of (result_dict, list_of_issue_strings).
    Issues are problems that should trigger a retry with feedback.
    """
    issues = []

    if not isinstance(result.get("actions"), list):
        return None, ["Response has no 'actions' list"]

    if not result["actions"]:
        return None, ["No actions were parsed from the command"]

    validated_actions = []
    podcast_names = [p["name"].lower() for p in podcasts]

    for action in result["actions"]:
        if not isinstance(action, dict):
            issues.append("Action is not a dict")
            continue

        action_type = action.get("type")
        if action_type not in ALLOWED_ACTIONS:
            issues.append(f"Unknown action type: {action_type}")
            continue

        # Check required fields
        missing = [f for f in REQUIRED_FIELDS.get(action_type, []) if not action.get(f)]
        if missing:
            issues.append(f"Action '{action_type}' missing required fields: {missing}")
            continue

        # Simulate execution: check if podcast name matches configured podcasts
        if action_type == "process_episode":
            podcast_name = action.get("podcast", "").lower()
            if not any(podcast_name in pn or pn in podcast_name for pn in podcast_names):
                issues.append(
                    f"Podcast '{action.get('podcast')}' not in configured list. "
                    f"Use 'adhoc_episode' instead of 'process_episode' for non-configured podcasts. "
                    f"Configured: {', '.join(p['name'] for p in podcasts)}"
                )
                continue

        if action_type == "find_segment":
            podcast_name = action.get("podcast", "").lower()
            if not any(podcast_name in pn or pn in podcast_name for pn in podcast_names):
                issues.append(
                    f"Podcast '{action.get('podcast')}' not in configured list. "
                    f"Use 'adhoc_episode' with a topic instead of 'find_segment' for non-configured podcasts."
                )
                continue

        # Check URL validity for process_url
        if action_type == "process_url":
            url = action.get("url", "")
            if not url.startswith(("http://", "https://")):
                issues.append(f"Invalid URL: '{url}' - must start with http:// or https://")
                continue

        validated_actions.append(action)

    result["actions"] = validated_actions
    return result, issues


def parse_command(user_input):
    """Parse a natural language command into structured actions.

    Uses an iterative loop: parse → validate → retry with feedback if issues found.

    Args:
        user_input: Natural language command string.

    Returns:
        dict with 'actions' list and 'description' string.
    """
    podcasts = load_podcasts()
    podcasts_str = "\n".join(f"- {p['name']} ({p.get('language', 'en')}): {p['url']}" for p in podcasts)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = PARSE_SYSTEM_PROMPT.format(
        podcasts=podcasts_str,
        today=datetime.now().strftime("%Y-%m-%d"),
    )

    messages = [{"role": "user", "content": user_input}]

    for attempt in range(PARSE_MAX_RETRIES):
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

        response_text = message.content[0].text.strip()

        # Strip markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            response_text = "\n".join(lines).strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            if attempt < PARSE_MAX_RETRIES - 1:
                logger.warning(f"Parse attempt {attempt + 1} failed (invalid JSON), retrying...")
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": "That response was not valid JSON. Please return ONLY valid JSON "
                        "in the exact format specified. No markdown, no explanation.",
                    },
                ]
                continue
            logger.error(f"Failed to parse NLP response after {PARSE_MAX_RETRIES} attempts")
            return {"actions": [], "description": "Could not parse command", "error": True}

        # Validate the parsed result
        validated, issues = _validate_parsed_result(result, podcasts)

        if validated and validated.get("actions") and not issues:
            logger.info(f"Parsed command (attempt {attempt + 1}): {validated.get('description', '')}")
            return validated

        if issues and attempt < PARSE_MAX_RETRIES - 1:
            feedback = "Issues with your response:\n" + "\n".join(f"- {issue}" for issue in issues)
            feedback += f"\n\nOriginal command: \"{user_input}\"\nPlease fix these issues and return corrected JSON."
            logger.warning(f"Parse attempt {attempt + 1} had issues: {issues}, retrying...")
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": feedback},
            ]
            continue

        # Last attempt or no issues but no actions
        if validated:
            logger.info(f"Parsed command (attempt {attempt + 1}): {validated.get('description', '')}")
            return validated

    return {"actions": [], "description": "Could not parse command", "error": True}
