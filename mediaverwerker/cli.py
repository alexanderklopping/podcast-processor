"""Mediaverwerker CLI - media processing tool."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import init, validate_environment
from .util import setup_logging
from .util import is_url
from .nlp import parse_command
from .pipeline import (
    run_full_pipeline,
    process_and_extract,
    process_adhoc_episode,
    process_individual_url,
    find_episode_by_name_and_date,
    process_episode,
    process_url,
)
from .tasks.feeds import update_all_rss_feeds, push_feeds_to_github
from .tasks.clip import clip_media, generate_srt, burn_subtitles
from .tasks.transcribe import transcribe_audio
from .tasks.segment import find_segment

app = typer.Typer(
    name="mediaverwerker",
    help="Mediaverwerker - CLI media processing tool.",
    no_args_is_help=False,
)
logger = logging.getLogger("mediaverwerker")


def _init():
    """Initialize environment."""
    setup_logging()
    init()


@app.command("process")
def cmd_process(
    podcast: Optional[str] = typer.Option(None, "--podcast", "-p", help="Podcast name"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Individual media URL"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Episode date (e.g., 'yesterday', '2026-03-12')"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic to extract"),
    output: str = typer.Option("article", "--output", "-o", help="Output format: transcript, article, clip"),
):
    """Process podcast episodes. Without --podcast, processes all new episodes."""
    _init()

    if not validate_environment():
        raise typer.Exit(1)

    if url:
        result = process_individual_url(url, topic=topic, output_format=output)
        if result.get("error"):
            typer.echo(f"Error: {result['error']}")
            raise typer.Exit(1)
        if result.get("segment_text"):
            typer.echo(f"\n--- Segment over '{topic}' ---")
            typer.echo(result["segment_text"])
            typer.echo("---")
        elif result.get("already_processed"):
            typer.echo(f"Al aanwezig in feed: {result.get('episode', {}).get('title', '?')}")
        else:
            typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")
        if result.get("feed_url"):
            typer.echo(f"Feed: {result['feed_url']}")
    elif podcast:
        result = process_and_extract(podcast, date=date, topic=topic, output_format=output)
        if "error" in result:
            typer.echo(f"Error: {result['error']}")
            raise typer.Exit(1)

        if topic and result.get("segment_text"):
            typer.echo(f"\n--- Segment over '{topic}' ---")
            typer.echo(result["segment_text"])
            typer.echo("---")
        elif result.get("article"):
            typer.echo("Article generated and saved.")
        elif result.get("transcript"):
            typer.echo(f"Transcript ({len(result['transcript'])} chars) saved.")
    else:
        run_full_pipeline()


@app.command("clip")
def cmd_clip(
    input_file: Path = typer.Argument(..., help="Path to video/audio file"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic to find and clip"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    subtitles: bool = typer.Option(True, "--subtitles/--no-subtitles", help="Generate SRT subtitles"),
    burn: bool = typer.Option(False, "--burn", help="Burn subtitles into video"),
    language: str = typer.Option("nl", "--language", "-l", help="Language code"),
):
    """Clip a segment from a video/audio file."""
    _init()

    if not input_file.exists():
        typer.echo(f"File not found: {input_file}")
        raise typer.Exit(1)

    out_dir = output_dir or input_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_file.stem

    # Transcribe with timestamps
    typer.echo(f"Transcribing: {input_file.name}")
    result = transcribe_audio(input_file, language=language, timestamps=True)
    segments = result["segments"]

    if topic:
        typer.echo(f"Finding segment about: {topic}")
        segment = find_segment(segments, topic)
        if not segment:
            typer.echo(f"Topic '{topic}' not found in transcript")
            raise typer.Exit(1)

        start, end = segment["start"], segment["end"]
        matched_segments = segment["segments"]
        clip_name = f"{base_name}_{topic.replace(' ', '_')}"
    else:
        typer.echo("No topic specified, using full file")
        typer.echo(f"Transcript: {result['text'][:500]}...")
        return

    # Clip
    clip_path = out_dir / f"{clip_name}.mp4"
    clip_media(input_file, clip_path, start, end)

    if subtitles:
        srt_path = out_dir / f"{clip_name}.srt"
        generate_srt(matched_segments, offset=start, srt_path=srt_path)

        if burn:
            final_path = out_dir / f"{clip_name}_ondertiteld.mp4"
            burn_subtitles(clip_path, srt_path, final_path)
            typer.echo(f"Done: {final_path}")
        else:
            typer.echo(f"Done: {clip_path} + {srt_path}")
    else:
        typer.echo(f"Done: {clip_path}")


@app.command("feeds")
def cmd_feeds():
    """Update and push RSS feeds."""
    _init()
    update_all_rss_feeds()
    push_feeds_to_github()
    typer.echo("Feeds updated.")


@app.command("run")
def cmd_run(
    command: str = typer.Argument(..., help="Natural language command"),
):
    """Execute a natural language command.

    Examples:
        mediaverwerker run "download hard fork van gisteren en transcript over Anthropic"
        mediaverwerker run "verwerk alle nieuwe afleveringen"
    """
    _init()

    if not validate_environment():
        raise typer.Exit(1)

    typer.echo(f"Parsing: {command}")

    if is_url(command):
        result = process_individual_url(command)
        if result.get("error"):
            typer.echo(f"Error: {result['error']}")
            raise typer.Exit(1)
        typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")
        if result.get("feed_url"):
            typer.echo(f"Feed: {result['feed_url']}")
        return

    parsed = parse_command(command)

    if parsed.get("error"):
        typer.echo(f"Could not understand command: {command}")
        raise typer.Exit(1)

    typer.echo(f"Plan: {parsed.get('description', '')}")

    for action in parsed.get("actions", []):
        action_type = action.get("type")

        if action_type == "process_all":
            run_full_pipeline()

        elif action_type == "process_episode":
            episode = find_episode_by_name_and_date(action.get("podcast", ""), action.get("date"))
            if episode:
                process_episode(episode)
            else:
                typer.echo(f"Episode not found: {action}")

        elif action_type == "process_url":
            result = process_individual_url(
                url=action.get("url", ""),
                topic=action.get("topic"),
                output_format=action.get("output", "article"),
                output_dir=action.get("output_dir"),
            )
            if result.get("error"):
                typer.echo(f"Error: {result['error']}")
            elif result.get("already_processed"):
                typer.echo(f"Al aanwezig in feed: {result.get('episode', {}).get('title', '?')}")
            else:
                typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")
            if result.get("feed_url"):
                typer.echo(f"Feed: {result['feed_url']}")

        elif action_type == "find_segment":
            result = process_and_extract(
                podcast_name=action.get("podcast", ""),
                date=action.get("date"),
                topic=action.get("topic"),
                output_format=action.get("output", "transcript"),
            )
            if result.get("segment_text"):
                typer.echo(f"\n--- Segment over '{action.get('topic')}' ---")
                typer.echo(result["segment_text"])
                typer.echo("---")
            elif result.get("transcript"):
                typer.echo(f"Full transcript ({len(result['transcript'])} chars)")
            elif result.get("error"):
                typer.echo(f"Error: {result['error']}")

        elif action_type == "adhoc_episode":
            result = process_adhoc_episode(
                podcast_query=action.get("podcast_query", ""),
                date=action.get("date"),
                topic=action.get("topic"),
                output_format=action.get("output", "transcript"),
                output_dir=action.get("output_dir"),
            )
            if result.get("error"):
                typer.echo(f"Error: {result['error']}")
            elif result.get("output_files"):
                typer.echo(f"Bestanden opgeslagen in {result['output_dir']}:")
                for f in result["output_files"]:
                    typer.echo(f"  {f}")
            else:
                typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")

        elif action_type == "feeds_update":
            update_all_rss_feeds()
            push_feeds_to_github()
            typer.echo("Feeds updated.")

        elif action_type == "process_url":
            result = process_url(
                url=action.get("url", ""),
                language=action.get("language", "en"),
            )
            if result.get("article_path"):
                typer.echo(f"Article saved: {result['article_path']}")

        else:
            typer.echo(f"Unknown action: {action_type}")


def _execute_nl(command):
    """Parse and execute a natural language command."""
    if not validate_environment():
        return

    if is_url(command):
        result = process_individual_url(command)
        if result.get("error"):
            typer.echo(f"Error: {result['error']}")
        else:
            typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")
            if result.get("feed_url"):
                typer.echo(f"Feed: {result['feed_url']}")
        return

    typer.echo(f"\nParsing...")
    parsed = parse_command(command)

    if parsed.get("error"):
        typer.echo(f"Niet begrepen: {command}")
        return

    typer.echo(f"Plan: {parsed.get('description', '')}\n")

    for action in parsed.get("actions", []):
        action_type = action.get("type")

        if action_type == "process_all":
            run_full_pipeline()

        elif action_type == "process_episode":
            episode = find_episode_by_name_and_date(action.get("podcast", ""), action.get("date"))
            if episode:
                process_episode(episode)
            else:
                typer.echo(f"Episode niet gevonden: {action}")

        elif action_type == "process_url":
            result = process_individual_url(
                url=action.get("url", ""),
                topic=action.get("topic"),
                output_format=action.get("output", "article"),
                output_dir=action.get("output_dir"),
            )
            if result.get("error"):
                typer.echo(f"Error: {result['error']}")
            elif result.get("already_processed"):
                typer.echo(f"Al aanwezig in feed: {result.get('episode', {}).get('title', '?')}")
            else:
                typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")
            if result.get("feed_url"):
                typer.echo(f"Feed: {result['feed_url']}")

        elif action_type == "find_segment":
            result = process_and_extract(
                podcast_name=action.get("podcast", ""),
                date=action.get("date"),
                topic=action.get("topic"),
                output_format=action.get("output", "transcript"),
            )
            if result.get("segment_text"):
                typer.echo(f"\n--- Segment over '{action.get('topic')}' ---")
                typer.echo(result["segment_text"])
                typer.echo("---")
            elif result.get("transcript"):
                typer.echo(f"Transcript ({len(result['transcript'])} chars)")
            elif result.get("error"):
                typer.echo(f"Error: {result['error']}")

        elif action_type == "adhoc_episode":
            result = process_adhoc_episode(
                podcast_query=action.get("podcast_query", ""),
                date=action.get("date"),
                topic=action.get("topic"),
                output_format=action.get("output", "transcript"),
                output_dir=action.get("output_dir"),
            )
            if result.get("error"):
                typer.echo(f"Error: {result['error']}")
            elif result.get("segment_text"):
                typer.echo(f"\n--- Segment over '{action.get('topic')}' ---")
                typer.echo(result["segment_text"])
                typer.echo("---")
            elif result.get("output_files"):
                typer.echo(f"Bestanden opgeslagen in {result['output_dir']}:")
                for f in result["output_files"]:
                    typer.echo(f"  {f}")
            else:
                typer.echo(f"Klaar: {result.get('episode', {}).get('title', '?')}")

        elif action_type == "feeds_update":
            update_all_rss_feeds()
            push_feeds_to_github()
            typer.echo("Feeds updated.")

        elif action_type == "process_url":
            result = process_url(
                url=action.get("url", ""),
                language=action.get("language", "en"),
            )
            if result.get("article_path"):
                typer.echo(f"Artikel opgeslagen: {result['article_path']}")

        else:
            typer.echo(f"Onbekende actie: {action_type}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Mediaverwerker - interactieve media processing tool.

    Start zonder argumenten voor interactieve modus, of gebruik subcommands.
    """
    if ctx.invoked_subcommand is not None:
        return

    _init()

    typer.echo("Mediaverwerker v0.1.0")
    typer.echo("Typ wat je wilt doen, of 'stop' om te stoppen.\n")

    while True:
        try:
            command = input("mediaverwerker> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nTot ziens!")
            break

        if not command:
            continue
        if command.lower() in ("stop", "quit", "exit", "q"):
            typer.echo("Tot ziens!")
            break
        if command.lower() == "help":
            typer.echo("Typ gewoon wat je wilt, bijvoorbeeld:")
            typer.echo('  "verwerk alle nieuwe afleveringen"')
            typer.echo('  "download dwarkesh van gisteren en transcript over Anthropic"')
            typer.echo('  "update de feeds"')
            typer.echo('  "stop" om te stoppen')
            continue

        _execute_nl(command)
        typer.echo("")


if __name__ == "__main__":
    app()
