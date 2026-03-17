# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mediaverwerker is a CLI media processing tool that downloads podcast episodes, transcribes them via OpenAI Whisper API, generates Dutch-language articles via Claude API, and publishes full-text RSS feeds via GitHub Pages. It supports natural language commands in Dutch and English.

## Commands

```bash
# Install for development
source .venv/bin/activate
pip install -e .

# Run interactively (REPL with natural language input)
mediaverwerker

# Subcommands
mediaverwerker process                          # Process all new episodes from configured podcasts
mediaverwerker process -p Dwarkesh -t "AI"      # Specific podcast + topic
mediaverwerker clip video.mp4 --topic "AI"      # Clip segment from video
mediaverwerker feeds                            # Update RSS feeds
mediaverwerker run "download hard fork"         # One-off natural language command

# System dependencies
brew install ffmpeg  # Required for audio splitting and video clipping
```

## Architecture

The `mediaverwerker/` package is a modular CLI built with Typer. The old monolithic `podcast_processor.py` still exists but is superseded by the package.

### Data flow

```
RSS feed → download audio → transcribe (Whisper API) → generate article (Claude API) → save markdown → generate RSS feed → push to GitHub Pages
```

### Key modules

- **`cli.py`** — Entry point. Interactive REPL mode (no args) or subcommands. The REPL sends user input to `nlp.py` for parsing.
- **`nlp.py`** — Uses Claude Haiku to parse natural language into a JSON action plan with typed actions (`process_all`, `adhoc_episode`, `find_segment`, etc.).
- **`pipeline.py`** — Orchestration. Contains `run_full_pipeline()` for scheduled runs and `process_adhoc_episode()` for one-off requests. Handles parallel execution via `ThreadPoolExecutor`.
- **`config.py`** — Paths, API keys, cloud mode detection. `BASE_DIR` is the project root (parent of the package).
- **`state.py`** — JSON-file-based state: processed episode GUIDs, failed episodes with retry counts, podcast config.

### Task modules (`tasks/`)

- **`download.py`** — HTTP download for podcast audio + `search_podcast()` via iTunes Search API for ad-hoc requests.
- **`transcribe.py`** — Whisper API with auto-chunking for files >25MB. Supports `timestamps=True` for verbose_json with segment-level timing.
- **`article.py`** — Claude article generation with a detailed Dutch-language system prompt. Articles are biography-style chapters.
- **`segment.py`** — Claude Haiku finds topic-relevant segments in timestamped transcripts.
- **`clip.py`** — ffmpeg wrappers: clip media, generate SRT, burn subtitles.
- **`feeds.py`** — RSS 2.0 feed generation with `content:encoded` CDATA. Pushes to separate `podcast-feeds` repo.

### Two podcast modes

1. **Recurring** (`podcasts.json`): Configured podcasts processed daily via GitHub Actions cron. Limited to last 2 episodes per podcast.
2. **Ad-hoc**: Any podcast, searched via iTunes API. One-off download + transcribe, no RSS feed generation.

### Cloud deployment

GitHub Actions workflows in `.github/workflows/`:
- `process-podcasts.yml` — Daily at 09:00 CET + manual trigger
- `on-demand.yml` — Arbitrary commands via `workflow_dispatch`

State persistence across runs: `processed_episodes.json` is stored in the separate `podcast-feeds` repo and restored at workflow start.

## Key constraints

- **Never process full podcast archives.** Only the latest 2 episodes per configured podcast (`feed.entries[:2]`).
- API keys loaded from `.env` file (local) or environment variables (cloud). Required: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
- `GITHUB_TOKEN` / `FEEDS_GITHUB_TOKEN` needed for pushing feeds in cloud mode.
- Articles are always generated in Dutch, regardless of source language.
