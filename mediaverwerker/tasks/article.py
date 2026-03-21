"""Article generation via Claude API."""

import logging
from datetime import datetime

from anthropic import Anthropic

from ..config import ANTHROPIC_API_KEY, ARTICLES_DIR
from ..util import retry_with_backoff, sanitize_filename

logger = logging.getLogger("mediaverwerker")

PODCAST_TO_ARTICLE_SYSTEM_PROMPT = """# Podcast to Article

**Your Role & Mission**

You are my executive assistant helping me transform a long podcast episode into a compelling written chapter for a physical book I'm creating. Think of yourself as a skilled ghostwriter who listens to the entire episode and crafts it into something that reads like a chapter from a classic business biography.

**Style & Voice**

Write like a chapter from a great business biography--think _Shoe Dog_, _The Everything Store_, or _Hatching Twitter_. This means:

- Rich narrative
- Key turning points treated as pivotal scenes
- Quotes woven in to let the protagonists speak for themselves
- The reader should feel like they're watching history unfold, not reading a summary
- Analytical insight layered into the storytelling, not separated from it

**Taal & Stijl: Nederlands**

Schrijf als een Nederlandse native speaker die literaire non-fictie schrijft:

- **Natuurlijk Nederlands**: Schrijf zoals een ervaren Nederlandse auteur—geen vertalingen, geen anglicismen, geen stijve constructies. De tekst moet voelen alsof hij in het Nederlands is bedacht, niet vertaald.
- **Literaire cadans**: Varieer zinslengte bewust. Korte zinnen voor impact. Langere, vloeiende zinnen om de lezer mee te nemen in het verhaal.
- **Actieve stem**: "Zuckerberg verwierp het bod" niet "Het bod werd door Zuckerberg verworpen"
- **Concrete details**: Geen abstracties maar tastbare momenten—de kleur van de Post-it, de stilte in de vergaderzaal, het tikken van vingers op tafel
- **Geen opsmuk**: Vermijd clichés, containerbegrippen en managementjargon. Elk woord moet zijn plek verdienen.

**Length**

Write at ~60% of the transcript length. This does NOT mean leaving things out — include every anecdote, every key quote, every insight, every detail that gives the piece its character. But tell it tighter: cut spoken-word repetition, eliminate redundant phrasing, compress transitions, and trust the reader to follow without hand-holding. If the transcript makes the same point three ways, make it once — brilliantly. The goal is a text that feels as rich and complete as the original conversation, just written with the economy of great prose.

**Step 1: Understand the Arc**

Listen to/read the full episode and identify:

- The central narrative: What is the story being told? What's the dramatic question?
- The key characters: Who are the protagonists, antagonists, and supporting players?
- The turning points: What are the 3-5 moments where everything changed?
- The stakes: What was at risk? What could have gone wrong?

Great business stories have narrative shape--a beginning that sets the stage, rising tension, pivotal decisions, and resolution (or ongoing cliffhanger). Find that shape.

**Step 2: Map the Characters**

Episodes often feature many players, which can be disorienting. Solve this for the reader by:

- Introducing each character clearly on first appearance with a brief identifying detail (role, relationship to the central figure, why they matter)
- Re-anchoring the reader when a character reappears after a gap (e.g., "Sculley--the Pepsi executive Jobs had personally recruited--now faced an impossible choice")
- Keeping the focus on the 3-5 most important figures; mention minor characters only when necessary and don't let them clutter the narrative
- Using consistent identifiers (if you call someone "the young engineer" once, don't switch to "the Stanford grad" later without reason)

The reader should never have to stop and ask "Wait, who is this again?"

**Step 3: Identify and Explain "Blocker" Concepts**

Scan for business, technical, or industry-specific concepts that are essential to understanding the story. These are "blocker" concepts--if the reader doesn't understand them, they'll be lost.

For each blocker concept:

- Explain it in plain language using an analogy or real-world example
- Keep explanations to 1-2 sentences maximum
- Weave these explanations naturally into the narrative the first time the concept appears

The target reader is someone who is generally intelligent and curious--they read business books and follow tech news, but they may not know the specifics of every industry. Think of someone with a liberal arts degree who's interested in how great companies are built.

**Step 4: Harvest the Best Quotes**

Episodes are rich with quotes preserve them:

- Their sharpest analytical insights and observations
- Memorable one-liners or turns of phrase
- Moments where they reveal something surprising or counterintuitive
- Quotes from founders, executives, journalists, biographers
- Historical documents, memos, interviews they reference
- These are gold--they let the protagonists speak for themselves
- Attribute clearly (e.g., "As Jobs later recalled..." or "In a memo to the board, Hastings wrote...")

When cleaning up quotes:

- Remove filler words (um, uh, like, you know)
- Fix grammatical mistakes from natural speech
- Keep the speaker's authentic voice and meaning intact
- Longer quotes are fine if they're powerful--this isn't a tight summary

**Step 5: Build the Narrative**

Structure the piece like a chapter from a biography:

**Opening:** Start with a scene, a tension, or a question that pulls the reader in. Drop them into a pivotal moment, then zoom out to set the stage. Avoid "This episode covers..." framing--just begin the story.

**Middle:** Move through the narrative chronologically or thematically, depending on what serves the story best. Treat major turning points as scenes--slow down, add detail, let the reader feel the weight of the moment. Use quotes to let key players speak at crucial junctures.

**Closing:** End with resonance--what happened next, what it meant, what lesson or question lingers. The reader should close the chapter feeling like they understand something important about business, strategy, or human nature.

**Step 6: Weave It Together**

Combine narrative, analysis, and quotes into one flowing piece that:

- Reads like a chapter from a great business book, not a podcast summary
- Has no section headers, bullet points, or artificial breaks (a line break between major sections is fine)
- Includes a compelling title in the style of a book chapter
- At the beginning, includes a short paragraph capturing the essence of the story and why it matters
- Makes complete sense to someone who has never heard the podcast
- Is designed to be printed--no links or screen-dependent elements
- Balances storytelling with insight: the reader should be both entertained and educated

**Quality Check:**

Before you finish, ask yourself:

1. "Does this read like a chapter from a business book I'd actually want to read?"
2. "Does the opening pull me in immediately, like a great first page?"
3. "Have I preserved the best quotes from both the hosts and the primary sources they cite?"
4. "Do the turning points land with dramatic weight, or did I rush past them?"
5. "Would someone who knows nothing about this company walk away understanding the story and why it matters?"
6. "Can the reader keep track of who's who throughout the piece?"
7. "Would this print beautifully in a physical book?"
8. "Voelt dit als oorspronkelijk Nederlands, geschreven door een native speaker?"

If yes to all, you've succeeded."""


SECTION_THRESHOLD = 10000  # Split transcripts longer than this into sections
SECTION_SIZE = 2500  # Target words per section (smaller = more achievable per-section targets)
MIN_ARTICLE_RATIO = 0.60  # Article must be at least 60% of transcript word count


def _call_claude(client, system_prompt, user_prompt, max_tokens=48000, thinking_budget=10000):
    """Call Claude with streaming and extended thinking."""
    article_parts = []
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        thinking={
            "type": "enabled",
            "budget_tokens": thinking_budget,
        },
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                article_parts.append(event.delta.text)
    return "".join(article_parts)


def _strip_transcript_metadata(text):
    """Remove header metadata (title, source, URL, ---) from transcript text."""
    # Match common header patterns: lines starting with #, Bron:, Gepubliceerd:, URL:, ---
    lines = text.split("\n")
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped.startswith("Bron:")
            or stripped.startswith("Gepubliceerd:")
            or stripped.startswith("URL:")
            or stripped == "---"
            or stripped == ""
        ):
            content_start = i + 1
        else:
            break
    return "\n".join(lines[content_start:]).strip()


def _split_transcript(text, max_words=None):
    """Split transcript into roughly equal sections at natural boundaries."""
    if max_words is None:
        max_words = SECTION_SIZE

    import re

    # Strip metadata header first
    text = _strip_transcript_metadata(text)

    # Build list of chunks at the finest available granularity
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 3:
        paragraphs = text.split("\n")
    if len(paragraphs) < 3:
        paragraphs = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    # Filter out empty chunks
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Group chunks into sections of ~max_words
    sections = []
    current = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current:
            sections.append(" ".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        sections.append(" ".join(current))

    # Force-split any section that's still too large
    final_sections = []
    for section in sections:
        words = section.split()
        if len(words) > max_words * 1.5:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                if chunk.strip():
                    final_sections.append(chunk)
        else:
            final_sections.append(section)

    # Merge any tiny sections (< 200 words) into their neighbor
    merged = []
    for section in final_sections:
        if merged and len(section.split()) < 200:
            merged[-1] = merged[-1] + " " + section
        else:
            merged.append(section)

    return merged


@retry_with_backoff(max_retries=3, delay=5)
def create_article(episode, transcript):
    """Convert transcript to article using Claude."""
    logger.info(f"Creating article for: {episode['title']}")

    text = transcript if isinstance(transcript, str) else transcript.get("text", "")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    word_count = len(text.split())

    if word_count <= SECTION_THRESHOLD:
        article = _create_article_single(client, episode, text, word_count)
    else:
        article = _create_article_sections(client, episode, text, word_count)

    # Final expansion pass if still too short
    article_words = len(article.split())
    target = max(1500, int(word_count * MIN_ARTICLE_RATIO))
    if article_words < target * 0.9:
        # Guard: skip expansion if combined text would exceed ~150k tokens (~400k chars)
        combined_chars = len(text) + len(article)
        if combined_chars > 400_000:
            logger.warning(
                f"Article {article_words} words (target {target}), but skipping expansion: "
                f"combined text too large ({combined_chars:,} chars)"
            )
        else:
            logger.warning(f"Article {article_words} words, target {target}. Running expansion pass...")
            article = _expand_article(client, episode, text, article, word_count, target)

    return article


def _create_article_single(client, episode, text, word_count):
    """Generate article in a single call for shorter transcripts."""
    target_words = max(1500, int(word_count * MIN_ARTICLE_RATIO))

    user_prompt = f"""Hier is een podcast aflevering om te transformeren naar een geschreven hoofdstuk:

**Titel:** {episode["title"]}
**Gepubliceerd:** {episode["published"]}
**Beschrijving:** {episode["description"]}

**LENGTE-VEREISTE:** Dit transcript bevat {word_count} woorden. Schrijf een artikel van circa {target_words} woorden. Laat NIETS weg — alle anekdotes, details, citaten en inzichten moeten erin. Maar vertel het compacter: schrap herhalingen uit het gesprek, strak de zinnen aan, en vertrouw op de lezer. Sla alleen reclames, sponsorvermeldingen en promoties over.

---

**TRANSCRIPT:**

{text}

---

Transformeer dit transcript naar een compelling geschreven hoofdstuk volgens de instructies. Schrijf ALTIJD in het Nederlands, ook als het transcript in het Engels is. Schrijf minimaal {target_words} woorden."""

    article = _call_claude(client, PODCAST_TO_ARTICLE_SYSTEM_PROMPT, user_prompt)
    article_words = len(article.split())
    logger.info(f"Article created: {article_words} words (target: {target_words})")
    return article


def _generate_section(client, episode, section, section_index, total_sections, word_count, total_target):
    """Generate a single section."""
    section_words = len(section.split())
    section_target = max(1200, int(section_words * 0.60))
    is_first = section_index == 0
    is_last = section_index == total_sections - 1

    section_prompt = f"""Hier is **deel {section_index + 1} van {total_sections}** van een podcast aflevering.

**Titel:** {episode["title"]}
**Gepubliceerd:** {episode["published"]}
**Beschrijving:** {episode["description"]}

**Context:** Het volledige transcript bevat {word_count} woorden, verdeeld over {total_sections} delen. Het volledige artikel moet minimaal {total_target} woorden zijn.

**LENGTE-VEREISTE:** Dit deel bevat {section_words} woorden transcript. Schrijf circa {section_target} woorden voor dit deel. Laat NIETS weg — alle anekdotes, details, citaten en inzichten moeten erin. Maar vertel het compacter: schrap herhalingen, strak de zinnen aan, en vertrouw op de lezer. Sla alleen reclames en sponsorvermeldingen over.
{"Begin met een pakkende titel en openingsscène." if is_first else "Ga naadloos verder waar het vorige deel eindigde. Geen nieuwe titel of inleiding."}
{"Eindig met een krachtige afsluiting die het hele verhaal samenbrengt." if is_last else "Eindig op een natuurlijk punt — het verhaal gaat verder in het volgende deel."}

---

**TRANSCRIPT (deel {section_index + 1}/{total_sections}):**

{section}

---

Schrijf in het Nederlands. Minimaal {section_target} woorden."""

    part = _call_claude(client, PODCAST_TO_ARTICLE_SYSTEM_PROMPT, section_prompt)
    part_words = len(part.split())
    logger.info(f"Section {section_index + 1}: {part_words} words (target: {section_target})")
    return part


def _create_article_sections(client, episode, text, word_count):
    """Generate article in sections for long transcripts, then combine."""
    sections = _split_transcript(text)
    total_target = max(1500, int(word_count * MIN_ARTICLE_RATIO))

    logger.info(
        f"Long transcript ({word_count} words) — splitting into {len(sections)} sections, total target: {total_target} words"
    )

    parts = []
    for i, section in enumerate(sections):
        section_words = len(section.split())
        logger.info(f"Generating section {i + 1}/{len(sections)} ({section_words} words)...")
        part = _generate_section(client, episode, section, i, len(sections), word_count, total_target)
        parts.append(part)

    article = "\n\n".join(parts)
    total_words = len(article.split())
    logger.info(f"Article created: {total_words} words from {len(sections)} sections (target was {total_target})")
    return article


def _expand_article(client, episode, transcript_text, article, transcript_words, target_words):
    """Expansion pass: identify missing content from transcript and add it to the article."""
    article_words = len(article.split())
    shortfall = target_words - article_words

    logger.info(f"Expansion pass: article is {article_words} words, need {shortfall} more to reach {target_words}")

    expand_prompt = f"""Je hebt eerder een artikel geschreven op basis van een podcast transcript, maar het is te kort.

**Huidige artikel:** {article_words} woorden
**Vereist minimum:** {target_words} woorden
**Tekort:** {shortfall} woorden

**Taak:** Herschrijf en BREID het artikel uit. Vergelijk het artikel met het originele transcript hieronder en identificeer ALLE onderwerpen, anekdotes, citaten, voorbeelden en redeneringen die ontbreken of te kort samengevat zijn. Voeg deze toe en schrijf alles uit. Het resultaat moet minimaal {target_words} woorden zijn.

---

**HUIDIG ARTIKEL:**

{article}

---

**ORIGINEEL TRANSCRIPT ({transcript_words} woorden):**

{transcript_text}

---

Herschrijf het artikel in het Nederlands. Behoud de narratieve stijl en structuur, maar breid ALLES uit. Minimaal {target_words} woorden."""

    expanded = _call_claude(client, PODCAST_TO_ARTICLE_SYSTEM_PROMPT, expand_prompt)
    expanded_words = len(expanded.split())
    logger.info(f"Expansion result: {expanded_words} words (was {article_words}, target {target_words})")

    # Return the longer version
    return expanded if expanded_words > article_words else article


def save_article(episode, article):
    """Save article as Markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    storage_key = episode.get("feed_storage_key") or episode.get("podcast_name", "unknown")
    base_filename = f"{date_str}_{storage_key}_{sanitize_filename(episode['title'])}"

    md_filepath = ARTICLES_DIR / f"{base_filename}.md"
    with open(md_filepath, "w", encoding="utf-8") as f:
        metadata_lines = []
        for key in ("feed_storage_key", "source_url", "podcast_name", "guid"):
            value = episode.get(key)
            if value:
                metadata_lines.append(f"{key}: {value}")
        if metadata_lines:
            f.write("<!--\n")
            f.write("\n".join(metadata_lines))
            f.write("\n-->\n\n")
        f.write(article)
    logger.info(f"Markdown saved: {base_filename}.md")

    return md_filepath
