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

CRITICAL: Write at the length specified in the user message. The reader wants the FULL story with ALL details, anecdotes, quotes, and insights preserved. Do NOT summarize or compress. If the transcript contains 10 interesting anecdotes, include all 10. If a speaker makes 8 distinct points, cover all 8 in depth. Longer is better than shorter — the reader chose to read this because they want the complete story, not a summary.

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


def _call_claude(client, system_prompt, user_prompt, max_tokens=32000, thinking_budget=16000):
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
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                article_parts.append(event.delta.text)
    return "".join(article_parts)


def _split_transcript(text, max_words=4000):
    """Split transcript into sections at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    sections = []
    current = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current:
            sections.append("\n\n".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        sections.append("\n\n".join(current))
    return sections


@retry_with_backoff(max_retries=3, delay=5)
def create_article(episode, transcript):
    """Convert transcript to article using Claude."""
    logger.info(f"Creating article for: {episode['title']}")

    text = transcript if isinstance(transcript, str) else transcript.get("text", "")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    word_count = len(text.split())

    if word_count <= SECTION_THRESHOLD:
        return _create_article_single(client, episode, text, word_count)
    else:
        return _create_article_sections(client, episode, text, word_count)


def _create_article_single(client, episode, text, word_count):
    """Generate article in a single call for shorter transcripts."""
    target_words = min(int(word_count * 0.85), max(1500, int(word_count * 0.67)))

    user_prompt = f"""Hier is een podcast aflevering om te transformeren naar een geschreven hoofdstuk:

**Titel:** {episode['title']}
**Gepubliceerd:** {episode['published']}
**Beschrijving:** {episode['description']}

**Lengte-richtlijn:** Dit transcript bevat {word_count} woorden. Schrijf een artikel van minimaal {target_words} woorden maar niet meer dan {int(word_count * 0.85)} woorden. Sla reclames, sponsorvermeldingen en promoties over. Bewaar alle inhoudelijke details, anekdotes en citaten. De lezer wil het volledige verhaal zonder compressie, maar geen opvulling.

---

**TRANSCRIPT:**

{text}

---

Transformeer dit transcript naar een compelling geschreven hoofdstuk volgens de instructies. Schrijf ALTIJD in het Nederlands, ook als het transcript in het Engels is."""

    article = _call_claude(client, PODCAST_TO_ARTICLE_SYSTEM_PROMPT, user_prompt)
    logger.info(f"Article created: {len(article.split())} words")
    return article


def _create_article_sections(client, episode, text, word_count):
    """Generate article in sections for long transcripts, then combine."""
    sections = _split_transcript(text)
    target_per_section = max(1500, int(len(sections[0].split()) * 0.67))
    total_target = max(1500, int(word_count * 0.67))

    logger.info(f"Long transcript ({word_count} words) — splitting into {len(sections)} sections")

    # Generate each section as a chapter part
    parts = []
    for i, section in enumerate(sections):
        section_words = len(section.split())
        section_target = max(1000, int(section_words * 0.75))
        is_first = i == 0
        is_last = i == len(sections) - 1

        section_prompt = f"""Hier is **deel {i + 1} van {len(sections)}** van een podcast aflevering.

**Titel:** {episode['title']}
**Gepubliceerd:** {episode['published']}
**Beschrijving:** {episode['description']}

**Context:** Het volledige transcript bevat {word_count} woorden. Dit deel bevat {section_words} woorden. Het doelartikel moet minimaal {total_target} woorden zijn in totaal.

**Instructie voor dit deel:** Schrijf minimaal {section_target} woorden voor dit deel. Sla reclames, sponsorvermeldingen en promoties over. Bewaar alle inhoudelijke details, anekdotes en citaten.
{"Begin met een pakkende titel en openingsscène." if is_first else "Ga naadloos verder waar het vorige deel eindigde. Geen nieuwe titel of inleiding."}
{"Eindig met een krachtige afsluiting die het hele verhaal samenbrengt." if is_last else "Eindig op een natuurlijk punt — het verhaal gaat verder in het volgende deel."}

---

**TRANSCRIPT (deel {i + 1}/{len(sections)}):**

{section}

---

Schrijf ALTIJD in het Nederlands, ook als het transcript in het Engels is."""

        logger.info(f"Generating section {i + 1}/{len(sections)} ({section_words} words)...")
        part = _call_claude(client, PODCAST_TO_ARTICLE_SYSTEM_PROMPT, section_prompt)
        parts.append(part)
        logger.info(f"Section {i + 1} done: {len(part.split())} words")

    article = "\n\n".join(parts)
    logger.info(f"Article created: {len(article.split())} words from {len(sections)} sections")
    return article


def save_article(episode, article):
    """Save article as Markdown."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    podcast_name = episode.get("podcast_name", "unknown")
    base_filename = f"{date_str}_{podcast_name}_{sanitize_filename(episode['title'])}"

    md_filepath = ARTICLES_DIR / f"{base_filename}.md"
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(article)
    logger.info(f"Markdown saved: {base_filename}.md")

    return md_filepath
