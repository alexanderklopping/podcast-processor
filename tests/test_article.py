"""Tests for article generation helper calls."""

import json

from mediaverwerker.tasks import article


def test_score_article_uses_valid_thinking_budget(monkeypatch):
    calls = []

    def fake_call_claude(client, system_prompt, user_prompt, max_tokens=48000, thinking_budget=10000):
        calls.append({"max_tokens": max_tokens, "thinking_budget": thinking_budget})
        return json.dumps(
            {
                "scores": {
                    "volledigheid": 8,
                    "nederlands": 8,
                    "narratief": 8,
                    "citaten": 8,
                    "leesbaarheid": 8,
                },
                "feedback": [],
            }
        )

    monkeypatch.setattr(article, "_call_claude", fake_call_claude)

    result = article._score_article(object(), "transcript text", "article text")

    assert result["gemiddelde"] == 8
    assert calls[0]["thinking_budget"] < calls[0]["max_tokens"]
