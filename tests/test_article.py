"""Tests for article generation helper calls."""

from mediaverwerker.tasks import article


def test_score_article_uses_structured_model(monkeypatch):
    calls = []

    def fake_generate_json(**kwargs):
        calls.append(kwargs)
        return {
            "scores": {
                "volledigheid": 8,
                "nederlands": 8,
                "narratief": 8,
                "citaten": 8,
                "leesbaarheid": 8,
            },
            "gemiddelde": 8,
            "feedback": [],
        }

    monkeypatch.setattr(article, "generate_json", fake_generate_json)

    result = article._score_article("transcript text", "article text")

    assert result["gemiddelde"] == 8
    assert calls[0]["task"] == "podcast_article_score"
    assert calls[0]["schema"] == article.ARTICLE_SCORE_SCHEMA
