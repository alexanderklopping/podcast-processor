"""Tests for centralized semantic AI routing."""

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mediaverwerker import ai


class FakeResponses:
    def __init__(self, output_text='{"ok": true}'):
        self.output_text = output_text
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            output_text=self.output_text,
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=4,
                total_tokens=16,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )


class FakeResponseStream:
    def __init__(self, request):
        self.request = request
        self.response = SimpleNamespace(output_text="streamed text", usage=None)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="streamed "),
                SimpleNamespace(type="response.output_text.delta", delta="text"),
            ]
        )

    def get_final_response(self):
        return self.response


class FakeStreamingResponses(FakeResponses):
    def stream(self, **kwargs):
        self.request = kwargs
        return FakeResponseStream(kwargs)


def test_generate_json_uses_structured_role_and_logs_usage(monkeypatch, caplog):
    responses = FakeResponses()
    monkeypatch.setattr(ai, "_client", lambda: SimpleNamespace(responses=responses))
    caplog.set_level(logging.INFO)

    result = ai.generate_json(
        task="test_task",
        instructions="Return JSON",
        input_text="input",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )

    assert result == {"ok": True}
    assert responses.request["model"] == ai.ROLE_MODELS["structured"]
    assert responses.request["metadata"] == {"task": "test_task", "role": "structured"}
    assert responses.request["text"]["format"]["type"] == "json_schema"
    assert '"input_tokens":12' in caplog.text


def test_generate_json_mentions_json_in_input_for_json_object_mode(monkeypatch):
    responses = FakeResponses()
    monkeypatch.setattr(ai, "_client", lambda: SimpleNamespace(responses=responses))

    result = ai.generate_json(
        task="command_parsing",
        instructions="Parse the command into a structured response.",
        input_text="USER:\nverwerk alle nieuwe afleveringen",
    )

    assert result == {"ok": True}
    assert responses.request["text"]["format"] == {"type": "json_object"}
    assert responses.request["input"].startswith("Return valid JSON.")


def test_model_for_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown AI role"):
        ai.model_for("unknown")


def test_generate_text_streams_editorial_output(monkeypatch):
    responses = FakeStreamingResponses()
    monkeypatch.setattr(ai, "_client", lambda: SimpleNamespace(responses=responses))

    result = ai.generate_text(
        task="podcast_article",
        role="editorial",
        instructions="Write",
        input_text="input",
        max_output_tokens=48000,
        reasoning_effort="medium",
        stream=True,
    )

    assert result == "streamed text"
    assert responses.request["model"] == ai.ROLE_MODELS["editorial"]
    assert responses.request["reasoning"] == {"effort": "medium"}


def test_task_modules_do_not_hardcode_model_ids():
    package_dir = Path(__file__).parents[1] / "mediaverwerker"
    pattern = re.compile(r"(?:claude|gemini|gpt|whisper)-[a-z0-9][a-z0-9.-]*")
    offenders = []
    for path in package_dir.rglob("*.py"):
        if path.name == "config.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(package_dir)))

    assert offenders == []
