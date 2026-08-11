"""Shared OpenAI text generation helpers with semantic model routing."""

import json
import logging
import time
from functools import lru_cache

from openai import OpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_BULK,
    OPENAI_MODEL_EDITORIAL,
    OPENAI_MODEL_POLISH,
    OPENAI_MODEL_STRUCTURED,
)

logger = logging.getLogger("mediaverwerker")

ROLE_MODELS = {
    "bulk": OPENAI_MODEL_BULK,
    "structured": OPENAI_MODEL_STRUCTURED,
    "polish": OPENAI_MODEL_POLISH,
    "editorial": OPENAI_MODEL_EDITORIAL,
}


@lru_cache(maxsize=1)
def _client():
    return OpenAI(api_key=OPENAI_API_KEY)


def model_for(role):
    """Return the configured model for a semantic workload role."""
    try:
        return ROLE_MODELS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown AI role: {role}") from exc


def _request_options(task, role, instructions, input_text, max_output_tokens, reasoning_effort):
    options = {
        "model": model_for(role),
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        "metadata": {"task": task, "role": role},
        "store": False,
    }
    if reasoning_effort:
        options["reasoning"] = {"effort": reasoning_effort}
    return options


def _log_result(task, role, model, started_at, response=None, error=None):
    payload = {
        "event": "ai_request",
        "task": task,
        "role": role,
        "model": model,
        "latency_ms": round((time.monotonic() - started_at) * 1000),
        "success": error is None,
    }
    usage = getattr(response, "usage", None)
    if usage:
        payload.update(
            {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        )
        output_details = getattr(usage, "output_tokens_details", None)
        if output_details:
            payload["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", None)
        input_details = getattr(usage, "input_tokens_details", None)
        if input_details:
            payload["cached_tokens"] = getattr(input_details, "cached_tokens", None)
    if error is not None:
        payload["error_type"] = type(error).__name__
    logger.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def generate_text(
    *,
    task,
    role,
    instructions,
    input_text,
    max_output_tokens,
    reasoning_effort=None,
    stream=False,
):
    """Generate free-form text for a named task."""
    model = model_for(role)
    started_at = time.monotonic()
    response = None
    try:
        options = _request_options(
            task,
            role,
            instructions,
            input_text,
            max_output_tokens,
            reasoning_effort,
        )
        if stream:
            chunks = []
            with _client().responses.stream(**options) as response_stream:
                for event in response_stream:
                    if event.type == "response.output_text.delta":
                        chunks.append(event.delta)
                response = response_stream.get_final_response()
            text = "".join(chunks) or response.output_text
        else:
            response = _client().responses.create(**options)
            text = response.output_text
        if not text or not text.strip():
            raise ValueError(f"OpenAI returned no text for task '{task}'")
        _log_result(task, role, model, started_at, response=response)
        return text.strip()
    except Exception as exc:
        _log_result(task, role, model, started_at, response=response, error=exc)
        raise


def generate_json(
    *,
    task,
    instructions,
    input_text,
    schema=None,
    max_output_tokens=2048,
    reasoning_effort="low",
):
    """Generate and parse JSON, optionally constrained by a strict JSON schema."""
    role = "structured"
    model = model_for(role)
    started_at = time.monotonic()
    response = None
    try:
        options = _request_options(
            task,
            role,
            instructions,
            input_text,
            max_output_tokens,
            reasoning_effort,
        )
        if schema:
            options["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": task.replace("-", "_")[:64],
                    "schema": schema,
                    "strict": True,
                }
            }
        else:
            options["text"] = {"format": {"type": "json_object"}}
            # The Responses API requires the input itself to mention JSON when
            # json_object mode is used. Instructions alone do not satisfy that
            # validation rule.
            options["input"] = f"Return valid JSON.\n\n{input_text}"
        response = _client().responses.create(**options)
        result = json.loads(response.output_text)
        _log_result(task, role, model, started_at, response=response)
        return result
    except Exception as exc:
        _log_result(task, role, model, started_at, response=response, error=exc)
        raise
