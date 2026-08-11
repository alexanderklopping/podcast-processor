"""Tests for mediaverwerker.config secret loading."""

import importlib
import shutil
import subprocess


def _reload_config(monkeypatch, env=None, op_values=None, op_binary="/opt/homebrew/bin/op"):
    managed_env = [
        "GROQ_API_KEY",
        "IS_CLOUD",
        "MEDIAVERWERKER_1PASSWORD_GROQ_API_KEY_REF",
        "MEDIAVERWERKER_1PASSWORD_OPENAI_API_KEY_REF",
        "MEDIAVERWERKER_1PASSWORD_ITEM",
        "MEDIAVERWERKER_1PASSWORD_VAULT",
        "MEDIAVERWERKER_DISABLE_1PASSWORD",
        "OPENAI_API_KEY",
        "OPENAI_MODEL_BULK",
        "OPENAI_MODEL_EDITORIAL",
        "OPENAI_MODEL_POLISH",
        "OPENAI_MODEL_STRUCTURED",
        "OP_BIN",
        "RENDER",
        "TRANSCRIPTION_PROVIDER",
    ]
    for key in managed_env:
        monkeypatch.delenv(key, raising=False)

    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    calls = []

    def fake_which(name):
        if name == "op":
            return op_binary
        return None

    def fake_run(cmd, capture_output=False, text=False, check=False):
        calls.append(cmd)
        if cmd[:2] == [op_binary, "read"]:
            ref = cmd[2]
            if op_values and ref in op_values:
                return subprocess.CompletedProcess(cmd, 0, stdout=op_values[ref], stderr="")
            raise subprocess.CalledProcessError(1, cmd, "", "missing item")
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    import mediaverwerker.config as config

    return importlib.reload(config), calls


class TestOnePasswordFallback:
    def test_loads_required_secrets_from_1password(self, monkeypatch):
        config, calls = _reload_config(
            monkeypatch,
            op_values={
                "op://Private/podcast-processor/GROQ_API_KEY": "groq-secret\n",
                "op://Private/podcast-processor/OPENAI_API_KEY": "openai-secret\n",
            },
        )

        assert config.GROQ_API_KEY == "groq-secret"
        assert config.OPENAI_API_KEY == "openai-secret"
        assert calls == [
            ["/opt/homebrew/bin/op", "read", "op://Private/podcast-processor/OPENAI_API_KEY"],
            ["/opt/homebrew/bin/op", "read", "op://Private/podcast-processor/GROQ_API_KEY"],
        ]

    def test_environment_variable_wins_over_1password(self, monkeypatch):
        config, calls = _reload_config(
            monkeypatch,
            env={
                "GROQ_API_KEY": "env-groq",
                "OPENAI_API_KEY": "env-openai",
            },
        )

        assert config.GROQ_API_KEY == "env-groq"
        assert config.OPENAI_API_KEY == "env-openai"
        assert calls == []

    def test_validate_environment_logs_1password_guidance_when_cli_missing(self, monkeypatch, caplog):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.delenv("IS_CLOUD", raising=False)
        monkeypatch.delenv("MEDIAVERWERKER_DISABLE_1PASSWORD", raising=False)
        monkeypatch.setenv("OP_BIN", "/definitely-missing/op")
        monkeypatch.setattr(shutil, "which", lambda name: None)

        import mediaverwerker.config as config

        config = importlib.reload(config)
        caplog.set_level("INFO")

        assert config.validate_environment() is False
        assert "Missing required environment variables: GROQ_API_KEY, OPENAI_API_KEY" in caplog.text
        assert "1Password CLI not found" in caplog.text
        assert "podcast-processor" in caplog.text

    def test_transcription_upload_limit_stays_below_api_rejection_size(self, monkeypatch):
        config, _calls = _reload_config(
            monkeypatch,
            env={
                "GROQ_API_KEY": "env-groq",
                "OPENAI_API_KEY": "env-openai",
                "TRANSCRIPTION_PROVIDER": "groq",
            },
        )

        assert config.MAX_WHISPER_SIZE == 24 * 1024 * 1024

    def test_model_defaults_are_centralized(self, monkeypatch):
        config, _calls = _reload_config(
            monkeypatch,
            env={"GROQ_API_KEY": "env-groq", "OPENAI_API_KEY": "env-openai"},
        )

        assert config.OPENAI_MODEL_BULK == "gpt-5-nano"
        assert config.OPENAI_MODEL_STRUCTURED == "gpt-5.4-nano"
        assert config.OPENAI_MODEL_POLISH == "gpt-5.6-luna"
        assert config.OPENAI_MODEL_EDITORIAL == "gpt-5.6-terra"
        assert config.GROQ_TRANSCRIPTION_MODEL == "whisper-large-v3-turbo"
