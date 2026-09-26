"""Local .env settings must reach requests without leaking into other providers."""

import json
import os
from pathlib import Path
import runpy

import dotenv
import httpx
import pytest

import config


def test_project_dotenv_is_loaded_without_mutating_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'export OPENAI_API_KEY="local-test-${DO_NOT_EXPAND}"\n'
        'OPENAI_BASE_URL=https://mock.invalid/v1 # local endpoint\n'
        'OPENAI_MODEL=test-model\nmax_tokens=65536\nthinking=enabled\n'
        'reasoning_effort=low\nstream=false\n', encoding="utf-8-sig")
    original_reader = dotenv.dotenv_values

    def read_project_file(path, **kwargs):
        assert Path(path) == config.LOCAL_LLM_FILE
        return original_reader(env_file, **kwargs)

    monkeypatch.setattr(dotenv, "dotenv_values", read_project_file)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-host-key")
    settings = runpy.run_path(config.__file__)
    assert settings["LLM_API_KEY"] == "local-test-${DO_NOT_EXPAND}"
    assert settings["LLM_BASE_URL"] == "https://mock.invalid/v1"
    assert settings["LLM_MODEL"] == "test-model"
    assert settings["OPENAI_MAX_TOKENS"] == "65536"
    assert settings["ANTHROPIC_API_KEY"] == settings["ANTHROPIC_BASE_URL"] == ""
    assert os.environ["OPENAI_API_KEY"] == "unrelated-host-key"


def test_openai_local_options_and_credentials_reach_request(monkeypatch):
    from brains import OpenAICompatBrain

    values = dict(LLM_BASE_URL="https://mock.invalid/v1", LLM_API_KEY="local-test-key",
                  LLM_MODEL="test-model", OPENAI_MAX_TOKENS="65536",
                  OPENAI_THINKING="enabled", OPENAI_REASONING_EFFORT="low",
                  OPENAI_STREAM="false")
    for name, value in values.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-host-key")
    captured = {}

    def respond(request):
        assert request.url.host == "mock.invalid"
        assert request.headers["authorization"] == "Bearer local-test-key"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    brain = OpenAICompatBrain(transport=httpx.MockTransport(respond))
    try:
        brain._request([{"role": "user", "content": "test"}], [])
    finally:
        brain._client.close()
    assert captured["model"] == "test-model"
    assert captured["max_tokens"] == 65536
    assert captured["thinking"] == {"type": "enabled"}
    assert captured["reasoning_effort"] == "low"
    assert captured["stream"] is False


@pytest.mark.parametrize(("setting", "value", "message"), [
    ("OPENAI_STREAM", "true", "stream=false"),
    ("OPENAI_MAX_TOKENS", "0", "max_tokens"),
    ("OPENAI_MAX_TOKENS", "not-a-number", "max_tokens"),
    ("OPENAI_THINKING", "invalid", "thinking"),
])
def test_unsupported_request_settings_fail_before_network(monkeypatch, setting, value, message):
    from brains import OpenAICompatBrain

    monkeypatch.setattr(config, "OPENAI_MAX_TOKENS", 8192)
    monkeypatch.setattr(config, "OPENAI_STREAM", "false")
    monkeypatch.setattr(config, "OPENAI_THINKING", "")
    monkeypatch.setattr(config, setting, value)
    with pytest.raises(ValueError, match=message):
        OpenAICompatBrain(base_url="https://mock.invalid", api_key="test", model="test")
