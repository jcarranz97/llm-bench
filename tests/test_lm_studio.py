"""Tests for llm_bench.lm_studio HTTP client (mocked urlopen, no network)."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from llm_bench import lm_studio as lm_studio_mod
from llm_bench.lm_studio import LMStudioClient, LMStudioError


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | list[Any], status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _install_fake_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> list[tuple[str, str | None]]:
    """Replace urllib.request.urlopen. Returns a log of (url, body) tuples."""
    calls: list[tuple[str, str | None]] = []

    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        url = req.full_url if hasattr(req, "full_url") else str(req)
        body = req.data.decode("utf-8") if getattr(req, "data", None) else None
        calls.append((url, body))
        return handler(url, body)

    monkeypatch.setattr(lm_studio_mod.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_list_models_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """LM Studio 2026 shape: {'models': [...]} with `key` + `loaded_instances`."""

    entry = {
        "type": "llm",
        "key": "qwen/qwen3.6-35b-a3b",
        "loaded_instances": [{"id": "qwen/qwen3.6-35b-a3b"}],
    }

    def handler(url: str, body: str | None) -> _FakeResponse:
        assert url.endswith("/api/v1/models")
        return _FakeResponse({"models": [entry]})

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient("http://localhost:1234")
    assert client.list_models() == [entry]


def test_list_models_falls_back_to_v0(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None):  # type: ignore[no-untyped-def]
        if url.endswith("/api/v1/models"):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))  # type: ignore[arg-type]
        assert url.endswith("/api/v0/models")
        return _FakeResponse({"data": [{"id": "old-model", "type": "llm"}]})

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient()
    assert client.list_models() == [{"id": "old-model", "type": "llm"}]


def test_list_models_both_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError("connection refused")

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient()
    with pytest.raises(LMStudioError) as exc:
        client.list_models()
    assert "Could not list models" in str(exc.value)


def test_loaded_model_ids_v1_uses_loaded_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 shape: a model is loaded iff `loaded_instances` is non-empty."""

    def handler(url: str, body: str | None) -> _FakeResponse:
        return _FakeResponse(
            {
                "models": [
                    {
                        "type": "llm",
                        "key": "qwen/qwen3-4b",
                        "loaded_instances": [{"id": "qwen/qwen3-4b"}],
                    },
                    {"type": "llm", "key": "google/gemma-3-4b", "loaded_instances": []},
                    {"type": "embedding", "key": "nomic", "loaded_instances": [{"id": "nomic"}]},
                ]
            }
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient()
    # Embeddings excluded; unloaded LLM excluded; loaded LLM kept.
    assert client.loaded_model_ids() == ["qwen/qwen3-4b"]


def test_all_llm_ids_returns_loaded_and_unloaded_llms(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None) -> _FakeResponse:
        return _FakeResponse(
            {
                "models": [
                    {
                        "type": "llm",
                        "key": "loaded-llm",
                        "loaded_instances": [{"id": "loaded-llm"}],
                    },
                    {"type": "llm", "key": "cold-llm", "loaded_instances": []},
                    {"type": "embedding", "key": "embed", "loaded_instances": []},
                ]
            }
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient()
    assert client.all_llm_ids() == ["loaded-llm", "cold-llm"]


def test_chat_posts_correct_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_body: list[dict[str, Any]] = []

    def handler(url: str, body: str | None) -> _FakeResponse:
        assert url.endswith("/api/v1/chat")
        assert body is not None
        captured_body.append(json.loads(body))
        return _FakeResponse(
            {
                "stats": {
                    "tokens_per_second": 42.0,
                    "time_to_first_token_seconds": 0.1,
                    "input_tokens": 10,
                    "total_output_tokens": 5,
                },
                "output": [{"type": "message", "content": "blue"}],
            }
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient()
    resp = client.chat(
        model="qwen/qwen3-4b",
        system="Reply concisely.",
        user_input="Hi",
        max_output_tokens=1,
    )
    assert resp["stats"]["tokens_per_second"] == 42.0
    # Only max_output_tokens is sent — LM Studio rejects unknown keys, including max_tokens.
    assert captured_body[0] == {
        "model": "qwen/qwen3-4b",
        "system_prompt": "Reply concisely.",
        "input": "Hi",
        "max_output_tokens": 1,
    }


def test_chat_propagates_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            url,
            400,
            "Bad Request",
            {},  # type: ignore[arg-type]
            io.BytesIO(
                b'{"error": {"message": "Unrecognized key(s) in object: \\"max_tokens\\""}}'
            ),
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LMStudioClient()
    with pytest.raises(LMStudioError) as exc:
        client.chat(model="x", system="s", user_input="u")
    msg = str(exc.value)
    assert "HTTP 400" in msg
    # The structured error.message field is extracted, so the raw JSON braces are gone.
    assert "Unrecognized key" in msg
    assert "\n" not in msg  # error must be one line
