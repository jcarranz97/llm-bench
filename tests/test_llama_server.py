"""Tests for llm_bench.llama_server HTTP client (mocked urlopen, no network)."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from llm_bench import llama_server as llama_server_mod
from llm_bench.llama_server import LlamaServerClient, LlamaServerError


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | list[Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _install_fake_urlopen(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        url = req.full_url if hasattr(req, "full_url") else str(req)
        body = req.data.decode("utf-8") if getattr(req, "data", None) else None
        return handler(url, body)

    monkeypatch.setattr(llama_server_mod.urllib.request, "urlopen", fake_urlopen)


def test_props_returns_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None) -> _FakeResponse:
        assert url.endswith("/props")
        return _FakeResponse(
            {"model_path": "/srv/models/Qwen2.5-7B.Q4_K_M.gguf", "build_info": "b4202"}
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient("http://localhost:8080")
    assert client.props()["build_info"] == "b4202"


def test_server_version_includes_build_info(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None) -> _FakeResponse:
        return _FakeResponse({"build_info": "b4202"})

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    assert client.server_version() == "llama-server (b4202)"


def test_server_version_falls_back_when_props_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(url: str, body: str | None):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError("connection refused")

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    assert client.server_version() == "llama-server"


def test_model_id_from_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None) -> _FakeResponse:
        return _FakeResponse({"model_alias": "qwen2.5-7b"})

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    assert client.model_id() == "qwen2.5-7b"


def test_model_id_from_model_path_basename(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None) -> _FakeResponse:
        return _FakeResponse({"model_path": "/srv/models/Qwen2.5-7B.Q4_K_M.gguf"})

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    assert client.model_id() == "Qwen2.5-7B.Q4_K_M"


def test_model_id_falls_back_to_v1_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None) -> _FakeResponse:
        if url.endswith("/props"):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))  # type: ignore[arg-type]
        assert url.endswith("/v1/models")
        return _FakeResponse({"data": [{"id": "models/qwen.gguf"}]})

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    assert client.model_id() == "models/qwen.gguf"


def test_completion_posts_correct_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def handler(url: str, body: str | None) -> _FakeResponse:
        assert url.endswith("/completion")
        assert body is not None
        captured.append(json.loads(body))
        return _FakeResponse(
            {
                "content": "ok",
                "timings": {
                    "prompt_n": 100,
                    "prompt_per_second": 250.0,
                    "predicted_n": 200,
                    "predicted_per_second": 35.0,
                },
            }
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    resp = client.completion(prompt="hi", n_predict=200)
    assert resp["timings"]["predicted_per_second"] == 35.0
    sent = captured[0]
    assert sent["prompt"] == "hi"
    assert sent["n_predict"] == 200
    # Caching is disabled so repeated probes don't free-ride on the prefix cache.
    assert sent["cache_prompt"] is False
    assert sent["temperature"] == 0.0
    assert sent["stream"] is False


def test_completion_propagates_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            url,
            500,
            "Server Error",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": {"message": "out of memory"}}'),
        )

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient()
    with pytest.raises(LlamaServerError) as exc:
        client.completion(prompt="x", n_predict=1)
    msg = str(exc.value)
    assert "HTTP 500" in msg
    assert "out of memory" in msg
    assert "\n" not in msg


def test_completion_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url: str, body: str | None):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError("connection refused")

    _install_fake_urlopen(monkeypatch, handler)
    client = LlamaServerClient("http://nope.invalid:8080")
    with pytest.raises(LlamaServerError) as exc:
        client.completion(prompt="x", n_predict=1)
    assert "Cannot reach" in str(exc.value)
