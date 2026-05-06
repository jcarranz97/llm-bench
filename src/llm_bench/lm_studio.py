"""HTTP client for LM Studio's local server (Responses-style /api/v1)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, cast


class LMStudioError(RuntimeError):
    """Raised when the LM Studio server cannot be reached or returns an error."""


class LMStudioClient:
    """
    Thin client for LM Studio's local HTTP API.

    Defaults to the Responses-style `/api/v1/chat` endpoint (matches the curl shape
    `{model, system_prompt, input}` → `{output, stats}`). Falls back to `/api/v0/models`
    for model listing on older builds.
    """

    def __init__(self, base_url: str = "http://localhost:1234", timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Internals ──────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            # Server bodies are JSON like {"error": {"message": "..."}} — flatten to one line.
            msg = detail
            try:
                parsed_err: Any = json.loads(detail)
                if isinstance(parsed_err, dict):
                    parsed_err_dict = cast(dict[str, Any], parsed_err)
                    err = parsed_err_dict.get("error")
                    if isinstance(err, dict) and isinstance(
                        cast(dict[str, Any], err).get("message"), str
                    ):
                        msg = cast(str, cast(dict[str, Any], err)["message"])
                    elif isinstance(err, str):
                        msg = err
            except (json.JSONDecodeError, ValueError):
                pass
            msg = " ".join(msg.split())  # collapse whitespace
            raise LMStudioError(f"HTTP {e.code}: {msg[:160]}") from e
        except urllib.error.URLError as e:
            raise LMStudioError(f"Cannot reach {url}: {e.reason}") from e
        try:
            parsed: Any = json.loads(payload) if payload else {}
        except json.JSONDecodeError as e:
            raise LMStudioError(f"Non-JSON response from {url}: {payload[:200]}") from e
        if not isinstance(parsed, dict):
            raise LMStudioError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return cast(dict[str, Any], parsed)

    # ── Public API ─────────────────────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        """
        Return the model array from LM Studio. Handles both the 2026 `/api/v1/models`
        shape (`{"models": [...]}` with `key` and `loaded_instances`) and older
        `/api/v0/models` (`{"data": [...]}` with `id` and `state`).
        """
        for path in ("/api/v1/models", "/api/v0/models"):
            try:
                resp = self._request("GET", path)
            except LMStudioError:
                continue
            for key in ("models", "data"):
                arr = resp.get(key)
                if isinstance(arr, list):
                    return cast(list[dict[str, Any]], arr)
        raise LMStudioError(
            f"Could not list models — neither /api/v1/models nor /api/v0/models worked at "
            f"{self.base_url}. Is LM Studio running with the local server enabled?"
        )

    def server_version(self) -> str:
        """Best-effort server identifier. Returns 'lm-studio' if no version is exposed."""
        try:
            models = self.list_models()
        except LMStudioError:
            return "lm-studio"
        if models:
            first = models[0]
            arch = first.get("arch") or first.get("architecture")
            if arch:
                return f"lm-studio (arch={arch})"
        return "lm-studio"

    @staticmethod
    def model_id(entry: dict[str, Any]) -> str:
        """Extract a chat-usable identifier from a model list entry (v1 or v0 shape)."""
        instances = entry.get("loaded_instances")
        if isinstance(instances, list) and instances:
            first = cast(list[Any], instances)[0]
            if isinstance(first, dict):
                inst_id = cast(dict[str, Any], first).get("id")
                if inst_id:
                    return str(inst_id)
        return str(entry.get("id") or entry.get("key") or entry.get("model") or "")

    @staticmethod
    def is_loaded(entry: dict[str, Any]) -> bool:
        """True if a model list entry represents a currently-loaded instance."""
        instances = entry.get("loaded_instances")
        if isinstance(instances, list):
            return len(cast(list[Any], instances)) > 0
        # Fallback for /api/v0 shape
        state = entry.get("state", "")
        return state in ("loaded", "")

    def loaded_model_ids(self) -> list[str]:
        """Return identifiers of currently-loaded LLM instances."""
        ids: list[str] = []
        for m in self.list_models():
            if m.get("type") not in (None, "llm"):
                continue
            if not self.is_loaded(m):
                continue
            mid = self.model_id(m)
            if mid:
                ids.append(mid)
        return ids

    def all_llm_ids(self) -> list[str]:
        """Return identifiers of every LLM (loaded or not). Embeddings/VLMs excluded."""
        ids: list[str] = []
        for m in self.list_models():
            if m.get("type") not in (None, "llm"):
                continue
            mid = self.model_id(m)
            if mid:
                ids.append(mid)
        return ids

    def chat(
        self,
        model: str,
        system: str,
        user_input: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        POST `/api/v1/chat` (Responses-style). Returns the parsed JSON response,
        which includes a `stats` object with `tokens_per_second`,
        `time_to_first_token_seconds`, `input_tokens`, `total_output_tokens`.
        """
        body: dict[str, Any] = {
            "model": model,
            "system_prompt": system,
            "input": user_input,
        }
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        return self._request("POST", "/api/v1/chat", body)
