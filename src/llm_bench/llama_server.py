"""HTTP client for llama.cpp's `llama-server` (native /completion endpoint)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, cast


class LlamaServerError(RuntimeError):
    """Raised when the llama-server cannot be reached or returns an error."""


class LlamaServerClient:
    """
    Thin client for llama.cpp's `llama-server`.

    Uses the native `POST /completion` endpoint, which returns a `timings` block
    with both prompt-processing and generation throughput — no need to time
    requests ourselves or compute speeds from token counts.
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 600.0) -> None:
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
                    elif isinstance(parsed_err_dict.get("message"), str):
                        msg = cast(str, parsed_err_dict["message"])
            except (json.JSONDecodeError, ValueError):
                pass
            msg = " ".join(msg.split())
            raise LlamaServerError(f"HTTP {e.code}: {msg[:160]}") from e
        except urllib.error.URLError as e:
            raise LlamaServerError(f"Cannot reach {url}: {e.reason}") from e
        try:
            parsed: Any = json.loads(payload) if payload else {}
        except json.JSONDecodeError as e:
            raise LlamaServerError(f"Non-JSON response from {url}: {payload[:200]}") from e
        if not isinstance(parsed, dict):
            raise LlamaServerError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return cast(dict[str, Any], parsed)

    # ── Public API ─────────────────────────────────────────────────────────

    def props(self) -> dict[str, Any]:
        """
        GET /props — server info. Includes `model_path`, `build_info`, etc.
        Some llama-server versions don't expose this; callers should handle errors.
        """
        return self._request("GET", "/props")

    def is_reachable(self) -> bool:
        """Connectivity probe: try /health first, fall back to /props (older builds)."""
        for probe_path in ("/health", "/props"):
            try:
                self._request("GET", probe_path)
                return True
            except LlamaServerError:
                continue
        return False

    def server_version(self) -> str:
        """Best-effort server identifier from /props build_info. Falls back to 'llama-server'."""
        try:
            p = self.props()
        except LlamaServerError:
            return "llama-server"
        build = p.get("build_info") or p.get("build")
        if isinstance(build, str) and build:
            return f"llama-server ({build})"
        return "llama-server"

    def devices(self) -> list[dict[str, Any]] | None:
        """Return the `devices` array from /props if the server exposes it.

        Newer llama.cpp builds populate this with one entry per device the
        runtime sees; older builds omit it. Returns `None` when the field is
        absent or /props is unreachable, so callers can warn-and-proceed.
        """
        try:
            p = self.props()
        except LlamaServerError:
            return None
        raw = p.get("devices")
        if isinstance(raw, list):
            return [cast(dict[str, Any], d) for d in cast(list[Any], raw) if isinstance(d, dict)]
        return None

    def model_id(self) -> str:
        """
        Return a friendly identifier for the loaded model. Tries /props's
        `model_alias` / `model_path`, falling back to '/v1/models', and finally
        a generic 'llama-server-model' so probes can still attribute results.
        """
        try:
            p = self.props()
            for key in ("model_alias", "model"):
                v = p.get(key)
                if isinstance(v, str) and v:
                    return v
            mp = p.get("model_path")
            if isinstance(mp, str) and mp:
                return mp.rsplit("/", 1)[-1].removesuffix(".gguf")
        except LlamaServerError:
            pass
        try:
            v1 = self._request("GET", "/v1/models")
            data = v1.get("data")
            if isinstance(data, list) and data:
                first = cast(list[Any], data)[0]
                if isinstance(first, dict):
                    first_id = cast(dict[str, Any], first).get("id")
                    if isinstance(first_id, str) and first_id:
                        return first_id
        except LlamaServerError:
            pass
        return "llama-server-model"

    def completion(
        self,
        prompt: str,
        n_predict: int,
        temperature: float = 0.0,
        cache_prompt: bool = False,
    ) -> dict[str, Any]:
        """
        POST /completion. Returns the parsed JSON response, which includes a
        `timings` block: `prompt_per_second`, `predicted_per_second`,
        `prompt_n`, `predicted_n`, etc.

        `cache_prompt=False` is set so that repeated probes don't get free
        prompt-processing from llama-server's prefix cache, which would skew
        the pp metric to infinity.
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": cache_prompt,
            "stream": False,
        }
        return self._request("POST", "/completion", body)
