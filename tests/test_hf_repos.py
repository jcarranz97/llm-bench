"""Verify every HuggingFace repo in the bundled model YAML files is reachable.

Run with:  PYTHONPATH="" uv run pytest tests/test_hf_repos.py --network -v
"""

import urllib.error
import urllib.request
from typing import Any

import pytest

from llm_bench.models import _bundled_profiles  # pyright: ignore[reportPrivateUsage]

_HF_API = "https://huggingface.co/api/models/{repo}"

# Collect (model_name, yaml_profile, hf_repo) for every entry in every bundled profile.
# Deduplicate by hf_repo so the same repo appearing in multiple profiles only runs once.
_seen: set[str] = set()
_REPO_PARAMS: list[Any] = []
for _profile in _bundled_profiles():
    for _model in _profile.models:
        if not _model.hf_repo or _model.hf_repo in _seen:
            continue
        _seen.add(_model.hf_repo)
        _REPO_PARAMS.append(
            pytest.param(
                _model.hf_repo,
                _model.name,
                _profile.profile,
                id=_model.hf_repo,
            )
        )


@pytest.mark.network
@pytest.mark.parametrize("hf_repo,model_name,profile", _REPO_PARAMS)
def test_hf_repo_exists(hf_repo: str, model_name: str, profile: str) -> None:
    """Assert that the HuggingFace repo responds with HTTP 200."""
    url = _HF_API.format(repo=hf_repo)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code

    assert status == 200, (
        f"[{profile}] '{model_name}' — repo '{hf_repo}' returned HTTP {status}. "
        "The repo may have been renamed or removed on HuggingFace."
    )
