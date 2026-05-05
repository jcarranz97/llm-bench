"""Click CLI: llm-bench run / sysinfo / results / models."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import click
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from llm_bench import __version__, storage
from llm_bench import models as model_registry
from llm_bench import sysinfo as sysinfo_mod
from llm_bench.llama_server import LlamaServerClient, LlamaServerError
from llm_bench.llama_server_runner import run_llama_server_benchmark
from llm_bench.lm_studio import LMStudioClient, LMStudioError
from llm_bench.lm_studio_runner import run_lm_studio_benchmark
from llm_bench.models import Model, ModelProfile
from llm_bench.parser import BenchResult, extract_json, parse_bench_output
from llm_bench.reporter import (
    build_compare_table,
    build_history_table,
    build_summary_table,
    build_sysinfo_panel,
    compute_scores,
)
from llm_bench.runner import get_llama_bench_version, run_benchmark

console = Console()

DEFAULT_LLAMA_BENCH = "/home/homelab/repos/llama.cpp/build/bin/llama-bench"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
DEFAULT_LLAMA_SERVER_URL = "http://localhost:8080"


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "")


def _models_from_csv(models_csv: str) -> list[Model]:
    ids = [m.strip() for m in models_csv.split(",") if m.strip()]
    return [Model(name=i, lm_studio_id=i) for i in ids]


def _ad_hoc_profile(models: list[Model], name: str = "ad-hoc") -> ModelProfile:
    return ModelProfile(
        profile=name,
        description="ad-hoc model list from --models",
        min_ram_gb=0,
        max_ram_gb=9999,
        models=models,
        source_path=None,
    )


# ── Root group ────────────────────────────────────────────────────────────────


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version")
def main() -> None:
    """llm-bench — real local LLM benchmarks.

    Auto-detects your hardware, picks the right model profile, runs
    llama-bench, caches results, and renders a ranked comparison table.

    Quick start:

    \b
        llm-bench sysinfo        # see what hardware was detected
        llm-bench run            # run benchmarks
        llm-bench results        # list past runs
    """


# ── llm-bench run ─────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--backend",
    type=click.Choice(["llama-bench", "lm-studio", "llama-server"], case_sensitive=False),
    default="llama-bench",
    show_default=True,
    help="Which backend to drive: local llama-bench binary, LM Studio HTTP server, "
    "or a llama.cpp llama-server HTTP service.",
)
@click.option(
    "--server-url",
    default=None,
    help="HTTP server base URL. Defaults: lm-studio → http://localhost:1234, "
    "llama-server → http://localhost:8080.",
)
@click.option(
    "--label",
    default=None,
    help="Label for the machine running the HTTP server (e.g. 'desktop', 'homelab'). "
    "Required when --server-url is non-localhost. Used in cache keys + reports.",
)
@click.option(
    "--probe",
    type=click.Choice(["pp-tg", "single"], case_sensitive=False),
    default="pp-tg",
    show_default=True,
    help="HTTP backend probe: 'pp-tg' mirrors llama-bench, 'single' is one realistic prompt.",
)
@click.option(
    "--models",
    "models_csv",
    default=None,
    help="Comma-separated LM Studio model IDs to benchmark, overrides profile.",
)
@click.option(
    "--all-available",
    is_flag=True,
    default=False,
    help="LM Studio: benchmark every LLM the server knows about (loaded or downloadable).",
)
@click.option(
    "--loaded-only",
    is_flag=True,
    default=False,
    help="LM Studio: benchmark only models currently loaded in memory.",
)
@click.option(
    "--llama-bench",
    "-b",
    default=DEFAULT_LLAMA_BENCH,
    show_default=True,
    help="Path to the llama-bench binary (only used with --backend llama-bench).",
)
@click.option(
    "--models-file",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override auto-selected model profile with a custom YAML file.",
)
@click.option(
    "--fresh", is_flag=True, default=False, help="Ignore cached results; re-run everything."
)
@click.option("--repetitions", "-r", default=5, show_default=True, help="Repetitions per test.")
@click.option(
    "--n-prompt", "-p", default=512, show_default=True, help="Prompt token count (pp test)."
)
@click.option(
    "--n-gen", "-n", default=200, show_default=True, help="Generation token count (tg test)."
)
@click.option(
    "--hf-token",
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace access token (or set HF_TOKEN env var).",
)
@click.option("--threads", "-t", default=None, type=int, help="CPU thread count for llama-bench.")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["table", "json", "markdown"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.argument("extra_llama_args", nargs=-1, type=click.UNPROCESSED)
def run(
    backend: str,
    server_url: str | None,
    label: str | None,
    probe: str,
    models_csv: str | None,
    all_available: bool,
    loaded_only: bool,
    llama_bench: str,
    models_file: Path | None,
    fresh: bool,
    repetitions: int,
    n_prompt: int,
    n_gen: int,
    hf_token: str | None,
    threads: int | None,
    output: str,
    extra_llama_args: tuple[str, ...],
) -> None:
    """Run benchmarks and display a ranked comparison table.

    Any unrecognised options are forwarded verbatim to llama-bench:

    \b
        llm-bench run -- -fa 1 -ngl 32

    LM Studio examples:

    \b
        llm-bench run --backend lm-studio
        llm-bench run --backend lm-studio --probe single --repetitions 3
        llm-bench run --backend lm-studio --server-url http://homelab:1234 --label homelab

    llama-server examples:

    \b
        llm-bench run --backend llama-server
        llm-bench run --backend llama-server --server-url http://desktop:8080 --label desktop
    """
    backend = backend.lower()
    extra: list[str] = list(extra_llama_args)

    # ── Backend-specific setup ────────────────────────────────────────────
    sysinfo = sysinfo_mod.collect()
    lm_client: LMStudioClient | None = None
    ls_client: LlamaServerClient | None = None

    # Pick a default URL per backend so users don't have to memorize ports.
    if server_url is None:
        server_url = (
            DEFAULT_LM_STUDIO_URL
            if backend == "lm-studio"
            else DEFAULT_LLAMA_SERVER_URL
            if backend == "llama-server"
            else ""
        )

    if backend == "lm-studio":
        is_local = _is_local_url(server_url)
        if not is_local and not label:
            console.print(
                "[bold red]Error:[/bold red] --label is required when --server-url is "
                "not localhost (so results from different boxes don't collide)."
            )
            sys.exit(1)
        if sum([bool(models_csv), bool(models_file), all_available, loaded_only]) > 1:
            console.print(
                "[bold red]Error:[/bold red] use only one of --models / --models-file / "
                "--all-available / --loaded-only."
            )
            sys.exit(1)

        lm_client = LMStudioClient(server_url)
        try:
            lm_client.list_models()  # connectivity probe
        except LMStudioError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)
        loaded_ids_list = lm_client.loaded_model_ids()
        loaded_ids = set(loaded_ids_list)

        # Pick a profile
        if models_csv:
            profile = _ad_hoc_profile(_models_from_csv(models_csv))
        elif models_file:
            profile = model_registry.load_profile_from_file(models_file)
        elif all_available:
            ids = lm_client.all_llm_ids()
            if not ids:
                console.print("[bold red]Error:[/bold red] no LLMs found on the server.")
                sys.exit(1)
            profile = _ad_hoc_profile(
                [Model(name=i, lm_studio_id=i) for i in ids],
                name="all-available",
            )
        elif loaded_only:
            if not loaded_ids_list:
                console.print(
                    "[bold red]Error:[/bold red] no models are currently loaded in LM Studio."
                )
                sys.exit(1)
            profile = _ad_hoc_profile(
                [Model(name=i, lm_studio_id=i) for i in loaded_ids_list],
                name="loaded-only",
            )
        else:
            named = model_registry.get_profile_by_name("lm_studio")
            if named is None:
                console.print(
                    "[bold red]Error:[/bold red] no 'lm_studio' profile found. "
                    "Pass --models, --models-file, --all-available, or --loaded-only."
                )
                sys.exit(1)
            profile = named
            # If the bundled profile doesn't match what's loaded, suggest a better path
            # rather than running through 5 guaranteed failures.
            in_profile = {m.identifier for m in profile.models}
            if loaded_ids and not (loaded_ids & in_profile):
                console.print(
                    "[bold red]Error:[/bold red] none of the bundled 'lm_studio' profile "
                    "models are loaded in LM Studio."
                )
                console.print(
                    "[dim]Currently loaded:[/dim] "
                    + (", ".join(loaded_ids_list) if loaded_ids_list else "(nothing loaded)")
                )
                console.print(
                    "[dim]Try one of:[/dim]\n"
                    "  llm-bench run --backend lm-studio --loaded-only\n"
                    "  llm-bench run --backend lm-studio --all-available\n"
                    "  llm-bench run --backend lm-studio --models "
                    f"{loaded_ids_list[0] if loaded_ids_list else '<id>'}"
                )
                sys.exit(1)

        bench_version = lm_client.server_version()
        if is_local:
            hw_fingerprint: str | None = None
        else:
            raw = f"{label}:{server_url}"
            hw_fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]
    elif backend == "llama-server":
        is_local = _is_local_url(server_url)
        if not is_local and not label:
            console.print(
                "[bold red]Error:[/bold red] --label is required when --server-url is "
                "not localhost (so results from different boxes don't collide)."
            )
            sys.exit(1)
        if all_available or loaded_only or models_file:
            console.print(
                "[bold red]Error:[/bold red] --all-available, --loaded-only and "
                "--models-file aren't meaningful for llama-server (one model per "
                "instance). Use --models to override the displayed name, or rely on /props."
            )
            sys.exit(1)

        ls_client = LlamaServerClient(server_url)
        # Connectivity probe — try /health first, fall back to /props since older
        # builds don't expose /health. Either succeeding means the server is up.
        reachable = False
        for probe_path in ("/health", "/props"):
            try:
                ls_client._request("GET", probe_path)
                reachable = True
                break
            except LlamaServerError:
                continue
        if not reachable:
            console.print(
                f"[bold red]Error:[/bold red] cannot reach llama-server at {server_url}. "
                "Is `llama-server` running on this URL?"
            )
            sys.exit(1)
        model_name = (
            models_csv.split(",")[0].strip() if models_csv else ls_client.model_id()
        )

        profile = _ad_hoc_profile(
            [Model(name=model_name, lm_studio_id=model_name)],
            name="llama-server",
        )
        bench_version = ls_client.server_version()
        if is_local:
            hw_fingerprint = None
        else:
            raw = f"{label}:{server_url}"
            hw_fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]
        loaded_ids_list = []
        loaded_ids = set()
    else:
        bench_path = Path(llama_bench)
        if not bench_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] llama-bench not found at [cyan]{llama_bench}[/cyan]"
            )
            console.print("Use [bold]--llama-bench[/bold] or set the correct path.")
            sys.exit(1)
        if threads is not None:
            extra.extend(["-t", str(threads)])
        if models_file:
            profile = model_registry.load_profile_from_file(models_file)
        else:
            profile = model_registry.select_profile(sysinfo)
        bench_version = get_llama_bench_version(llama_bench)
        hw_fingerprint = None
        loaded_ids = set()

    # ── Sysinfo / header panels ───────────────────────────────────────────
    is_remote_http = backend in ("lm-studio", "llama-server") and not _is_local_url(server_url)
    if is_remote_http:
        backend_title = "LM Studio" if backend == "lm-studio" else "llama-server"
        console.print(
            Panel.fit(
                f"[bold]Remote target[/bold]  [cyan]{label}[/cyan]\n"
                f"[dim]url:[/dim]  {server_url}\n"
                f"[dim]profile:[/dim]  {profile.description}",
                title=f"[bold]{backend_title}[/bold]",
                border_style="cyan",
            )
        )
    else:
        console.print(build_sysinfo_panel(sysinfo, profile.description))
    console.print()

    run_id = storage.new_run_id()
    is_http_backend = backend in ("lm-studio", "llama-server")
    meta = storage.make_run_meta(
        run_id=run_id,
        sysinfo=sysinfo,
        llama_bench_version=bench_version,
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
        profile_name=profile.profile,
        model_count=len(profile.models),
        backend=backend,
        server_url=server_url if is_http_backend else None,
        label=label if is_http_backend else None,
        hw_fingerprint_override=hw_fingerprint,
    )

    if is_http_backend:
        header_top = (
            f"[bold cyan]Benchmark Run[/bold cyan]  [dim]{run_id}[/dim]\n"
            f"[dim]server:[/dim]  {server_url}  [dim]({bench_version})[/dim]\n"
            f"[dim]probe:[/dim]   {probe}  [dim]rep={repetitions}"
            + (f"  pp={n_prompt}  tg={n_gen}" if probe == "pp-tg" else "")
            + "[/dim]"
            + ("  [yellow](--fresh)[/yellow]" if fresh else "")
        )
    else:
        header_top = (
            f"[bold cyan]Benchmark Run[/bold cyan]  [dim]{run_id}[/dim]\n"
            f"[dim]binary:[/dim]  {llama_bench}  [dim]({bench_version})[/dim]\n"
            f"[dim]config:[/dim]  pp={n_prompt}  tg={n_gen}  rep={repetitions}"
            + ("  [yellow](--fresh)[/yellow]" if fresh else "")
        )
    console.print(Panel.fit(header_top, border_style="cyan"))
    console.print()

    # Warn (don't fail) when a profile model isn't loaded — LM Studio can JIT-load,
    # but the user should know it'll be slower for the first request.
    if backend == "lm-studio" and loaded_ids:
        missing = [m.identifier for m in profile.models if m.identifier not in loaded_ids]
        if missing and len(missing) < len(profile.models):
            console.print(
                "[yellow]⚠ Some profile models are not currently loaded — LM Studio will "
                "load them on first request:[/yellow] "
                + ", ".join(missing[:5])
                + (f"  …(+{len(missing) - 5} more)" if len(missing) > 5 else "")
            )
            console.print()

    results: list[BenchResult] = []
    cached_flags: dict[str, bool] = {}
    total = len(profile.models)

    completed_lines: list[str] = []
    recent_log: deque[str] = deque(maxlen=6)
    _state: dict = {"label": "", "name": "", "t0": 0.0, "active": False}
    _spinner = Spinner("dots", style="bold cyan")

    def _make_live() -> Group:
        rows: list = []
        for ln in completed_lines:
            rows.append(Text.from_markup(ln))
        if _state["active"]:
            elapsed = time.monotonic() - _state["t0"]
            m, s = divmod(int(elapsed), 60)
            frame = _spinner.render(time.monotonic())
            header = Text()
            header.append("  ")
            header.append_text(frame)  # type: ignore[arg-type]
            header.append(f"  {_state['label']} ", style="dim")
            header.append(_state["name"], style="bold")
            header.append(f"  {m}:{s:02d}", style="dim")
            rows.append(header)
            log_text = Text()
            if recent_log:
                for ln in recent_log:
                    log_text.append(f"  {ln}\n", style="dim cyan")
            else:
                log_text.append("  waiting for llama-bench output…", style="dim")
            rows.append(Panel(log_text, border_style="dim", padding=(0, 1)))
        return Group(*rows) if rows else Group(Text(""))

    class _Renderable:
        def __rich__(self) -> Group:  # type: ignore[override]
            return _make_live()

    with Live(_Renderable(), console=console, refresh_per_second=8):
        for idx, model in enumerate(profile.models):
            slot = f"[{idx + 1}/{total}]"
            markup_slot = f"\\[{idx + 1}/{total}]"
            ident = model.identifier

            # ── Cache lookup ──────────────────────────────────────────────
            if not fresh:
                cached = storage.find_cached_result(meta, ident)
                if cached is not None:
                    tg_disp = f"{cached.tg_avg_ts:.2f}" if cached.tg_avg_ts else "?"
                    completed_lines.append(
                        f"  {markup_slot} [dim]↩ cached[/dim]  [bold]{escape(model.name)}[/bold]"
                        f"  TG={tg_disp} t/s"
                    )
                    results.append(cached)
                    cached_flags[ident] = True
                    continue

            _state.update(
                {"label": slot, "name": model.name, "t0": time.monotonic(), "active": True}
            )
            recent_log.clear()

            def on_status(line: str) -> None:
                recent_log.append(line[:200])

            if backend == "lm-studio":
                assert lm_client is not None
                result = run_lm_studio_benchmark(
                    client=lm_client,
                    model_id=ident,
                    n_prompt=n_prompt,
                    n_gen=n_gen,
                    repetitions=repetitions,
                    probe=probe,
                    on_status=on_status,
                )
            elif backend == "llama-server":
                assert ls_client is not None
                result = run_llama_server_benchmark(
                    client=ls_client,
                    model_id=ident,
                    n_prompt=n_prompt,
                    n_gen=n_gen,
                    repetitions=repetitions,
                    probe=probe,
                    on_status=on_status,
                )
            else:
                stdout, stderr, returncode = run_benchmark(
                    llama_bench=llama_bench,
                    hf_repo=ident,
                    n_prompt=n_prompt,
                    n_gen=n_gen,
                    repetitions=repetitions,
                    hf_token=hf_token,
                    extra_args=extra,
                    on_status=on_status,
                )
                if returncode != 0:
                    last_err = (
                        stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
                    )
                    result = BenchResult(
                        model_name=model.name,
                        hf_repo=ident,
                        error=f"exit {returncode}: {last_err[:70]}",
                    )
                else:
                    json_data = extract_json(stdout)
                    result = parse_bench_output(model.name, ident, json_data)

            _state["active"] = False
            recent_log.clear()

            if result.error:
                completed_lines.append(
                    f"  {markup_slot} [yellow]⚠[/yellow] [bold]{escape(model.name)}[/bold]"
                    f"  [yellow]{escape(result.error)}[/yellow]"
                )
            else:
                tg_str = f"{result.tg_avg_ts:.2f} t/s" if result.tg_avg_ts else "?"
                completed_lines.append(
                    f"  {markup_slot} [green]✓[/green] [bold]{escape(model.name)}[/bold]"
                    f"  TG={tg_str}"
                )

            results.append(result)
            cached_flags[ident] = False

    # ── Compute scores and display ────────────────────────────────────────
    scores = compute_scores(results)

    console.print()

    if output == "table":
        console.print(build_summary_table(results, scores, cached_flags, profile, sysinfo))
        console.print()
        console.print(
            "[dim]Score = 0.7×(TG t/s ÷ best) + 0.3×(PP t/s ÷ best), scaled 0–100. "
            "Ratings are relative within this run.[/dim]"
        )
    elif output == "json":
        payload = [
            {
                **r.to_dict(),
                "score": scores.get(r.hf_repo),
                "cached": cached_flags.get(r.hf_repo, False),
            }
            for r in results
        ]
        console.print_json(json.dumps(payload, indent=2))
    elif output == "markdown":
        _print_markdown_table(results, scores, cached_flags)

    # ── Save results ──────────────────────────────────────────────────────
    run_dir = storage.save_run(meta, results, cached_flags)
    console.print(f"\n[dim]Results saved → {run_dir}  (id: {run_id})[/dim]")


def _print_markdown_table(
    results: list[BenchResult],
    scores: dict[str, float],
    cached_flags: dict[str, bool],
) -> None:
    sorted_r = sorted(results, key=lambda r: scores.get(r.hf_repo, -1), reverse=True)
    header = "| # | Model | PP t/s | TG t/s | Score |"
    sep = "|---|-------|--------|--------|-------|"
    console.print(header)
    console.print(sep)
    for i, r in enumerate(sorted_r, 1):
        pp = f"{r.pp_avg_ts:.2f}" if r.pp_avg_ts else "—"
        tg = f"{r.tg_avg_ts:.2f}" if r.tg_avg_ts else "—"
        score = f"{scores.get(r.hf_repo, 0):.1f}"
        cached = " *(cached)*" if cached_flags.get(r.hf_repo) else ""
        console.print(f"| {i} | {r.model_name}{cached} | {pp} | {tg} | {score} |")


# ── llm-bench sysinfo ─────────────────────────────────────────────────────────


@main.command()
def sysinfo() -> None:
    """Show detected hardware and the model profile that would be selected."""
    info = sysinfo_mod.collect()
    profile = model_registry.select_profile(info)
    console.print(build_sysinfo_panel(info, profile.description))
    console.print()
    console.print(
        f"[bold]Profile:[/bold]  [cyan]{profile.profile}[/cyan]  —  {profile.description}"
    )
    console.print(f"[bold]Models :[/bold]  {len(profile.models)} models in this profile")
    for m in profile.models:
        fit = (
            "[green]✓[/green]"
            if m.estimated_size_gb < info.available_ram_gb * 0.85
            else "[yellow]⚠[/yellow]"
        )
        console.print(f"  {fit}  [bold]{m.name}[/bold]  [dim]~{m.estimated_size_gb:.1f} GiB[/dim]")


# ── llm-bench models ──────────────────────────────────────────────────────────


@main.group()
def models() -> None:
    """Manage and inspect model profiles."""


@models.command("list")
def models_list() -> None:
    """List all available model profiles."""
    for p in model_registry.all_profiles():
        source = f"[dim]{p.source_path}[/dim]" if p.source_path else ""
        console.print(f"[bold cyan]{p.profile}[/bold cyan]  {p.description}  {source}")
        for m in p.models:
            console.print(
                f"  [dim]·[/dim] {m.name}  "
                f"[dim]{m.identifier}  ~{m.estimated_size_gb:.1f} GiB[/dim]"
            )
        console.print()


# ── llm-bench results ─────────────────────────────────────────────────────────


@main.group()
def results() -> None:
    """Browse and compare past benchmark runs."""


@results.command("list")
def results_list() -> None:
    """List all saved benchmark runs."""
    metas = storage.list_runs()
    if not metas:
        console.print(
            "[dim]No runs saved yet. Run [bold]llm-bench run[/bold] to get started.[/dim]"
        )
        return
    console.print(build_history_table(metas))


@results.command("show")
@click.argument("run_id")
def results_show(run_id: str) -> None:
    """Show the results table for a specific run."""
    try:
        meta, bench_results, cached = storage.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id}")
        sys.exit(1)

    sysinfo = sysinfo_mod.collect()
    profile = model_registry.get_profile_by_name(meta.profile_name)
    if profile is None:
        # Profile no longer on disk (e.g. ad-hoc --models run): synthesize a stub
        # so the size map still keys correctly off identifiers.
        profile = ModelProfile(
            profile=meta.profile_name,
            description=f"(profile '{meta.profile_name}' not found on disk)",
            min_ram_gb=0,
            max_ram_gb=9999,
            models=[Model(name=r.model_name, hf_repo=r.hf_repo) for r in bench_results],
        )
    scores = compute_scores(bench_results)
    console.print(build_summary_table(bench_results, scores, cached, profile, sysinfo))
    backend_tag = (
        f"  ·  backend: {meta.backend}" + (f" ({meta.label})" if meta.label else "")
        if meta.backend != "llama-bench"
        else ""
    )
    console.print(
        f"\n[dim]Run: {run_id}  ·  {meta.timestamp[:19]}  ·  hw: {meta.hw_fingerprint}"
        f"{backend_tag}[/dim]"
    )


@results.command("compare")
@click.argument("run_id_a")
@click.argument("run_id_b")
def results_compare(run_id_a: str, run_id_b: str) -> None:
    """Compare two runs side-by-side."""
    try:
        meta_a, results_a, _ = storage.load_run(run_id_a)
        meta_b, results_b, _ = storage.load_run(run_id_b)
    except FileNotFoundError as exc:
        console.print(f"[red]Run not found:[/red] {exc}")
        sys.exit(1)

    console.print(build_compare_table(meta_a, results_a, meta_b, results_b))
