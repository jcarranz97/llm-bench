"""Click CLI: llm-bench run / sysinfo / results / models."""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

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
    "--llama-bench",
    "-b",
    default=DEFAULT_LLAMA_BENCH,
    show_default=True,
    help="Path to the llama-bench binary.",
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
    """
    bench_path = Path(llama_bench)
    if not bench_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] llama-bench not found at [cyan]{llama_bench}[/cyan]"
        )
        console.print("Use [bold]--llama-bench[/bold] or set the correct path.")
        sys.exit(1)

    extra: list[str] = list(extra_llama_args)
    if threads is not None:
        extra.extend(["-t", str(threads)])

    # ── Hardware detection ────────────────────────────────────────────────
    sysinfo = sysinfo_mod.collect()

    # ── Model profile selection ───────────────────────────────────────────
    if models_file:
        profile = model_registry.load_profile_from_file(models_file)
    else:
        profile = model_registry.select_profile(sysinfo)

    console.print(build_sysinfo_panel(sysinfo, profile.description))
    console.print()

    bench_version = get_llama_bench_version(llama_bench)
    run_id = storage.new_run_id()
    meta = storage.make_run_meta(
        run_id=run_id,
        sysinfo=sysinfo,
        llama_bench_version=bench_version,
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
        profile_name=profile.profile,
        model_count=len(profile.models),
    )

    console.print(
        Panel.fit(
            f"[bold cyan]Benchmark Run[/bold cyan]  [dim]{run_id}[/dim]\n"
            f"[dim]binary:[/dim]  {llama_bench}  [dim]({bench_version})[/dim]\n"
            f"[dim]config:[/dim]  pp={n_prompt}  tg={n_gen}  rep={repetitions}"
            + ("  [yellow](--fresh)[/yellow]" if fresh else ""),
            border_style="cyan",
        )
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
            label = f"[{idx + 1}/{total}]"
            markup_label = f"\\[{idx + 1}/{total}]"

            # ── Cache lookup ──────────────────────────────────────────────
            if not fresh:
                cached = storage.find_cached_result(meta, model.hf_repo)
                if cached is not None:
                    completed_lines.append(
                        f"  {markup_label} [dim]↩ cached[/dim]  [bold]{escape(model.name)}[/bold]"
                        f"  TG={cached.tg_avg_ts:.2f} t/s"
                    )
                    results.append(cached)
                    cached_flags[model.hf_repo] = True
                    continue

            _state.update(
                {"label": label, "name": model.name, "t0": time.monotonic(), "active": True}
            )
            recent_log.clear()

            def on_status(line: str) -> None:
                recent_log.append(line[:200])

            stdout, stderr, returncode = run_benchmark(
                llama_bench=llama_bench,
                hf_repo=model.hf_repo,
                n_prompt=n_prompt,
                n_gen=n_gen,
                repetitions=repetitions,
                hf_token=hf_token,
                extra_args=extra,
                on_status=on_status,
            )

            _state["active"] = False
            recent_log.clear()

            if returncode != 0:
                last_err = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
                result = BenchResult(
                    model_name=model.name,
                    hf_repo=model.hf_repo,
                    error=f"exit {returncode}: {last_err[:70]}",
                )
                completed_lines.append(
                    f"  {markup_label} [red]✗[/red] [bold]{escape(model.name)}[/bold]"
                    "  [red]failed[/red]"
                )
            else:
                json_data = extract_json(stdout)
                result = parse_bench_output(model.name, model.hf_repo, json_data)
                if result.error:
                    completed_lines.append(
                        f"  {markup_label} [yellow]⚠[/yellow] [bold]{escape(model.name)}[/bold]"
                        f"  [yellow]{escape(result.error)}[/yellow]"
                    )
                else:
                    tg_str = f"{result.tg_avg_ts:.2f} t/s" if result.tg_avg_ts else "?"
                    completed_lines.append(
                        f"  {markup_label} [green]✓[/green] [bold]{escape(model.name)}[/bold]"
                        f"  TG={tg_str}"
                    )

            results.append(result)
            cached_flags[model.hf_repo] = False

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
                f"  [dim]·[/dim] {m.name}  [dim]{m.hf_repo}  ~{m.estimated_size_gb:.1f} GiB[/dim]"
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
    profile = model_registry.select_profile(sysinfo)
    scores = compute_scores(bench_results)
    console.print(build_summary_table(bench_results, scores, cached, profile, sysinfo))
    console.print(
        f"\n[dim]Run: {run_id}  ·  {meta.timestamp[:19]}  ·  hw: {meta.hw_fingerprint}[/dim]"
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
