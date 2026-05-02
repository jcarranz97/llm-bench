"""Rich terminal UI: tables, panels, ratings, and compare views."""

from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_bench.models import ModelProfile
from llm_bench.parser import BenchResult
from llm_bench.storage import RunMeta
from llm_bench.sysinfo import SystemInfo

# ── Scoring & rating ─────────────────────────────────────────────────────────


def compute_scores(results: list[BenchResult]) -> dict[str, float]:
    """Return a 0–100 composite score per hf_repo. Higher = faster."""
    valid = [r for r in results if r.error is None]
    if not valid:
        return {}
    max_tg = max((r.tg_avg_ts for r in valid if r.tg_avg_ts), default=1.0) or 1.0
    max_pp = max((r.pp_avg_ts for r in valid if r.pp_avg_ts), default=1.0) or 1.0
    return {
        r.hf_repo: round(
            0.7 * ((r.tg_avg_ts or 0) / max_tg * 100) + 0.3 * ((r.pp_avg_ts or 0) / max_pp * 100),
            1,
        )
        for r in valid
    }


def _stars(score: float) -> str:
    filled = "[bold yellow]★[/bold yellow]"
    empty = "[dim]★[/dim]"
    n = 5 if score >= 90 else 4 if score >= 70 else 3 if score >= 50 else 2 if score >= 30 else 1
    return filled * n + empty * (5 - n)


def _fit_indicator(estimated_gb: float, available_ram_gb: float) -> str:
    ratio = estimated_gb / max(available_ram_gb, 0.1)
    if ratio <= 0.85:
        return "[green]✓ Fits[/green]"
    if ratio <= 1.0:
        return "[yellow]⚠ Tight[/yellow]"
    return "[red]✗ Too large[/red]"


# ── Sysinfo panel ─────────────────────────────────────────────────────────────


def build_sysinfo_panel(sysinfo: SystemInfo, profile_name: str) -> Panel:
    ram_bar_width = 20
    used_ratio = 1 - sysinfo.available_ram_gb / max(sysinfo.total_ram_gb, 0.1)
    filled = int(used_ratio * ram_bar_width)
    ram_bar = "[green]" + "█" * filled + "[/green]" + "░" * (ram_bar_width - filled)
    ram_line = (
        f"{ram_bar} {sysinfo.available_ram_gb:.1f} GB free / {sysinfo.total_ram_gb:.1f} GB total"
    )

    if sysinfo.gpus:
        gpu_lines = "\n".join(
            f"  [cyan]{g.name}[/cyan]"
            + (f"  [dim]{g.vram_gb:.1f} GB VRAM[/dim]" if g.vram_gb else "")
            + f"  ({g.backend})"
            for g in sysinfo.gpus
        )
    else:
        gpu_lines = "  [dim]No GPU detected — running on CPU[/dim]"

    if sysinfo.cpu_physical_cores > 0:
        cpu_detail = f"({sysinfo.cpu_physical_cores} cores, {sysinfo.cpu_cores} threads)"
    else:
        cpu_detail = f"({sysinfo.cpu_cores} threads)"

    content = (
        f"[bold]CPU[/bold]  {sysinfo.cpu_model}  [dim]{cpu_detail}[/dim]\n"
        f"[bold]RAM[/bold]  {ram_line}\n"
        f"[bold]GPU[/bold]\n{gpu_lines}\n"
        f"[bold]OS [/bold]  {sysinfo.os}\n"
        f"\n[dim]Selected profile:[/dim]  [bold cyan]{profile_name}[/bold cyan]"
    )
    return Panel(content, title="[bold]System Info[/bold]", border_style="cyan", padding=(0, 1))


# ── Main results table ────────────────────────────────────────────────────────


def build_summary_table(
    results: list[BenchResult],
    scores: dict[str, float],
    cached_flags: dict[str, bool],
    profile: ModelProfile,
    sysinfo: SystemInfo,
) -> Table:
    model_size_map = {m.hf_repo: m.estimated_size_gb for m in profile.models}

    table = Table(
        title="[bold]LLM Benchmark Results[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        padding=(0, 1),
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Model", no_wrap=False)
    table.add_column("Fits?", justify="center", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Params", justify="right", no_wrap=True)
    table.add_column("Backend", justify="center", no_wrap=True)
    table.add_column("Threads", justify="right", no_wrap=True)
    table.add_column("PP t/s\n±std", justify="right", style="green", no_wrap=True)
    table.add_column("TG t/s\n±std", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Rating", justify="left", no_wrap=True)

    sorted_results = sorted(
        results,
        key=lambda r: scores.get(r.hf_repo, -1.0),
        reverse=True,
    )

    for rank, r in enumerate(sorted_results, start=1):
        cached = cached_flags.get(r.hf_repo, False)
        cached_badge = " [dim](cached)[/dim]" if cached else ""
        name_cell = Text.from_markup(
            f"[bold]{r.model_name}[/bold]{cached_badge}\n[dim]{r.hf_repo}[/dim]"
        )

        if r.error:
            table.add_row(
                str(rank),
                name_cell,
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                Text.from_markup(f"[red]{r.error}[/red]"),
            )
            continue

        est_gb = model_size_map.get(r.hf_repo, r.model_size_gib)
        fit = _fit_indicator(est_gb, sysinfo.available_ram_gb)
        score = scores.get(r.hf_repo, 0.0)

        pp = f"{r.pp_avg_ts:.2f}\n±{r.pp_std_ts:.2f}" if r.pp_avg_ts is not None else "N/A"
        tg = f"{r.tg_avg_ts:.2f}\n±{r.tg_std_ts:.2f}" if r.tg_avg_ts is not None else "N/A"
        size = f"{r.model_size_gib:.2f} GiB" if r.model_size_bytes else "N/A"
        params = f"{r.model_params_b:.2f} B" if r.model_n_params else "N/A"

        table.add_row(
            str(rank),
            name_cell,
            Text.from_markup(fit),
            size,
            params,
            r.backend or "N/A",
            str(r.threads) if r.threads else "N/A",
            pp,
            tg,
            f"{score:.1f}",
            Text.from_markup(_stars(score)),
        )

    return table


# ── Run history table ─────────────────────────────────────────────────────────


def build_history_table(metas: list[RunMeta]) -> Table:
    table = Table(
        title="[bold]Past Benchmark Runs[/bold]",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        border_style="blue",
        padding=(0, 1),
    )
    table.add_column("Run ID", style="bold", no_wrap=True)
    table.add_column("Timestamp", no_wrap=True)
    table.add_column("Profile", no_wrap=True)
    table.add_column("Models", justify="right")
    table.add_column("HW Fingerprint", style="dim", no_wrap=True)
    table.add_column("llama-bench", style="dim")

    for m in metas:
        table.add_row(
            m.run_id,
            m.timestamp[:19].replace("T", " "),
            m.profile_name,
            str(m.model_count),
            m.hw_fingerprint,
            m.llama_bench_version,
        )
    return table


# ── Compare table ─────────────────────────────────────────────────────────────


def build_compare_table(
    meta_a: RunMeta,
    results_a: list[BenchResult],
    meta_b: RunMeta,
    results_b: list[BenchResult],
) -> Table:
    table = Table(
        title=f"[bold]Compare[/bold]  {meta_a.run_id}  vs  {meta_b.run_id}",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        padding=(0, 1),
        show_lines=True,
    )
    table.add_column("Model", no_wrap=True)
    table.add_column(f"PP t/s\n{meta_a.run_id[:14]}", justify="right", style="green")
    table.add_column(f"PP t/s\n{meta_b.run_id[:14]}", justify="right", style="green")
    table.add_column("PP Δ", justify="right")
    table.add_column(f"TG t/s\n{meta_a.run_id[:14]}", justify="right", style="cyan")
    table.add_column(f"TG t/s\n{meta_b.run_id[:14]}", justify="right", style="cyan")
    table.add_column("TG Δ", justify="right")

    map_b = {r.hf_repo: r for r in results_b}

    for r_a in results_a:
        r_b = map_b.get(r_a.hf_repo)
        pp_a = f"{r_a.pp_avg_ts:.2f}" if r_a.pp_avg_ts else "—"
        tg_a = f"{r_a.tg_avg_ts:.2f}" if r_a.tg_avg_ts else "—"

        if r_b is None:
            table.add_row(r_a.model_name, pp_a, "[dim]n/a[/dim]", "", tg_a, "[dim]n/a[/dim]", "")
            continue

        pp_b = f"{r_b.pp_avg_ts:.2f}" if r_b.pp_avg_ts else "—"
        tg_b = f"{r_b.tg_avg_ts:.2f}" if r_b.tg_avg_ts else "—"

        pp_delta = _delta_str(r_a.pp_avg_ts, r_b.pp_avg_ts)
        tg_delta = _delta_str(r_a.tg_avg_ts, r_b.tg_avg_ts)

        table.add_row(
            f"[bold]{r_a.model_name}[/bold]\n[dim]{r_a.hf_repo}[/dim]",
            pp_a,
            pp_b,
            Text.from_markup(pp_delta),
            tg_a,
            tg_b,
            Text.from_markup(tg_delta),
        )
    return table


def _delta_str(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return ""
    pct = (b - a) / a * 100
    if pct > 0:
        return f"[green]+{pct:.1f}%[/green]"
    return f"[red]{pct:.1f}%[/red]"
