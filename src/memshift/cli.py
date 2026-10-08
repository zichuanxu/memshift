"""Command-line entry point (plan §7). One subcommand per stage task as it lands."""

from __future__ import annotations

from pathlib import Path

import typer

from memshift import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="MemShift benchmark tools.")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def gen(
    seed: int = typer.Option(0, help="World seed."),
    branch: str = typer.Option("natural", help="natural | src | def | cons"),
    months: int = typer.Option(12),
    out: Path = typer.Option(Path("worlds"), help="Output root (not in git)."),
) -> None:
    """Generate one world branch and write JSONL + manifest + SHA256 (plan §4.2)."""
    from memshift.gen.io import write_world
    from memshift.gen.shifts import generate_branch

    params = {} if months == 12 else {"months": months, "fork_month": max(1, months - 1)}
    typer.echo(str(write_world(generate_branch(seed, branch, params), out)))


@app.command()
def smoke(
    seed: int = typer.Option(0, help="World seed."),
    probes: int = typer.Option(50, help="Number of probes."),
) -> None:
    """Stage 0.4: one 3-month world, HoldMem (1,1), 50 probes; measures t_i and t_q."""
    from memshift.experiments.smoke import run

    typer.echo(str(run(seed, probes)))


@app.command()
def e4(
    seeds: str = typer.Option(..., help="Seed range: '0-9' or '0,1,2'."),
    name: str = typer.Option("e4", help="Run name prefix under runs/."),
    analyse: bool = typer.Option(True, help="Write analysis/e4_metrics.json after the run."),
) -> None:
    """E4: calendar sync semantics, {ideal, official} x {B} x {HoldMem, zero} (plan §6)."""
    from memshift.experiments.e1 import parse_seeds
    from memshift.experiments.e4 import run

    run_dir = run(parse_seeds(seeds), name)
    if analyse:
        from memshift.analysis.e4 import analyse as analyse_run

        analyse_run(run_dir)
    typer.echo(str(run_dir))


@app.command()
def verify(run_dir: Path = typer.Argument(..., help="A directory under runs/.")) -> None:
    """Re-hash every output file of a run and compare with its manifest."""
    from memshift.runinfo import verify_manifest

    bad = verify_manifest(run_dir)
    if bad:
        typer.echo(f"MISMATCH: {bad}", err=True)
        raise typer.Exit(code=1)
    typer.echo("manifest OK")


if __name__ == "__main__":
    app()
