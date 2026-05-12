"""topofuzz CLI entrypoint.

Real commands are implemented in later build steps. This module exists so that
`topofuzz --help` works at the end of Weekend 1 step 1.
"""

from __future__ import annotations

import typer

from topofuzz import __version__

# Reserved flag namespace per BLUEPRINT_v5plus.md §16 #2. These flags are NOT
# wired yet — step 13 adds raise-on-use guards. Listed here so future
# contributors do not accidentally repurpose these names for unrelated options.
RESERVED_FLAGS: tuple[str, ...] = (
    "--conformance",
    "--signoff",
    "--judge-pool",
    "--sandbox",
    "--key-uri",
    "--kms",
    "--allow-side-effects",
    "--tenant-id",
    "--per-cell-entropy",
)

app = typer.Typer(
    name="topofuzz",
    help="Topology Safety Fuzzer for multi-agent LLM systems (TSF v5+).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Print version and exit."
    ),
) -> None:
    if version:
        typer.echo(f"topofuzz {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


_NOT_IMPLEMENTED_EXIT = 2


def _not_yet(command: str, lands: str) -> None:
    typer.echo(
        f"topofuzz {command}: not yet implemented (lands {lands}).",
        err=True,
    )
    raise typer.Exit(code=_NOT_IMPLEMENTED_EXIT)


@app.command()
def scan(fixture: str) -> None:
    """Static risk badge for a LangGraph topology. (Not yet implemented.)"""
    _not_yet("scan", "Weekend 5 step 14")


@app.command()
def run(fixture: str) -> None:
    """Run topology fuzzing against an eval set. (Not yet implemented.)"""
    _not_yet("run", "Weekend 2 step 5")


@app.command()
def report(run_dir: str) -> None:
    """Open the HTML report for a run. (Not yet implemented.)"""
    _not_yet("report", "Weekend 3 step 11")


@app.command()
def verify(run_dir: str) -> None:
    """Verify DSSE signature on a run's report. (Not yet implemented.)"""
    _not_yet("verify", "Weekend 3 step 10a")


if __name__ == "__main__":
    app()
