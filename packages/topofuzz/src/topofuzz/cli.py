"""topofuzz CLI entrypoint.

Real commands are implemented in later build steps. This module exists so that
`topofuzz --help` works at the end of Weekend 1 step 1.
"""

from __future__ import annotations

import typer

from topofuzz import __version__

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


@app.command()
def scan(fixture: str) -> None:
    """Static risk badge for a LangGraph topology. (Not yet implemented.)"""
    raise NotImplementedError("tsf scan is implemented in Weekend 5 step 14.")


@app.command()
def run(fixture: str) -> None:
    """Run topology fuzzing against an eval set. (Not yet implemented.)"""
    raise NotImplementedError("tsf run is implemented in Weekend 2.")


@app.command()
def report(run_dir: str) -> None:
    """Open the HTML report for a run. (Not yet implemented.)"""
    raise NotImplementedError("tsf report is implemented in Weekend 3 step 11.")


@app.command()
def verify(run_dir: str) -> None:
    """Verify DSSE signature on a run's report. (Not yet implemented.)"""
    raise NotImplementedError("tsf verify is implemented in Weekend 3 step 10a.")


if __name__ == "__main__":
    app()
