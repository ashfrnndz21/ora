"""CLI entry point — typer commands: serve, hub."""

from __future__ import annotations

import typer

app = typer.Typer(name="sqlagent", help="The NL2SQL Agentic Runtime")


@app.command()
def serve(
    db: str = typer.Option("", help="Database URL (e.g., sqlite:///data.db, postgresql://...)"),
    model: str = typer.Option("gpt-4o", help="LLM model"),
    port: int = typer.Option(8080, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    auth: bool = typer.Option(False, help="Enable authentication"),
):
    """Start the sqlagent workspace server."""
    import os, pathlib, uvicorn
    from sqlagent.config import AgentConfig
    from sqlagent.server import create_app

    # Auto-load .env from project root if present
    _env_path = pathlib.Path(".env")
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    config = AgentConfig(
        llm_model=model,
        host=host,
        port=port,
        auth_enabled=auth,
    )
    app_instance = create_app(config, default_db=db)

    typer.echo(f"\n  sqlagent workspace started")
    typer.echo(f"  Workspace    →  http://{host}:{port}")
    typer.echo(f"  REST API     →  http://{host}:{port}/docs")
    if db:
        typer.echo(f"  DB: {db}")
    typer.echo(f"  Model: {model}\n")

    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command()
def hub(
    action: str = typer.Argument("list", help="list | install"),
    pack: str = typer.Option("", help="Pack name to install"),
):
    """Manage QueryHub training packs."""
    if action == "list":
        from sqlagent.hub import list_packs
        packs = list_packs()
        for p in packs:
            typer.echo(f"  {p['name']:25s} {p['description']}")
    elif action == "install":
        if not pack:
            typer.echo("Usage: sqlagent hub install --pack retail-asean")
            raise typer.Exit(1)
        typer.echo(f"Installing pack: {pack}")
        # Would need an agent instance — for CLI, just show info
        typer.echo("Use the Python API or REST endpoint to install packs.")


def main():
    app()


if __name__ == "__main__":
    main()
