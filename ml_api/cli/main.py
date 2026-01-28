"""Typer CLI application for ML API."""

import typer
from typing import Optional
import uvicorn

app = typer.Typer(
    name="ml-api",
    help="ML API Command Line Interface",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    log_level: str = typer.Option(
        "info", "--log-level", "-l", help="Log level (debug, info, warning, error, critical)"
    ),
    access_log: bool = typer.Option(True, "--access-log/--no-access-log", help="Enable access log"),
    proxy_headers: bool = typer.Option(
        False, "--proxy-headers", help="Enable proxy headers (for reverse proxy setups)"
    ),
    forwarded_allow_ips: Optional[str] = typer.Option(
        None, "--forwarded-allow-ips", help="Comma-separated list of IPs to trust for proxy headers"
    ),
    ssl_keyfile: Optional[str] = typer.Option(None, "--ssl-keyfile", help="Path to SSL key file"),
    ssl_certfile: Optional[str] = typer.Option(
        None, "--ssl-certfile", help="Path to SSL certificate file"
    ),
    limit_concurrency: Optional[int] = typer.Option(
        None, "--limit-concurrency", help="Maximum number of concurrent connections"
    ),
    limit_max_requests: Optional[int] = typer.Option(
        None, "--limit-max-requests", help="Maximum number of requests before worker restart"
    ),
    timeout_keep_alive: int = typer.Option(
        5, "--timeout-keep-alive", help="Keep-alive timeout in seconds"
    ),
):
    """
    Start the ML API server with uvicorn.

    Examples:

        ml-api serve --workers 4 --port 8000

        ml-api serve --reload --log-level debug

        ml-api serve --workers 4 --proxy-headers --forwarded-allow-ips="127.0.0.1,10.0.0.0/8"

        ml-api serve --ssl-keyfile key.pem --ssl-certfile cert.pem
    """
    typer.echo(f"Starting ML API server on {host}:{port} with {workers} worker(s)...")

    uvicorn_config = {
        "app": "app.main:app",
        "host": host,
        "port": port,
        "log_level": log_level.lower(),
        "access_log": access_log,
        "proxy_headers": proxy_headers,
        "timeout_keep_alive": timeout_keep_alive,
    }

    # Add optional parameters only if provided
    if workers > 1:
        uvicorn_config["workers"] = workers
        if reload:
            typer.echo(
                "Warning: --reload is not compatible with multiple workers. Disabling reload."
            )
            reload = False

    if reload:
        uvicorn_config["reload"] = True

    if forwarded_allow_ips:
        uvicorn_config["forwarded_allow_ips"] = forwarded_allow_ips

    if ssl_keyfile:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile

    if ssl_certfile:
        uvicorn_config["ssl_certfile"] = ssl_certfile

    if limit_concurrency:
        uvicorn_config["limit_concurrency"] = limit_concurrency

    if limit_max_requests:
        uvicorn_config["limit_max_requests"] = limit_max_requests

    uvicorn.run(**uvicorn_config)


@app.command()
def version():
    """Show version information."""
    from ml_api.core.config import settings

    typer.echo(f"ML API version {settings.app_version}")
    typer.echo(f"Environment: {settings.environment}")


if __name__ == "__main__":
    app()
