"""Testcontainer fixtures with dynamic port allocation."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy, LogMessageWaitStrategy
from testcontainers.postgres import PostgresContainer


@dataclass(frozen=True)
class Container:
    """Connection info for a running testcontainer.

    Attributes:
        host: Container host (always localhost)
        port: Dynamic mapped port
        url: Pre-built connection URL
    """

    host: str
    port: int
    url: str


@contextmanager
def _start_container(
    image: str,
    port: int,
    *,
    env: dict[str, str] | None = None,
    command: str | None = None,
    wait_for_log: str | None = None,
    wait_for_http: tuple[int, str] | None = None,
) -> Generator[Container]:
    """Start a generic Docker container with dynamic port. Internal building block.

    Args:
        image: Docker image
        port: Internal container port to expose
        env: Environment variables
        command: Container command override
        wait_for_log: Log message indicating readiness
        wait_for_http: Tuple of (port, path) for HTTP readiness check

    Yields:
        Container with http:// URL and dynamic port
    """
    c = DockerContainer(image).with_exposed_ports(port)
    if env:
        for k, v in env.items():
            c = c.with_env(k, v)
    if command:
        c = c.with_command(command)
    if wait_for_log:
        c = c.waiting_for(LogMessageWaitStrategy(wait_for_log))
    if wait_for_http:
        http_port, http_path = wait_for_http
        c = c.waiting_for(HttpWaitStrategy(http_port, http_path).for_status_code(200))

    with c:
        mapped = int(c.get_exposed_port(port))
        yield Container(
            host="localhost",
            port=mapped,
            url=f"http://localhost:{mapped}",
        )


@contextmanager
def start_postgres(
    image: str = "postgres:15",
    password: str | None = None,
    username: str = "postgres",
    dbname: str = "postgres",
) -> Generator[Container]:
    """Start a Postgres container with dynamic port.

    Args:
        image: Docker image (postgres:15, timescale/timescaledb:latest-pg15)
        password: DB password. Defaults to POSTGRES_PASSWORD env var.
        username: Database username
        dbname: Database name

    Yields:
        Container with postgresql:// URL and dynamic port
    """
    password = password or os.environ["POSTGRES_PASSWORD"]
    with PostgresContainer(image, username=username, password=password, dbname=dbname) as c:
        port = int(c.get_exposed_port(5432))
        yield Container(
            host="localhost",
            port=port,
            url=f"postgres://{username}:{password}@localhost:{port}/{dbname}",
        )


@contextmanager
def start_mlflow() -> Generator[Container]:
    """Start an MLflow tracking server with dynamic port.

    Yields:
        Container with http:// URL pointing to MLflow UI/API
    """
    with _start_container(
        "ghcr.io/mlflow/mlflow:latest",
        5000,
        command="mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///tmp/mlflow",
        wait_for_log="Application startup complete",
    ) as c:
        yield c


@contextmanager
def start_pushgateway() -> Generator[Container]:
    """Start a Prometheus Pushgateway with dynamic port.

    Yields:
        Container with http:// URL pointing to Pushgateway
    """
    with _start_container(
        "prom/pushgateway:latest",
        9091,
        wait_for_http=(9091, "/metrics"),
    ) as c:
        yield c
