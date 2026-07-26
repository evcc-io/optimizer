# Install uv
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --no-group dev

# Copy the project into the intermediate image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-group dev

FROM python:3.13-slim

# Create non-root user
RUN groupadd -r app && useradd -r -g app -s /bin/false app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Run the application
ENV PYTHONUNBUFFERED=1
ENV OPTIMIZER_TIME_LIMIT=10
ENV OPTIMIZER_NUM_THREADS=1
# the access log is the only source of per request latency, keep the format lean.
# the config module reaps solvers left behind by a killed worker, see gunicorn_conf.py
ENV GUNICORN_CMD_ARGS="--workers 4 --max-requests 32 --config python:optimizer.gunicorn_conf --access-logfile - --access-logformat '%(m)s %(U)s %(s)s %(D)s'"
USER app
CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:7050", "optimizer.app:app"]
