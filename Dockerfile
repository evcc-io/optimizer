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

ADD --checksum=sha256:f551e7b843e25becee466a447118f6f44f219c4e46cfb4670829ecd3cf47e7d8 \
    https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.10/Cbc-releases.2.10.10-x86_64-ubuntu22-gcc1130-static.tar.gz \
    /tmp/cbc.tar.gz

FROM python:3.13-slim

# Create non-root user
RUN groupadd -r app && useradd -r -g app -s /bin/false app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ARG TARGETARCH

# pulp bundles CBC 2.10.10 for arm64 but 2.10.3 from 2019 for amd64, and resolves the solver by a
# fixed path with no override, so the bundled amd64 binary is linked to the fetched 2.10.10 build.
RUN --mount=from=builder,source=/tmp/cbc.tar.gz,target=/tmp/cbc.tar.gz set -eu; \
    if [ "$TARGETARCH" = "amd64" ]; then \
        tar -xzf /tmp/cbc.tar.gz -C /usr/local ./bin/cbc; \
        find /app/.venv -path '*/pulp/solverdir/cbc/*/cbc' -exec ln -sf /usr/local/bin/cbc {} \; ; \
    fi; \
    echo | "$(/app/.venv/bin/python -c 'from pulp.apis.coin_api import pulp_cbc_path; print(pulp_cbc_path)')" \
        | grep -q 'Version: 2.10.10'

# Run the application
ENV PYTHONUNBUFFERED=1
ENV OPTIMIZER_TIME_LIMIT=10
ENV OPTIMIZER_NUM_THREADS=1
# the access log is the only source of per request latency, keep the format lean.
# the config module reaps solvers left behind by a killed worker, see gunicorn_conf.py
ENV GUNICORN_CMD_ARGS="--workers 4 --max-requests 32 --config python:optimizer.gunicorn_conf --access-logfile - --access-logformat '%(m)s %(U)s %(s)s %(D)s'"
USER app
CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:7050", "optimizer.app:app"]
