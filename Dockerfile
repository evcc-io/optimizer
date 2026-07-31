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

# Official CBC 2.10.10 build. Fetched on the build platform, it is only unpacked here.
FROM --platform=$BUILDPLATFORM python:3.13-slim AS cbc
ADD --checksum=sha256:f551e7b843e25becee466a447118f6f44f219c4e46cfb4670829ecd3cf47e7d8 \
    https://github.com/coin-or/Cbc/releases/download/releases%2F2.10.10/Cbc-releases.2.10.10-x86_64-ubuntu22-gcc1130-static.tar.gz \
    /tmp/cbc.tar.gz
RUN tar -xzf /tmp/cbc.tar.gz -C /usr/local ./bin/cbc

FROM python:3.13-slim

# Create non-root user
RUN groupadd -r app && useradd -r -g app -s /bin/false app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ARG TARGETARCH

# pulp bundles CBC 2.10.10 for arm64 but 2.10.3 from 2019 for amd64, and resolves the solver by a
# fixed path with no override, so the bundled amd64 binary is linked to the fetched 2.10.10 build.
# The fetched binary is mounted rather than copied, so it does not weigh on the arm64 image.
RUN --mount=from=cbc,source=/usr/local/bin/cbc,target=/tmp/cbc set -eu; \
    if [ "$TARGETARCH" = "amd64" ]; then \
        cp /tmp/cbc /usr/local/bin/cbc; \
        find /app/.venv -path '*/pulp/solverdir/cbc/*/cbc' -exec ln -sf /usr/local/bin/cbc {} \; ; \
    fi; \
    solver=$(/app/.venv/bin/python -c "from pulp.apis.coin_api import pulp_cbc_path; print(pulp_cbc_path)"); \
    echo | "$solver" | grep -q "Version: 2.10.10"; \
    /app/.venv/bin/python -c "import pulp; \
        p = pulp.LpProblem('smoke', pulp.LpMaximize); \
        x = pulp.LpVariable('x', 0, 1, cat='Binary'); \
        p += x; \
        p.solve(pulp.PULP_CBC_CMD(msg=0)); \
        assert pulp.LpStatus[p.status] == 'Optimal', pulp.LpStatus[p.status]"

# Run the application
ENV PYTHONUNBUFFERED=1
ENV OPTIMIZER_TIME_LIMIT=10
ENV OPTIMIZER_NUM_THREADS=1
# the access log is the only source of per request latency, keep the format lean.
# the config module reaps solvers left behind by a killed worker, see gunicorn_conf.py
ENV GUNICORN_CMD_ARGS="--workers 4 --max-requests 32 --config python:optimizer.gunicorn_conf --access-logfile - --access-logformat '%(m)s %(U)s %(s)s %(D)s'"
USER app
CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:7050", "optimizer.app:app"]
