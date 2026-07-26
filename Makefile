DOCKER_IMAGE     ?= evcc/optimizer
DOCKER_TAG       ?= latest
# Container Apps nodes are amd64; an arm64-only tag takes production down
DOCKER_PLATFORMS ?= linux/amd64,linux/arm64

RESOURCE_GROUP   ?= rg-optimizer-prod
CONTAINER_APP    ?= optimizer

# OPTIMIZER_DUMP_SLOW_REQUESTS in infra/main.bicep, and where make slow collects it
SLOW_LOG_REMOTE  ?= /tmp/slow-requests.jsonl
SLOW_LOG         ?= slow-requests.jsonl

default:: build docker-build-local

build::
	go generate ./...

install::
	uv sync

upgrade::
	uv lock --upgrade

lint::
	uv run autopep8 --in-place --recursive .
	uv run ruff check --fix

test::
	uv run pytest

run::
	uv run python -m optimizer.app

run-gunicorn::
	uv run gunicorn --bind "0.0.0.0:7050" "optimizer.app:app"

loadtest::
	uv run locust --host http://localhost:7050 --headless -t 30s -u 2 --only-summary
	uv run locust --host http://localhost:7050 --headless -t 30s -u 4 --only-summary

docker:: docker-build docker-run

# Buildx cannot load a manifest list into the local image store, so a
# multi-platform tag only exists once pushed to the registry.
docker-build::
	docker buildx build . --platform $(DOCKER_PLATFORMS) --tag $(DOCKER_IMAGE):$(DOCKER_TAG) --push

# Host architecture only, loaded locally for docker-run. Never push this.
docker-build-local::
	docker buildx build . --tag $(DOCKER_IMAGE):$(DOCKER_TAG) --load

docker-run:: docker-build-local
	docker run -p 7050:7050 -it $(DOCKER_IMAGE):$(DOCKER_TAG)

fly::
	fly deploy --local-only

deploy::
	az deployment group create \
		--resource-group $(RESOURCE_GROUP) \
		--template-file infra/main.bicep \
		--parameters containerImage=$(DOCKER_IMAGE):$(DOCKER_TAG)

# Each replica keeps its own dump and loses it on restart, so collect from all of them.
# exec needs a TTY, so this only works from an interactive shell. It also leaks a NUL
# every few KiB into the stream, and prefixes its own INFO lines, both stripped here.
slow::
	@for replica in $$(az containerapp replica list -g $(RESOURCE_GROUP) -n $(CONTAINER_APP) --query "[].name" -o tsv); do \
		az containerapp exec -g $(RESOURCE_GROUP) -n $(CONTAINER_APP) --replica "$$replica" \
			--command "cat $(SLOW_LOG_REMOTE)"; \
	done | tr -d '\0\r' | grep '^{' > $(SLOW_LOG) || true
	@echo "$$(wc -l < $(SLOW_LOG) | tr -d ' ') slow requests collected in $(SLOW_LOG)"
