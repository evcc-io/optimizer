DOCKER_IMAGE ?= evcc/optimizer
DOCKER_TAG ?= latest
# Container Apps nodes are amd64; an arm64-only tag takes production down
DOCKER_PLATFORMS ?= linux/amd64,linux/arm64

RESOURCE_GROUP ?= rg-optimizer-prod

default: build docker-build-local

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

docker: docker-build docker-run

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
