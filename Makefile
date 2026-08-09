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

# ARM throttles replica list and exec with HTTP 429, so every az call goes through
# retry with doubling backoff. Anything that is not a throttle fails on the first try.
AZ_RETRIES     ?= 5
AZ_RETRY_WAIT  ?= 5
export AZ_RETRIES
export AZ_RETRY_WAIT

define AZ_RETRY_SH
retry() { \
	wait=$$AZ_RETRY_WAIT; try=1; \
	while :; do \
		out=$$("$$@" 2>&1) && { printf '%s\n' "$$out"; return 0; }; \
		case "$$out" in \
			*429*|*TooManyRequests*|*Retry-After*|*RateLimit*) ;; \
			*) printf '%s\n' "$$out" >&2; return 1 ;; \
		esac; \
		if [ "$$try" -ge "$$AZ_RETRIES" ]; then \
			printf 'giving up after %s throttled attempts: %s\n' "$$try" "$$out" >&2; \
			return 1; \
		fi; \
		printf 'throttled, retrying in %ss\n' "$$wait" >&2; \
		sleep "$$wait"; wait=$$((wait * 2)); try=$$((try + 1)); \
	done; \
}
endef
export AZ_RETRY_SH

# Each replica keeps its own dump and loses it on restart, so collect from all of them.
# exec needs a TTY, so this only works from an interactive shell. It also leaks a NUL
# every few KiB into the stream, and prefixes its own INFO lines, both stripped here.
# Collected into a temp file first: a throttled run must not overwrite a good log.
# Replicas keep serving the same dump between runs, so new lines are appended to the
# existing log and duplicates dropped, keeping first-seen order.
slow::
	@eval "$$AZ_RETRY_SH"; \
	replicas=$$(retry az containerapp replica list -g $(RESOURCE_GROUP) -n $(CONTAINER_APP) --query "[].name" -o tsv) || exit 1; \
	tmp=$$(mktemp); total=0; failed=0; before=0; \
	if [ -f $(SLOW_LOG) ]; then cat $(SLOW_LOG) >> $$tmp; before=$$(wc -l < $(SLOW_LOG) | tr -d ' '); fi; \
	for replica in $$replicas; do \
		total=$$((total + 1)); \
		if out=$$(retry az containerapp exec -g $(RESOURCE_GROUP) -n $(CONTAINER_APP) \
			--replica "$$replica" --command "cat $(SLOW_LOG_REMOTE)"); then \
			printf '%s\n' "$$out" | tr -d '\0\r' | grep '^{' >> $$tmp || true; \
		else \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ "$$failed" -gt 0 ]; then echo "warning: $$failed of $$total replicas gave no log" >&2; fi; \
	if [ "$$failed" -eq "$$total" ]; then \
		rm -f $$tmp; \
		echo "no replica log collected, keeping $(SLOW_LOG) as is" >&2; \
		exit 1; \
	fi; \
	awk '!seen[$$0]++' $$tmp > $$tmp.dedup && mv $$tmp.dedup $(SLOW_LOG); rm -f $$tmp; \
	after=$$(wc -l < $(SLOW_LOG) | tr -d ' '); \
	echo "$$after slow requests in $(SLOW_LOG) ($$((after - before)) new)"

# ponytail: checks the retry loop only, the az calls themselves stay manual.
slow-check::
	@eval "$$AZ_RETRY_SH"; AZ_RETRY_WAIT=0; AZ_RETRIES=3; log=$$(mktemp); \
	retry sh -c 'echo "(TooManyRequests) Retry-After: 30" >&2; exit 1' 2>$$log; \
	grep -q 'giving up after 3' $$log || { echo "FAIL: throttle not retried to the cap"; exit 1; }; \
	retry sh -c 'echo "(ResourceNotFound) nope" >&2; exit 1' 2>$$log; \
	grep -q 'retrying' $$log && { echo "FAIL: non-throttle error retried"; exit 1; }; \
	out=$$(retry echo '{"ok":1}') && [ "$$out" = '{"ok":1}' ] || { echo "FAIL: success not passed through"; exit 1; }; \
	rm -f $$log; echo "retry guard ok"
