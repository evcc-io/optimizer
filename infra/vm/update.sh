#!/bin/bash
# Pulls the shared TLS certificate and the configured image, then (re)starts the stack.
# Runs from cloud-init at first boot, from cron for the certificate, and from deploy.yml
# through `az vm run-command` on every deploy.
#   update.sh              image from the VM's `image` tag (main.bicep sets it)
#   update.sh <image>      explicit image, used by deploy.yml
#   update.sh --cert-only  certificate refresh only
set -euo pipefail
cd /opt/optimizer

KV=__KV__
HOST=__HOSTNAME__
CERT=certs/$HOST.pem
SECRET=tls-${HOST//./-}

imds() { curl -sf -H Metadata:true "http://169.254.169.254/metadata/$1"; }
tag() { imds 'instance/compute/tagsList?api-version=2021-02-01' | jq -r --arg n "$1" '.[] | select(.name==$n) | .value'; }
kv() {
  local token
  token=$(imds 'identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net' | jq -r .access_token)
  curl -sf -H "Authorization: Bearer $token" "https://$KV.vault.azure.net/secrets/$1?api-version=7.4" | jq -r .value
}
hap() { echo "$1" | nc -U -w1 /var/run/haproxy/admin.sock; }

# the certificate is issued on the acme owner (acme-hook.sh) and distributed through Key Vault
if pem=$(kv "$SECRET") && [ -n "$pem" ] && ! cmp -s <(printf '%s\n' "$pem") "$CERT"; then
  printf '%s\n' "$pem" > "$CERT.new" && mv "$CERT.new" "$CERT"
  docker compose kill -s HUP haproxy 2>/dev/null || true
fi

if [ "$(tag acme)" = owner ] && [ ! -f /etc/cron.d/optimizer-acme ]; then
  # hourly until the first certificate exists; certbot's own timer renews it from then on
  echo "17 * * * * root test -d /etc/letsencrypt/live/$HOST || certbot certonly --standalone --http-01-port 8402 -n --agree-tos --register-unsafely-without-email -d $HOST --deploy-hook /opt/optimizer/acme-hook.sh" > /etc/cron.d/optimizer-acme
fi

[ "${1:-}" = --cert-only ] && exit 0

IMAGE=${1:-$(tag image)}
printf 'IMAGE=%s\nJWT_TOKEN_SECRET=%s\n' "$IMAGE" "$(kv jwt-token-secret)" > .env
chmod 600 .env

# drain the local workers: in-flight solves finish, new requests spill to Container Apps meanwhile
if [ -S /var/run/haproxy/admin.sock ]; then
  hap 'set server local/workers state drain' || true
  for _ in $(seq 1 60); do
    [ "$(hap 'show servers conn local' | awk 'NR==2 {print $7}')" = 0 ] && break
    sleep 1
  done
fi

docker compose pull -q
docker compose up -d --remove-orphans
for _ in $(seq 1 30); do curl -sf -o /dev/null localhost:7050/ && break; sleep 1; done
hap 'set server local/workers state ready' 2>/dev/null || true
