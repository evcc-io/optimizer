#!/bin/bash
# certbot deploy hook on the acme owner: publish the fresh certificate through Key Vault,
# the other VMs pick it up in update.sh --cert-only within 30 minutes.
set -euo pipefail

KV=__KV__
HOST=__HOSTNAME__

pem=$(cat "/etc/letsencrypt/live/$HOST/fullchain.pem" "/etc/letsencrypt/live/$HOST/privkey.pem")
token=$(curl -sf -H Metadata:true 'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net' | jq -r .access_token)
jq -n --arg v "$pem" '{value: $v, contentType: "application/x-pem-file"}' |
  curl -sf -X PUT -H "Authorization: Bearer $token" -H 'Content-Type: application/json' -d @- \
    "https://$KV.vault.azure.net/secrets/tls-${HOST//./-}?api-version=7.4" > /dev/null

/opt/optimizer/update.sh --cert-only
