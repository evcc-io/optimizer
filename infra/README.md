# Infrastructure

One Bicep template, `main.bicep`, deployed by `.github/workflows/deploy.yml`.

## Layout

- **VM floor**: `optimizer-vm-0` (regular, buy a 1 year reservation for its size) plus the
  `optimizer-spot` Flexible scale set (Spot, try-and-restore on eviction). Each VM runs
  HAProxy and one gunicorn worker per vCPU from `vm/docker-compose.yml`. A Standard Load
  Balancer on `optimizer-lb-ip` fronts them on 80/443 and provides their outbound SNAT.
- **Container Apps** `optimizer`: overflow only. `minReplicas: 0`; HAProxy sends a request
  there when every local worker is busy or the local container is down (deploy drain,
  restart). Set `restrictIngressToVms=true` once DNS points at the load balancer so only
  the VMs can reach it.
- **TLS**: certbot on the acme owner (`optimizer-vm-0`, tag `acme=owner`) answers HTTP-01 on
  port 8402, every VM's HAProxy forwards `/.well-known/acme-challenge/` there. The deploy
  hook stores fullchain+key as Key Vault secret `tls-<hostname with dashes>`, the other VMs
  pull it every 30 minutes. Until the first issuance HAProxy serves a self-signed placeholder.
- **Secrets**: the VMs share the user-assigned identity `optimizer-vm-id` with *Key Vault
  Secrets Officer* on `kv-optimizer-prod`, read `jwt-token-secret`, write the certificate.
- **Image**: never in cloud-init. The VM tag `image` carries the deployed tag, `update.sh`
  reads it through IMDS at boot (first boot, Spot restore). `deploy.yml` runs
  `update.sh <image>` on every VM after the Bicep step, draining HAProxy first.

## Cutover from Container Apps only

1. Add the repository secret `VM_SSH_PUBLIC_KEY` (any ed25519 public key; day to day access
   is `az vm run-command`, the key is a break-glass).
   The deploy service principal holds Contributor on the resource group and cannot write
   role assignments, so the identity and its Key Vault role are created once by an Owner and
   only adopted by Bicep:
   `az identity create -g rg-optimizer-prod -n optimizer-vm-id -l germanywestcentral` and
   `az role assignment create --role "Key Vault Secrets Officer" --assignee-object-id <principalId> --assignee-principal-type ServicePrincipal --scope <kv-optimizer-prod id>`.
   Done 2026-10-04.
2. Run Deploy. DNS still points at Container Apps, nothing changes for clients. Check the floor:
   `curl -k --resolve optimizer.evcc.io:443:<lb ip> https://optimizer.evcc.io/` (self-signed).
3. Point the `optimizer.evcc.io` A record at the load balancer IP (Bicep output `lbIp`).
   Within the hour certbot issues the real certificate and all VMs pick it up.
4. Run Deploy with `restrictIngressToVms=true` (workflow input). Container Apps now serves
   the VMs only. The custom domain binding on Container Apps can go in a later cleanup.
5. Buy the reservation: Reservations → Virtual machines → `D4pls v6`, germanywestcentral, 1 year, quantity 1.

## Operations

- Deploy: `gh workflow run deploy.yml --ref <branch> -f image_tag=<sha>`.
- Ad hoc on a VM: `az vm run-command invoke -g rg-optimizer-prod -n optimizer-vm-0 --command-id RunShellScript --scripts "docker logs --tail 50 optimizer-optimizer-1"`.
- Logs stay on the VMs (`docker logs`) for now. The Container Apps Log Analytics queries only
  see the overflow share; shipping the VM access log with the Azure Monitor Agent is the next step.
- Spot eviction: the scale set restores the instance when capacity returns, up to 1 hour.
  Meanwhile `optimizer-vm-0` plus Container Apps carry the load.
