@description('Container image to deploy')
param containerImage string = 'evcc/optimizer:latest'

@description('Azure region for all resources')
param location string = 'germanywestcentral'

// Capacity floor: one regular VM (reserve it for a year) plus Spot instances, all behind one
// load balancer. Container Apps keeps running as overflow only, see infra/README.md.
// Dpls v6 is the lowest memory ratio ARM size: each vCPU is a full Cobalt core (no SMT) and
// benchmarked equal to the AMD core in single thread, 16% ahead with four concurrent solves.
@description('VM size for the floor; one gunicorn worker per vCPU')
param vmSize string = 'Standard_D4pls_v6'

@description('vCPUs of vmSize, drives the worker count and the HAProxy spill threshold')
param vmCpus int = 4

@description('Spot instances next to the regular VM')
param spotCount int = 1

@description('SSH public key for the break-glass admin user; routine access goes through az vm run-command')
param vmSshPublicKey string

@description('Restrict Container Apps ingress to the VM floor; enable once DNS points at the load balancer')
param restrictIngressToVms bool = false

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-optimizer-prod'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2025-02-01' = {
  name: 'optimizer-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

@description('Custom hostname served by the container app')
param customHostname string = 'optimizer.evcc.io'

@description('Name of the existing managed certificate in the environment for customHostname')
param managedCertificateName string = 'mc-optimizer-env-optimizer-evcc-i-5846'

resource containerAppEnv 'Microsoft.App/managedEnvironments@2025-01-01' = {
  name: 'optimizer-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2025-01-01' = {
  name: 'optimizer'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 7050
        customDomains: [
          {
            name: customHostname
            bindingType: 'SniEnabled'
            certificateId: '${containerAppEnv.id}/managedCertificates/${managedCertificateName}'
          }
        ]
        // overflow traffic arrives from the VMs through the load balancer's outbound SNAT
        ipSecurityRestrictions: restrictIngressToVms ? [
          {
            name: 'vm-floor'
            description: 'HAProxy overflow from the VM floor'
            ipAddressRange: '${lbIp.properties.ipAddress}/32'
            action: 'Allow'
          }
        ] : null
      }
      secrets: [
        {
          name: 'jwt-token-secret'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/jwt-token-secret'
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'optimizer'
          image: containerImage
          // one vCPU per replica keeps allocation close to demand. The solve is CPU bound,
          // so a coarser replica rounds up into cores that are paid for and never used.
          resources: {
            cpu: json('1')
            memory: '2Gi'
          }
          env: [
            // requests that reach the limit walk a cost optimal plateau rather than close a gap:
            // replaying 20 collected ones, 17 end on the same objective at 10 s as at 20 s. The
            // exception cost 4.7 percent, so this halves the latency and the core they hold on a
            // one vCPU replica at the price of a worse schedule for a small share of them.
            { name: 'OPTIMIZER_TIME_LIMIT', value: '10' }
            { name: 'OPTIMIZER_NUM_THREADS', value: '1' }
            // the dump threshold is the time limit, so this collects everything above 10 s now.
            // The file is ephemeral, a replica restart takes it with it.
            { name: 'OPTIMIZER_DUMP_SLOW_REQUESTS', value: '/tmp/slow-requests.jsonl' }
            {
              name: 'GUNICORN_CMD_ARGS'
              // one worker per vCPU, and the replica carries one. Two workers on one core let
              // a pair of concurrent solves halve each other's speed, which pushed a 20 s solve
              // past the request timeout and cost a worker, and with it a core, for good.
              // the timeout sits above the worst elapsed time seen in production, 29 s, so it
              // catches a genuinely stuck request without cutting a legitimate solve short.
              // the config module reaps a solver that outlived its worker anyway.
              // the access log is the only source of per request latency. %(D)s is the
              // response time in microseconds, the rest of the format stays lean on purpose.
              // the jitter is a spread around max-requests, not a second budget: at 500 against
              // 100 a worker recycled somewhere between 100 and 600 requests, so at roughly one
              // request per second per replica it restarted every 2 to 10 minutes and paid a cold
              // start each time. 50 keeps the staggering that stops replicas recycling in lockstep.
              value: '--workers 1 --timeout 60 --max-requests 100 --max-requests-jitter 50 --config python:optimizer.gunicorn_conf --access-logfile - --access-logformat \'%(m)s %(U)s %(s)s %(D)s\''
            }
            { name: 'JWT_TOKEN_SECRET', secretRef: 'jwt-token-secret' }
          ]
          probes: [
            {
              type: 'startup'
              tcpSocket: {
                port: 7050
              }
              periodSeconds: 5
              failureThreshold: 10
            }
            // without an explicit readiness probe the platform polls its own from the moment the
            // container exists, and the app is not listening yet: importing it alone takes over
            // three seconds before gunicorn binds. That produced 419 'readiness probe failed:
            // connection refused' warnings and 258 container starts in 24 hours, every one of them
            // billed cold time serving nothing.
            {
              type: 'readiness'
              // tcp, not http against the health route. One worker per replica means a ten second
              // solve owns the whole process, so an http probe would time out mid solve and take
              // a healthy replica out of rotation for doing exactly what it is there to do.
              tcpSocket: {
                port: 7050
              }
              initialDelaySeconds: 20
              periodSeconds: 10
              failureThreshold: 6
            }
          ]
        }
      ]
      scale: {
        // overflow only: the VM floor carries the base load, replicas exist while it spills
        minReplicas: 0
        maxReplicas: 50
        // KEDA takes the maximum over both rules. The CPU rule is the one that matches the
        // bottleneck, the concurrency rule stays as a fast reacting guard for bursts.
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '8'
              }
            }
          }
          {
            name: 'cpu-scaling'
            custom: {
              type: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '75'
              }
            }
          }
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------------------
// VM floor
// ---------------------------------------------------------------------------------------

var acaFqdn = 'optimizer.${containerAppEnv.properties.defaultDomain}'
// first usable address of the subnet, pinned so every VM can forward ACME challenges to it
var acmeOwnerIp = '10.10.0.4'

// The VM files live next to this template. Placeholders are filled here, then the whole set is
// embedded into cloud-init as base64 so YAML indentation never gets a say. Nothing in the result
// changes between deploys: customData is immutable on a VM and a Spot restore boots from it.
var haproxyCfg = replace(replace(replace(loadTextContent('vm/haproxy.cfg'), '__ACA_FQDN__', acaFqdn), '__ACME_OWNER_IP__', acmeOwnerIp), '__WORKERS__', string(vmCpus))
var composeYml = replace(loadTextContent('vm/docker-compose.yml'), '__WORKERS__', string(vmCpus))
var updateSh = replace(replace(loadTextContent('vm/update.sh'), '__KV__', keyVault.name), '__HOSTNAME__', customHostname)
var acmeHookSh = replace(replace(loadTextContent('vm/acme-hook.sh'), '__KV__', keyVault.name), '__HOSTNAME__', customHostname)
var cloudInit = base64(replace(replace(replace(replace(replace(loadTextContent('vm/cloud-init.yaml'), '__HAPROXY_B64__', base64(haproxyCfg)), '__COMPOSE_B64__', base64(composeYml)), '__UPDATE_B64__', base64(updateSh)), '__ACME_HOOK_B64__', base64(acmeHookSh)), '__HOSTNAME__', customHostname))

var ubuntuArm64 = {
  publisher: 'Canonical'
  offer: 'ubuntu-24_04-lts'
  sku: 'server-arm64'
  version: 'latest'
}

// update.sh reads `image` through IMDS at boot, so a fresh instance starts on the deployed tag
var vmTags = {
  role: 'optimizer'
  image: containerImage
}

var osDisk = {
  createOption: 'FromImage'
  diskSizeGB: 30
  managedDisk: {
    storageAccountType: 'StandardSSD_LRS'
  }
}

var linuxConfiguration = {
  disablePasswordAuthentication: true
  ssh: {
    publicKeys: [
      {
        path: '/home/optimizer/.ssh/authorized_keys'
        keyData: vmSshPublicKey
      }
    ]
  }
}

// one identity for every floor VM: read jwt-token-secret, write the shared TLS certificate.
// Its Key Vault Secrets Officer assignment is not declared here: the deploy service principal
// is a Contributor and cannot write role assignments, so an Owner created it once by hand
// (infra/README.md).
resource vmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'optimizer-vm-id'
  location: location
}

resource vmNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: 'optimizer-vm-nsg'
  location: location
  properties: {
    // no SSH from the internet, management goes through az vm run-command
    securityRules: [
      {
        name: 'http'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: 'Internet'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '80'
        }
      }
      {
        name: 'https'
        properties: {
          priority: 110
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: 'Internet'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '443'
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: 'optimizer-vnet'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: ['10.10.0.0/24']
    }
    subnets: [
      {
        name: 'vms'
        properties: {
          addressPrefix: '10.10.0.0/24'
          networkSecurityGroup: {
            id: vmNsg.id
          }
        }
      }
    ]
  }
}

resource lbIp 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: 'optimizer-lb-ip'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

var lbFrontendId = resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', 'optimizer-lb', 'public')
var lbPoolId = resourceId('Microsoft.Network/loadBalancers/backendAddressPools', 'optimizer-lb', 'vms')
var lbProbeId = resourceId('Microsoft.Network/loadBalancers/probes', 'optimizer-lb', 'https')

resource lb 'Microsoft.Network/loadBalancers@2024-05-01' = {
  name: 'optimizer-lb'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    frontendIPConfigurations: [
      {
        name: 'public'
        properties: {
          publicIPAddress: {
            id: lbIp.id
          }
        }
      }
    ]
    backendAddressPools: [
      {
        name: 'vms'
      }
    ]
    probes: [
      {
        name: 'https'
        properties: {
          protocol: 'Tcp'
          port: 443
          intervalInSeconds: 5
          numberOfProbes: 2
        }
      }
    ]
    loadBalancingRules: [for port in [80, 443]: {
      name: 'tcp-${port}'
      properties: {
        frontendIPConfiguration: { id: lbFrontendId }
        backendAddressPool: { id: lbPoolId }
        probe: { id: lbProbeId }
        protocol: 'Tcp'
        frontendPort: port
        backendPort: port
        idleTimeoutInMinutes: 4
        // outbound goes through the explicit rule below, not the inbound rules' SNAT
        disableOutboundSnat: true
      }
    }]
    // the VMs have no public IP of their own; this is their path to Docker Hub, Key Vault and Container Apps
    outboundRules: [
      {
        name: 'internet'
        properties: {
          frontendIPConfigurations: [{ id: lbFrontendId }]
          backendAddressPool: { id: lbPoolId }
          protocol: 'All'
          allocatedOutboundPorts: 0
          idleTimeoutInMinutes: 4
          enableTcpReset: true
        }
      }
    ]
  }
}

resource vm0Nic 'Microsoft.Network/networkInterfaces@2024-05-01' = {
  name: 'optimizer-vm-0-nic'
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig'
        properties: {
          subnet: { id: vnet.properties.subnets[0].id }
          privateIPAllocationMethod: 'Static'
          privateIPAddress: acmeOwnerIp
          loadBalancerBackendAddressPools: [{ id: lbPoolId }]
        }
      }
    ]
  }
  dependsOn: [lb]
}

// the regular VM: always there, carries the Let's Encrypt account, worth a one year reservation
resource vm0 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: 'optimizer-vm-0'
  location: location
  tags: union(vmTags, { acme: 'owner' })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${vmIdentity.id}': {} }
  }
  properties: {
    hardwareProfile: { vmSize: vmSize }
    storageProfile: {
      imageReference: ubuntuArm64
      osDisk: osDisk
    }
    osProfile: {
      computerName: 'optimizer-vm-0'
      adminUsername: 'optimizer'
      customData: cloudInit
      linuxConfiguration: linuxConfiguration
    }
    networkProfile: {
      networkInterfaces: [{ id: vm0Nic.id }]
    }
  }
}

// Spot capacity: evicted instances are deallocated and restored by the platform when capacity
// returns, the regular VM and Container Apps carry the load in between
resource spot 'Microsoft.Compute/virtualMachineScaleSets@2024-07-01' = {
  name: 'optimizer-spot'
  location: location
  tags: vmTags
  sku: {
    name: vmSize
    tier: 'Standard'
    capacity: spotCount
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${vmIdentity.id}': {} }
  }
  properties: {
    orchestrationMode: 'Flexible'
    platformFaultDomainCount: 1
    spotRestorePolicy: {
      enabled: true
      restoreTimeout: 'PT1H'
    }
    virtualMachineProfile: {
      priority: 'Spot'
      evictionPolicy: 'Deallocate'
      billingProfile: { maxPrice: -1 }
      storageProfile: {
        imageReference: ubuntuArm64
        osDisk: osDisk
      }
      osProfile: {
        computerNamePrefix: 'optimizer-spot-'
        adminUsername: 'optimizer'
        customData: cloudInit
        linuxConfiguration: linuxConfiguration
      }
      networkProfile: {
        networkApiVersion: '2020-11-01'
        networkInterfaceConfigurations: [
          {
            name: 'nic'
            properties: {
              primary: true
              networkSecurityGroup: { id: vmNsg.id }
              ipConfigurations: [
                {
                  name: 'ipconfig'
                  properties: {
                    subnet: { id: vnet.properties.subnets[0].id }
                    loadBalancerBackendAddressPools: [{ id: lbPoolId }]
                  }
                }
              ]
            }
          }
        ]
      }
    }
  }
  dependsOn: [lb]
}

output fqdn string = containerApp.properties.configuration.ingress.fqdn
output lbIp string = lbIp.properties.ipAddress
