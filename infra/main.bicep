@description('Container image to deploy')
param containerImage string = 'evcc/optimizer:latest'

@description('Azure region for all resources')
param location string = 'germanywestcentral'

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
            { name: 'OPTIMIZER_TIME_LIMIT', value: '20' }
            { name: 'OPTIMIZER_NUM_THREADS', value: '1' }
            // roughly 2 percent of requests exhaust the time limit. Keep them for replay.
            // The file is ephemeral, a replica restart takes it with it.
            { name: 'OPTIMIZER_DUMP_SLOW_REQUESTS', value: '/tmp/slow-requests.jsonl' }
            {
              name: 'GUNICORN_CMD_ARGS'
              // one worker per vCPU. Oversubscribing a CPU bound solver only moves the queue
              // from the ingress into the kernel scheduler and inflates tail latency.
              // the access log is the only source of per request latency. %(D)s is the
              // response time in microseconds, the rest of the format stays lean on purpose.
              value: '--workers 2 --timeout 40 --max-requests 100 --max-requests-jitter 500 --access-logfile - --access-logformat \'%(m)s %(U)s %(s)s %(D)s\''
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
          ]
        }
      ]
      scale: {
        minReplicas: 1
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

output fqdn string = containerApp.properties.configuration.ingress.fqdn
