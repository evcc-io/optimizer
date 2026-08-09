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
