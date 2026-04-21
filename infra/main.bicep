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
          resources: {
            cpu: json('2')
            memory: '4Gi'
          }
          env: [
            { name: 'OPTIMIZER_TIME_LIMIT', value: '18' }
            { name: 'OPTIMIZER_NUM_THREADS', value: '2' }
            {
              name: 'GUNICORN_CMD_ARGS'
              value: '--workers 1 --timeout 40 --max-requests 5000 --max-requests-jitter 500'
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
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
    }
  }
}

output fqdn string = containerApp.properties.configuration.ingress.fqdn
