# Container Apps: consumption plan. The dashboard scales to zero when
# idle (no traffic = no compute cost beyond the ACR registry's flat
# ~$5/month). The trader app is the exception: it runs the autonomous
# loop continuously (a `while True` process, not a request handler), so
# it CANNOT scale to zero -- min_replicas = 1 below means it's billed for
# compute the whole time it exists, not just when someone's watching. At
# 0.25 vCPU / 0.5Gi that's roughly the free monthly allowance's worth
# every ~8 days, so expect a real (if modest, single-digit-to-low-teens
# dollars/month) charge beyond the ACR flat fee once it's been running a
# full month -- unlike everything else in this deployment, which stays
# inside the free tier.
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "quantml-logs"
  resource_group_name = azurerm_resource_group.quantml.name
  location            = azurerm_resource_group.quantml.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "env" {
  name                       = "quantml-env"
  resource_group_name        = azurerm_resource_group.quantml.name
  location                   = azurerm_resource_group.quantml.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

# Runs autonomous.run() in a background thread behind a tiny internal
# control API (trader_service.py) -- /pause and /resume, called by the
# dashboard's (password-gated) start/stop buttons. Internal-only ingress:
# not reachable from the public internet, only from other apps in this
# same Container Apps environment. Starts PAUSED (see trader_service.py's
# module docstring) -- deploying or restarting this app never starts
# real trading on its own; someone has to click "Start."
resource "azurerm_container_app" "trader" {
  name                         = "quantml-trader"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.quantml.name
  revision_mode                = "Single"

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.acr.admin_password
  }
  secret {
    name  = "alpaca-api-key-id"
    value = var.alpaca_api_key_id
  }
  secret {
    name  = "alpaca-api-secret-key"
    value = var.alpaca_api_secret_key
  }

  registry {
    server               = azurerm_container_registry.acr.login_server
    username             = azurerm_container_registry.acr.admin_username
    password_secret_name = "acr-password"
  }

  template {
    container {
      name    = "quantml-trader"
      image   = "${azurerm_container_registry.acr.login_server}/quantml-dashboard:${var.image_tag}"
      cpu     = 0.25
      memory  = "0.5Gi"
      command = ["uvicorn", "quantml.trader_service:app", "--host", "0.0.0.0", "--port", "8080"]

      env {
        name        = "ALPACA_API_KEY_ID"
        secret_name = "alpaca-api-key-id"
      }
      env {
        name        = "ALPACA_API_SECRET_KEY"
        secret_name = "alpaca-api-secret-key"
      }
    }
    min_replicas = 1
    max_replicas = 1
  }

  ingress {
    external_enabled = false
    target_port       = 8080
    transport          = "auto"
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

resource "azurerm_container_app" "dashboard" {
  name                         = "quantml-dashboard"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.quantml.name
  revision_mode                = "Single"

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.acr.admin_password
  }
  secret {
    name  = "alpaca-api-key-id"
    value = var.alpaca_api_key_id
  }
  secret {
    name  = "alpaca-api-secret-key"
    value = var.alpaca_api_secret_key
  }
  secret {
    name  = "dashboard-password"
    value = var.dashboard_password
  }

  registry {
    server               = azurerm_container_registry.acr.login_server
    username              = azurerm_container_registry.acr.admin_username
    password_secret_name  = "acr-password"
  }

  template {
    container {
      name   = "quantml-dashboard"
      image  = "${azurerm_container_registry.acr.login_server}/quantml-dashboard:${var.image_tag}"
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name        = "ALPACA_API_KEY_ID"
        secret_name = "alpaca-api-key-id"
      }
      env {
        name        = "ALPACA_API_SECRET_KEY"
        secret_name = "alpaca-api-secret-key"
      }
      env {
        name  = "DASHBOARD_USERNAME"
        value = var.dashboard_username
      }
      env {
        name        = "DASHBOARD_PASSWORD"
        secret_name = "dashboard-password"
      }
      # https:// -- Container Apps' internal ingress redirects plain HTTP
      # to HTTPS, and that redirect silently turns a POST into a GET
      # (standard redirect-following behavior), which broke /pause and
      # /resume with a 405 the first time this was deployed with http://
      # here. Traffic still never leaves the Container Apps environment's
      # own network either way -- this only changes the URL scheme, not
      # where the request can go.
      env {
        name  = "TRADER_INTERNAL_URL"
        value = "https://${azurerm_container_app.trader.ingress[0].fqdn}"
      }
    }
    min_replicas = 0
    max_replicas = 1
  }

  ingress {
    external_enabled = true
    target_port       = 8080
    transport          = "auto"
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}
