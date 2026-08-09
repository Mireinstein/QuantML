# Container Apps: consumption plan, scales to zero when idle (no traffic =
# no compute cost beyond the ACR registry's flat ~$5/month). The generous
# free monthly allowance (180K vCPU-seconds / 360K GiB-seconds) covers a
# low-traffic demo dashboard comfortably.
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "quantiq-logs"
  resource_group_name = azurerm_resource_group.quantiq.name
  location            = azurerm_resource_group.quantiq.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "env" {
  name                       = "quantiq-env"
  resource_group_name        = azurerm_resource_group.quantiq.name
  location                   = azurerm_resource_group.quantiq.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

resource "azurerm_container_app" "dashboard" {
  name                         = "quantiq-dashboard"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.quantiq.name
  revision_mode                = "Single"

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.acr.admin_password
  }

  registry {
    server               = azurerm_container_registry.acr.login_server
    username              = azurerm_container_registry.acr.admin_username
    password_secret_name  = "acr-password"
  }

  template {
    container {
      name   = "quantiq-dashboard"
      image  = "${azurerm_container_registry.acr.login_server}/quantiq-dashboard:${var.image_tag}"
      cpu    = 0.25
      memory = "0.5Gi"
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
