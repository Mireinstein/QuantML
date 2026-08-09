variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "image_tag" {
  description = "Tag of the quantml-dashboard image in ACR to deploy"
  type        = string
  default     = "latest"
}

# The four variables below are never given defaults and are marked
# sensitive so `terraform plan`/`apply` never print their values. Set
# them via TF_VAR_* environment variables (see
# python/scripts/set_azure_trading_env.sh), never by hardcoding a value
# here or in a committed .tfvars file.

variable "alpaca_api_key_id" {
  description = "Alpaca PAPER trading API key ID -- used by both the dashboard (on-demand trades) and the trader Container App (autonomous loop)"
  type        = string
  sensitive   = true
}

variable "alpaca_api_secret_key" {
  description = "Alpaca PAPER trading API secret key"
  type        = string
  sensitive   = true
}

variable "dashboard_username" {
  description = "HTTP Basic Auth username gating the dashboard's action endpoints (place a trade, start/stop the bot) -- read-only endpoints stay public regardless"
  type        = string
  sensitive   = true
}

variable "dashboard_password" {
  description = "HTTP Basic Auth password for dashboard_username"
  type        = string
  sensitive   = true
}
