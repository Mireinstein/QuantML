# QuantML dashboard -- Azure deployment.
#
# The azurerm provider reads ARM_CLIENT_ID / ARM_CLIENT_SECRET /
# ARM_TENANT_ID / ARM_SUBSCRIPTION_ID from the environment automatically
# (see python/scripts/set_azure_env.sh) -- no credentials appear in this
# file or in Terraform state's provider config.
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "quantml" {
  name     = "quantml-rg"
  location = var.location
}

# Container registry names must be globally unique across all of Azure --
# this suffix avoids a name collision with someone else's "quantmlacr".
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}
