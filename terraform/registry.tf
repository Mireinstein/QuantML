# Basic SKU (~$5/month) -- the cheapest tier that still supports az acr
# build (building the image in the cloud, so this deployment doesn't
# depend on Docker being installed locally). Admin credentials (rather
# than a user-assigned managed identity + AcrPull role assignment) are
# used for the Container App to pull images: the deploy service principal
# only has the Contributor role, which deliberately excludes
# Microsoft.Authorization/roleAssignments/write (granting that would be a
# privilege-escalation path), so this app can't assign itself an AcrPull
# role. Admin credentials are the correct choice given that constraint --
# a managed identity would be the better practice with a more privileged
# (User Access Administrator) deploy principal.
resource "azurerm_container_registry" "acr" {
  name                = "quantiqacr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.quantiq.name
  location            = azurerm_resource_group.quantiq.location
  sku                 = "Basic"
  admin_enabled       = true
}
