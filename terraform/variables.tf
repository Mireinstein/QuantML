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
