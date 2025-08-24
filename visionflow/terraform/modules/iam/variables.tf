# IAM Module Variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "gke_cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
}

variable "organization_id" {
  description = "Organization ID for organization-level policies"
  type        = string
  default     = null
}

variable "create_service_account_keys" {
  description = "Whether to create service account keys (not recommended for production)"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
