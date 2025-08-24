# Networking Module Variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "domain" {
  description = "Primary domain for the application"
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "visionflow-cluster"
}

variable "iap_oauth2_client_id" {
  description = "OAuth2 client ID for Identity-Aware Proxy"
  type        = string
  default     = ""
}

variable "iap_oauth2_client_secret" {
  description = "OAuth2 client secret for Identity-Aware Proxy"
  type        = string
  sensitive   = true
  default     = ""
}

variable "blocked_ip_ranges" {
  description = "IP ranges to block in Cloud Armor"
  type        = list(string)
  default = [
    # Common malicious IP ranges - update as needed
    "192.0.2.0/24",    # Test network
    "198.51.100.0/24", # Test network
    "203.0.113.0/24"   # Test network
  ]
}

variable "enable_cdn" {
  description = "Enable Cloud CDN for static content"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
