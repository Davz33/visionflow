# Secrets Module Variables

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

variable "secrets" {
  description = "Map of secrets to create"
  type = map(object({
    secret_data = string
    labels      = optional(map(string))
  }))
  sensitive = true
}

variable "secret_accessors" {
  description = "Map of secret names to service account emails that can access them"
  type        = map(string)
  default     = {}
}

variable "kms_key_id" {
  description = "KMS key ID for encryption (production only)"
  type        = string
  default     = null
}

# Database configuration
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  default     = "visionflow123"
}

variable "db_host" {
  description = "Database host"
  type        = string
  default     = "localhost"
}

# Redis configuration
variable "redis_host" {
  description = "Redis host"
  type        = string
  default     = "localhost"
}

variable "redis_port" {
  description = "Redis port"
  type        = string
  default     = "6379"
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
  default     = "redis123"
}

# MLFlow configuration
variable "mlflow_password" {
  description = "MLFlow database password"
  type        = string
  sensitive   = true
  default     = "mlflow123"
}

variable "mlflow_bucket_name" {
  description = "MLFlow artifacts bucket name"
  type        = string
  default     = "visionflow-mlflow-artifacts"
}

# Service account configuration
variable "gcp_service_account_key" {
  description = "Google Cloud service account key JSON"
  type        = string
  sensitive   = true
  default     = null
}

# TLS certificates
variable "tls_certificates" {
  description = "TLS certificates for internal services"
  type        = map(string)
  sensitive   = true
  default     = {}
}

# Secret rotation
variable "enable_secret_rotation" {
  description = "Enable automatic secret rotation"
  type        = bool
  default     = false
}

variable "function_source_bucket" {
  description = "Bucket containing Cloud Function source code"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
