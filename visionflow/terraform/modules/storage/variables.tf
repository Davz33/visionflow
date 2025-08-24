# Storage Module Variables

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

variable "media_bucket_name" {
  description = "Name of the media storage bucket"
  type        = string
}

variable "mlflow_bucket_name" {
  description = "Name of the MLflow artifacts bucket"
  type        = string
}

variable "terraform_bucket_name" {
  description = "Name of the Terraform state bucket"
  type        = string
}

variable "backup_region" {
  description = "Region for backup bucket (defaults to main region)"
  type        = string
  default     = null
}

variable "mlflow_service_account" {
  description = "Service account email for MLflow"
  type        = string
  default     = ""
}

variable "terraform_service_account" {
  description = "Service account email for Terraform"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
