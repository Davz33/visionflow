# Database Module Variables

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

variable "vpc_network_id" {
  description = "VPC network ID"
  type        = string
}

# PostgreSQL Configuration
variable "postgres_instance_name" {
  description = "Name of the PostgreSQL instance"
  type        = string
}

variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "POSTGRES_15"
}

variable "postgres_tier" {
  description = "PostgreSQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "postgres_disk_size" {
  description = "PostgreSQL disk size in GB"
  type        = number
  default     = 20
}

variable "postgres_max_disk_size" {
  description = "Maximum PostgreSQL disk size in GB"
  type        = number
  default     = 100
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
  default     = "visionflow123"
}

variable "mlflow_password" {
  description = "MLFlow PostgreSQL password"
  type        = string
  sensitive   = true
  default     = "mlflow123"
}

# Redis Configuration
variable "redis_instance_name" {
  description = "Name of the Redis instance"
  type        = string
}

variable "redis_tier" {
  description = "Redis tier"
  type        = string
  default     = "STANDARD_HA"
}

variable "redis_memory_size" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_7_0"
}

variable "redis_location_id" {
  description = "Redis location ID"
  type        = string
  default     = null
}

variable "redis_alternative_location_id" {
  description = "Redis alternative location ID"
  type        = string
  default     = null
}

variable "redis_reserved_ip_range" {
  description = "Reserved IP range for Redis"
  type        = string
  default     = "10.3.0.0/29"
}

# Monitoring
variable "notification_channels" {
  description = "Notification channels for alerts"
  type        = list(string)
  default     = []
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
