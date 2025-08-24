# DNS Module Variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "domain" {
  description = "Primary domain name"
  type        = string
}

variable "dns_zone_name" {
  description = "Name of the DNS zone"
  type        = string
}

variable "api_ip_address" {
  description = "IP address for API"
  type        = string
}

variable "mlflow_ip_address" {
  description = "IP address for MLFlow"
  type        = string
}

variable "vpc_network_url" {
  description = "VPC network URL for DNS policy"
  type        = string
  default     = ""
}

variable "mx_records" {
  description = "MX records for email"
  type        = list(string)
  default     = []
}

variable "txt_records" {
  description = "TXT records (key = subdomain, value = record value)"
  type        = map(string)
  default = {
    "@" = "v=spf1 include:_spf.google.com ~all"
  }
}

variable "caa_records" {
  description = "CAA records for certificate authority authorization"
  type        = list(string)
  default = [
    "0 issue \"letsencrypt.org\"",
    "0 issuewild \"letsencrypt.org\"",
    "0 iodef \"mailto:security@example.com\""
  ]
}

variable "enable_private_zone" {
  description = "Enable private DNS zone"
  type        = bool
  default     = false
}

variable "internal_services" {
  description = "Internal service DNS records"
  type        = map(string)
  default = {
    postgres = "10.0.1.10"
    redis    = "10.0.1.11"
    mlflow   = "10.0.1.12"
  }
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
