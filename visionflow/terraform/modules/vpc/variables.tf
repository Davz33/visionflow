# VPC Module Variables

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

variable "vpc_name" {
  description = "Name of the VPC network"
  type        = string
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
}

variable "subnet_cidr" {
  description = "CIDR block for the subnet"
  type        = string
}

variable "secondary_ranges" {
  description = "Secondary IP ranges for pods and services"
  type = map(object({
    range_name    = string
    ip_cidr_range = string
  }))
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
