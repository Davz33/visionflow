# GKE Module Variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
}

variable "zone" {
  description = "The GCP zone"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
}

variable "network" {
  description = "VPC network name"
  type        = string
}

variable "subnetwork" {
  description = "Subnet name"
  type        = string
}

variable "pods_range_name" {
  description = "Name of the secondary range for pods"
  type        = string
}

variable "services_range_name" {
  description = "Name of the secondary range for services"
  type        = string
}

variable "service_account" {
  description = "Service account email for GKE nodes"
  type        = string
}

variable "node_pools" {
  description = "GKE node pool configurations"
  type = map(object({
    machine_type     = string
    min_count       = number
    max_count       = number
    disk_size_gb    = number
    disk_type       = string
    image_type      = string
    auto_repair     = bool
    auto_upgrade    = bool
    preemptible     = bool
    accelerator_type = optional(string)
    accelerator_count = optional(number)
    taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })))
    labels = optional(map(string))
  }))
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
