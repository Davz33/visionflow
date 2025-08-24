# Monitoring Module Variables

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "notification_channels" {
  description = "Notification channels for monitoring alerts"
  type = list(object({
    display_name = string
    type         = string
    labels       = map(string)
    user_labels  = optional(map(string))
  }))
  default = []
}

variable "api_domain" {
  description = "API domain for uptime checks"
  type        = string
  default     = "api.visionflow.ai"
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed monitoring with additional metrics"
  type        = bool
  default     = true
}

variable "alert_thresholds" {
  description = "Threshold values for alerts"
  type = object({
    error_rate           = optional(number, 5)
    latency_seconds      = optional(number, 30)
    cpu_utilization      = optional(number, 80)
    memory_utilization   = optional(number, 85)
    disk_utilization     = optional(number, 85)
    pod_restart_count    = optional(number, 3)
  })
  default = {}
}

variable "slo_targets" {
  description = "Service Level Objective targets"
  type = object({
    availability         = optional(number, 0.99)
    latency_percentile   = optional(number, 0.95)
    latency_threshold_ms = optional(number, 1000)
  })
  default = {}
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}
