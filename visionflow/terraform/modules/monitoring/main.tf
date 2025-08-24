# Monitoring Module for VisionFlow

# Enable required APIs
resource "google_project_service" "monitoring_apis" {
  for_each = toset([
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "clouderrorreporting.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Notification channels
resource "google_monitoring_notification_channel" "notification_channels" {
  for_each = { for idx, channel in var.notification_channels : idx => channel }

  display_name = each.value.display_name
  type         = each.value.type
  project      = var.project_id

  labels      = each.value.labels
  user_labels = merge(var.labels, each.value.user_labels != null ? each.value.user_labels : {})

  depends_on = [google_project_service.monitoring_apis]
}

# Custom metrics
resource "google_logging_metric" "error_rate" {
  name    = "visionflow_error_rate_${var.environment}"
  project = var.project_id

  filter = <<-EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="visionflow"
    severity>=ERROR
  EOT

  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "DOUBLE"
    unit        = "1"
    display_name = "VisionFlow Error Rate"
  }

  value_extractor = "EXTRACT(jsonPayload.level)"

  label_extractors = {
    service   = "EXTRACT(jsonPayload.service)"
    pod_name  = "EXTRACT(resource.labels.pod_name)"
  }
}

resource "google_logging_metric" "generation_latency" {
  name    = "visionflow_generation_latency_${var.environment}"
  project = var.project_id

  filter = <<-EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="visionflow"
    jsonPayload.event="video_generation_completed"
  EOT

  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "DOUBLE"
    unit        = "s"
    display_name = "VisionFlow Generation Latency"
  }

  value_extractor = "EXTRACT(jsonPayload.generation_time)"

  label_extractors = {
    model_type = "EXTRACT(jsonPayload.model)"
    user_id    = "EXTRACT(jsonPayload.user_id)"
  }
}

# Alert policies
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "VisionFlow High Error Rate - ${var.environment}"
  project      = var.project_id

  conditions {
    display_name = "High error rate condition"

    condition_threshold {
      filter          = "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"visionflow\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 5

      aggregations {
        alignment_period     = "60s"
        per_series_aligner  = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields     = ["resource.labels.container_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = [for channel in google_monitoring_notification_channel.notification_channels : channel.name]

  enabled = true

  documentation {
    content   = "High error rate detected in VisionFlow ${var.environment} environment"
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "VisionFlow High Generation Latency - ${var.environment}"
  project      = var.project_id

  conditions {
    display_name = "High generation latency condition"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/visionflow_generation_latency_${var.environment}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 30 # 30 seconds

      aggregations {
        alignment_period    = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = [for channel in google_monitoring_notification_channel.notification_channels : channel.name]

  enabled = true

  documentation {
    content   = "High generation latency detected in VisionFlow ${var.environment} environment"
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "pod_crash_loop" {
  display_name = "VisionFlow Pod Crash Loop - ${var.environment}"
  project      = var.project_id

  conditions {
    display_name = "Pod crash loop condition"

    condition_threshold {
      filter          = "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"visionflow\" AND metric.type=\"kubernetes.io/container/restart_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 3

      aggregations {
        alignment_period    = "60s"
        per_series_aligner = "ALIGN_DELTA"
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = [for channel in google_monitoring_notification_channel.notification_channels : channel.name]

  enabled = true

  documentation {
    content   = "Pod crash loop detected in VisionFlow ${var.environment} environment"
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "disk_usage" {
  display_name = "VisionFlow High Disk Usage - ${var.environment}"
  project      = var.project_id

  conditions {
    display_name = "High disk usage condition"

    condition_threshold {
      filter          = "resource.type=\"k8s_node\" AND metric.type=\"kubernetes.io/node/disk/used_bytes\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 85

      aggregations {
        alignment_period    = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = [for channel in google_monitoring_notification_channel.notification_channels : channel.name]

  enabled = true

  documentation {
    content   = "High disk usage detected on VisionFlow nodes in ${var.environment} environment"
    mime_type = "text/markdown"
  }
}

# Monitoring dashboard
resource "google_monitoring_dashboard" "visionflow_dashboard" {
  dashboard_json = jsonencode({
    displayName = "VisionFlow ${title(var.environment)} Dashboard"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"visionflow\""
                    aggregation = {
                      alignmentPeriod    = "60s"
                      perSeriesAligner  = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields     = ["resource.labels.container_name"]
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Requests/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          xPos   = 6
          widget = {
            title = "Error Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/visionflow_error_rate_${var.environment}\""
                    aggregation = {
                      alignmentPeriod   = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Errors/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          yPos   = 4
          widget = {
            title = "Generation Latency"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/visionflow_generation_latency_${var.environment}\""
                    aggregation = {
                      alignmentPeriod   = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Latency (seconds)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          xPos   = 6
          yPos   = 4
          widget = {
            title = "Resource Usage"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"k8s_container\" AND metric.type=\"kubernetes.io/container/cpu/core_usage_time\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner  = "ALIGN_RATE"
                        crossSeriesReducer = "REDUCE_SUM"
                      }
                    }
                  }
                  plotType = "LINE"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"k8s_container\" AND metric.type=\"kubernetes.io/container/memory/used_bytes\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner  = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_SUM"
                      }
                    }
                  }
                  plotType = "LINE"
                }
              ]
              yAxis = {
                label = "Usage"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })

  project = var.project_id

  depends_on = [google_project_service.monitoring_apis]
}

# SLO (Service Level Objective)
resource "google_monitoring_slo" "api_availability" {
  service      = google_monitoring_service.visionflow_service.service_id
  slo_id       = "api-availability-${var.environment}"
  display_name = "API Availability SLO"
  project      = var.project_id

  goal                = 0.99
  rolling_period_days = 30

  request_based_sli {
    good_total_ratio {
      total_service_filter = "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"visionflow\""
      good_service_filter  = "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"visionflow\" AND metric.labels.response_code!~\"5.*\""
    }
  }
}

# Service for SLO
resource "google_monitoring_service" "visionflow_service" {
  service_id   = "visionflow-${var.environment}"
  display_name = "VisionFlow ${title(var.environment)}"
  project      = var.project_id

  user_labels = var.labels
}

# Uptime checks
resource "google_monitoring_uptime_check_config" "api_uptime_check" {
  display_name = "VisionFlow API Uptime Check - ${var.environment}"
  project      = var.project_id
  timeout      = "10s"
  period       = "60s"

  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = var.api_domain
    }
  }

  content_matchers {
    content = "healthy"
    matcher = "CONTAINS_STRING"
  }

  selected_regions = ["USA", "EUROPE", "ASIA_PACIFIC"]
}

# Log-based metrics for business metrics
resource "google_logging_metric" "video_generations" {
  name    = "visionflow_video_generations_${var.environment}"
  project = var.project_id

  filter = <<-EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="visionflow"
    jsonPayload.event="video_generation_started"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
    display_name = "VisionFlow Video Generations"
  }

  value_extractor = "1"

  label_extractors = {
    model_type = "EXTRACT(jsonPayload.model)"
    user_tier  = "EXTRACT(jsonPayload.user_tier)"
  }
}
