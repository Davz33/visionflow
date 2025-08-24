# Monitoring Module Outputs

output "notification_channel_names" {
  description = "Names of notification channels"
  value       = [for channel in google_monitoring_notification_channel.notification_channels : channel.name]
}

output "notification_channel_ids" {
  description = "IDs of notification channels"
  value       = [for channel in google_monitoring_notification_channel.notification_channels : channel.id]
}

output "alert_policy_names" {
  description = "Names of alert policies"
  value = [
    google_monitoring_alert_policy.high_error_rate.name,
    google_monitoring_alert_policy.high_latency.name,
    google_monitoring_alert_policy.pod_crash_loop.name,
    google_monitoring_alert_policy.disk_usage.name
  ]
}

output "dashboard_url" {
  description = "URL of the monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.visionflow_dashboard.id}?project=${var.project_id}"
}

output "slo_name" {
  description = "Name of the SLO"
  value       = google_monitoring_slo.api_availability.name
}

output "service_name" {
  description = "Name of the monitoring service"
  value       = google_monitoring_service.visionflow_service.name
}

output "uptime_check_name" {
  description = "Name of the uptime check"
  value       = google_monitoring_uptime_check_config.api_uptime_check.name
}

output "custom_metrics" {
  description = "Custom log-based metrics"
  value = {
    error_rate          = google_logging_metric.error_rate.name
    generation_latency  = google_logging_metric.generation_latency.name
    video_generations   = google_logging_metric.video_generations.name
  }
}

output "monitoring_links" {
  description = "Useful monitoring links"
  value = {
    dashboard      = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.visionflow_dashboard.id}?project=${var.project_id}"
    alerts         = "https://console.cloud.google.com/monitoring/alerting?project=${var.project_id}"
    slo            = "https://console.cloud.google.com/monitoring/slo?project=${var.project_id}"
    uptime_checks  = "https://console.cloud.google.com/monitoring/uptime?project=${var.project_id}"
    logs           = "https://console.cloud.google.com/logs/query?project=${var.project_id}"
  }
}
