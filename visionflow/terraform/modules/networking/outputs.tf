# Networking Module Outputs

output "api_ip_address" {
  description = "Static IP address for API"
  value       = google_compute_global_address.api_ip.address
}

output "mlflow_ip_address" {
  description = "Static IP address for MLFlow"
  value       = google_compute_global_address.mlflow_ip.address
}

output "internal_lb_ip_address" {
  description = "Internal load balancer IP address"
  value       = google_compute_address.internal_lb_ip.address
}

output "load_balancer_ip" {
  description = "Main load balancer IP address (API)"
  value       = google_compute_global_address.api_ip.address
}

output "api_ssl_certificate_id" {
  description = "API SSL certificate ID"
  value       = google_compute_managed_ssl_certificate.api_ssl_cert.id
}

output "mlflow_ssl_certificate_id" {
  description = "MLFlow SSL certificate ID"
  value       = google_compute_managed_ssl_certificate.mlflow_ssl_cert.id
}

output "api_backend_service_id" {
  description = "API backend service ID"
  value       = google_compute_backend_service.api_backend.id
}

output "mlflow_backend_service_id" {
  description = "MLFlow backend service ID"
  value       = google_compute_backend_service.mlflow_backend.id
}

output "api_url_map_id" {
  description = "API URL map ID"
  value       = google_compute_url_map.api_url_map.id
}

output "mlflow_url_map_id" {
  description = "MLFlow URL map ID"
  value       = google_compute_url_map.mlflow_url_map.id
}

output "security_policy_id" {
  description = "Cloud Armor security policy ID"
  value       = google_compute_security_policy.api_security_policy.id
}

output "health_check_ids" {
  description = "Health check IDs"
  value = {
    api    = google_compute_health_check.api_health_check.id
    mlflow = google_compute_health_check.mlflow_health_check.id
  }
}

output "forwarding_rule_ids" {
  description = "Forwarding rule IDs"
  value = {
    api_https    = google_compute_global_forwarding_rule.api_https_forwarding_rule.id
    api_http     = google_compute_global_forwarding_rule.api_http_forwarding_rule.id
    mlflow_https = google_compute_global_forwarding_rule.mlflow_https_forwarding_rule.id
  }
}
