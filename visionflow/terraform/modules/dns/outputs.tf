# DNS Module Outputs

output "dns_zone_name" {
  description = "Name of the DNS zone"
  value       = google_dns_managed_zone.main.name
}

output "dns_zone_nameservers" {
  description = "Nameservers for the DNS zone"
  value       = google_dns_managed_zone.main.name_servers
}

output "dns_zone_id" {
  description = "ID of the DNS zone"
  value       = google_dns_managed_zone.main.id
}

output "api_domain" {
  description = "API domain name"
  value       = "api.${var.domain}"
}

output "mlflow_domain" {
  description = "MLflow domain name"
  value       = "mlflow.${var.domain}"
}

output "www_domain" {
  description = "WWW domain name"
  value       = "www.${var.domain}"
}

output "environment_api_domain" {
  description = "Environment-specific API domain"
  value       = var.environment != "prod" ? "${var.environment}-api.${var.domain}" : "api.${var.domain}"
}

output "environment_mlflow_domain" {
  description = "Environment-specific MLFlow domain"
  value       = var.environment != "prod" ? "${var.environment}-mlflow.${var.domain}" : "mlflow.${var.domain}"
}

output "dns_policy_id" {
  description = "DNS policy ID"
  value       = google_dns_policy.default.id
}

output "private_zone_name" {
  description = "Private DNS zone name"
  value       = var.enable_private_zone ? google_dns_peering_zone.private[0].name : null
}

output "all_domains" {
  description = "All configured domains"
  value = {
    api              = "api.${var.domain}"
    mlflow           = "mlflow.${var.domain}"
    www              = "www.${var.domain}"
    environment_api  = var.environment != "prod" ? "${var.environment}-api.${var.domain}" : "api.${var.domain}"
    environment_mlflow = var.environment != "prod" ? "${var.environment}-mlflow.${var.domain}" : "mlflow.${var.domain}"
  }
}
