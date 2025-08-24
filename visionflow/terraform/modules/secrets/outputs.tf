# Secrets Module Outputs

output "secret_names" {
  description = "Names of created secrets"
  value       = [for secret in google_secret_manager_secret.secrets : secret.secret_id]
}

output "secret_ids" {
  description = "IDs of created secrets"
  value = {
    for k, secret in google_secret_manager_secret.secrets : k => secret.id
  }
}

output "db_credentials_secret_id" {
  description = "Database credentials secret ID"
  value       = google_secret_manager_secret.db_credentials.id
}

output "redis_credentials_secret_id" {
  description = "Redis credentials secret ID"
  value       = google_secret_manager_secret.redis_credentials.id
}

output "mlflow_config_secret_id" {
  description = "MLFlow configuration secret ID"
  value       = google_secret_manager_secret.mlflow_config.id
}

output "gcp_service_account_key_secret_id" {
  description = "GCP service account key secret ID"
  value       = var.gcp_service_account_key != null ? google_secret_manager_secret.gcp_service_account_key[0].id : null
}

output "tls_certificate_secret_ids" {
  description = "TLS certificate secret IDs"
  value = {
    for k, secret in google_secret_manager_secret.tls_certificates : k => secret.id
  }
}

output "secret_rotation_topic" {
  description = "Secret rotation Pub/Sub topic"
  value       = google_pubsub_topic.secret_rotation.name
}

output "secret_versions" {
  description = "Secret version names"
  value = {
    for k, version in google_secret_manager_secret_version.secret_versions : k => version.name
  }
  sensitive = true
}
