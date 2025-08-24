# IAM Module Outputs

output "gke_service_account_email" {
  description = "Email of the GKE service account"
  value       = google_service_account.gke_service_account.email
}

output "gke_service_account_id" {
  description = "ID of the GKE service account"
  value       = google_service_account.gke_service_account.id
}

output "workload_identity_service_account" {
  description = "Workload Identity service account email"
  value       = google_service_account.workload_identity_sa.email
}

output "workload_identity_service_account_id" {
  description = "Workload Identity service account ID"
  value       = google_service_account.workload_identity_sa.id
}

output "cloudbuild_service_account_email" {
  description = "Email of the Cloud Build service account"
  value       = google_service_account.cloudbuild_sa.email
}

output "mlflow_service_account_email" {
  description = "Email of the MLFlow service account"
  value       = google_service_account.mlflow_sa.email
}

output "security_admin_service_account_email" {
  description = "Email of the security admin service account"
  value       = google_service_account.security_admin_sa.email
}

output "custom_role_name" {
  description = "Name of the custom VisionFlow role"
  value       = google_project_iam_custom_role.visionflow_role.name
}

output "service_account_key" {
  description = "Service account key (if created)"
  value       = var.create_service_account_keys ? google_service_account_key.workload_identity_key[0].private_key : null
  sensitive   = true
}
