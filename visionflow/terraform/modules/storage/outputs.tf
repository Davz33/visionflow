# Storage Module Outputs

output "media_bucket_name" {
  description = "Name of the media storage bucket"
  value       = google_storage_bucket.media_bucket.name
}

output "media_bucket_url" {
  description = "URL of the media storage bucket"
  value       = google_storage_bucket.media_bucket.url
}

output "media_bucket_self_link" {
  description = "Self link of the media storage bucket"
  value       = google_storage_bucket.media_bucket.self_link
}

output "mlflow_bucket_name" {
  description = "Name of the MLflow artifacts bucket"
  value       = google_storage_bucket.mlflow_bucket.name
}

output "mlflow_bucket_url" {
  description = "URL of the MLflow artifacts bucket"
  value       = google_storage_bucket.mlflow_bucket.url
}

output "mlflow_bucket_self_link" {
  description = "Self link of the MLflow artifacts bucket"
  value       = google_storage_bucket.mlflow_bucket.self_link
}

output "terraform_bucket_name" {
  description = "Name of the Terraform state bucket"
  value       = google_storage_bucket.terraform_state_bucket.name
}

output "terraform_bucket_url" {
  description = "URL of the Terraform state bucket"
  value       = google_storage_bucket.terraform_state_bucket.url
}

output "logs_bucket_name" {
  description = "Name of the logs bucket"
  value       = google_storage_bucket.logs_bucket.name
}

output "backup_bucket_name" {
  description = "Name of the backup bucket"
  value       = google_storage_bucket.backup_bucket.name
}

output "notification_topic_name" {
  description = "Name of the bucket notification topic"
  value       = google_pubsub_topic.bucket_notifications.name
}

output "bucket_urls" {
  description = "All bucket URLs"
  value = {
    media     = google_storage_bucket.media_bucket.url
    mlflow    = google_storage_bucket.mlflow_bucket.url
    terraform = google_storage_bucket.terraform_state_bucket.url
    logs      = google_storage_bucket.logs_bucket.url
    backup    = google_storage_bucket.backup_bucket.url
  }
}
