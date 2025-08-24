# VisionFlow Infrastructure Outputs

# Network Outputs
output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = module.vpc.vpc_network_name
}

output "vpc_network_id" {
  description = "ID of the VPC network"
  value       = module.vpc.vpc_network_id
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = module.vpc.subnet_name
}

output "subnet_cidr" {
  description = "CIDR of the subnet"
  value       = module.vpc.subnet_cidr
}

# GKE Outputs
output "gke_cluster_name" {
  description = "Name of the GKE cluster"
  value       = module.gke.cluster_name
}

output "gke_cluster_endpoint" {
  description = "Endpoint of the GKE cluster"
  value       = module.gke.endpoint
  sensitive   = true
}

output "gke_cluster_ca_certificate" {
  description = "CA certificate of the GKE cluster"
  value       = module.gke.ca_certificate
  sensitive   = true
}

output "gke_cluster_location" {
  description = "Location of the GKE cluster"
  value       = module.gke.location
}

# Kubectl Configuration Command
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${module.gke.cluster_name} --zone ${var.zone} --project ${var.project_id}"
}

# Database Outputs
output "postgres_connection_name" {
  description = "PostgreSQL connection name"
  value       = module.database.postgres_connection_name
}

output "postgres_private_ip" {
  description = "PostgreSQL private IP address"
  value       = module.database.postgres_private_ip
}

output "redis_host" {
  description = "Redis host address"
  value       = module.database.redis_host
}

output "redis_port" {
  description = "Redis port"
  value       = module.database.redis_port
}

# Storage Outputs
output "media_bucket_name" {
  description = "Name of the media storage bucket"
  value       = module.storage.media_bucket_name
}

output "media_bucket_url" {
  description = "URL of the media storage bucket"
  value       = module.storage.media_bucket_url
}

output "mlflow_bucket_name" {
  description = "Name of the MLflow artifacts bucket"
  value       = module.storage.mlflow_bucket_name
}

output "mlflow_bucket_url" {
  description = "URL of the MLflow artifacts bucket"
  value       = module.storage.mlflow_bucket_url
}

# IAM Outputs
output "gke_service_account_email" {
  description = "Email of the GKE service account"
  value       = module.iam.gke_service_account_email
}

output "workload_identity_service_account" {
  description = "Workload Identity service account"
  value       = module.iam.workload_identity_service_account
}

# Network Outputs
output "api_ip_address" {
  description = "Static IP address for API"
  value       = module.networking.api_ip_address
}

output "mlflow_ip_address" {
  description = "Static IP address for MLflow"
  value       = module.networking.mlflow_ip_address
}

output "load_balancer_ip" {
  description = "Load balancer IP address"
  value       = module.networking.load_balancer_ip
}

# DNS Outputs
output "dns_zone_name" {
  description = "Name of the DNS zone"
  value       = module.dns.dns_zone_name
}

output "dns_zone_nameservers" {
  description = "Nameservers for the DNS zone"
  value       = module.dns.dns_zone_nameservers
}

output "api_domain" {
  description = "API domain name"
  value       = module.dns.api_domain
}

output "mlflow_domain" {
  description = "MLflow domain name"
  value       = module.dns.mlflow_domain
}

# Secret Manager Outputs
output "secret_names" {
  description = "Names of created secrets"
  value       = module.secrets.secret_names
}

# Monitoring Outputs
output "notification_channel_names" {
  description = "Names of notification channels"
  value       = module.monitoring.notification_channel_names
}

# Environment Configuration
output "environment_config" {
  description = "Environment configuration for applications"
  value = {
    # Database
    DB_HOST     = module.database.postgres_private_ip
    DB_PORT     = "5432"
    DB_NAME     = "visionflow"
    
    # Redis
    REDIS_HOST = module.database.redis_host
    REDIS_PORT = module.database.redis_port
    REDIS_DB   = "0"
    
    # Storage
    STORAGE_BUCKET = module.storage.media_bucket_name
    MLFLOW_BUCKET  = module.storage.mlflow_bucket_name
    
    # MLFlow
    MLFLOW_TRACKING_URI = "http://mlflow-service:5000"
    MLFLOW_DEFAULT_ARTIFACT_ROOT = "gs://${module.storage.mlflow_bucket_name}"
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT = var.project_id
    GOOGLE_CLOUD_REGION  = var.region
    
    # Vertex AI
    VERTEX_AI_PROJECT = var.project_id
    VERTEX_AI_REGION  = var.region
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = "8000"
    
    # Environment
    ENVIRONMENT = var.environment
  }
  sensitive = false
}

# Deployment Information
output "deployment_info" {
  description = "Information for deployment"
  value = {
    project_id     = var.project_id
    cluster_name   = module.gke.cluster_name
    cluster_zone   = var.zone
    api_domain     = module.dns.api_domain
    mlflow_domain  = module.dns.mlflow_domain
    environment    = var.environment
  }
}
