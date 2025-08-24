# VisionFlow Infrastructure - Main Terraform Configuration
# Provisions complete GCP infrastructure for VisionFlow platform

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Configure remote state backend
  backend "gcs" {
    bucket = "visionflow-terraform-state"
    prefix = "terraform/state"
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Data sources
data "google_client_config" "default" {}

data "google_project" "project" {
  project_id = var.project_id
}

# Configure Kubernetes provider
provider "kubernetes" {
  host                   = "https://${module.gke.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.ca_certificate)
}

# Configure Helm provider
provider "helm" {
  kubernetes {
    host                   = "https://${module.gke.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(module.gke.ca_certificate)
  }
}

# Local values
locals {
  common_labels = {
    project     = "visionflow"
    environment = var.environment
    managed_by  = "terraform"
  }
  
  dns_zone_name = replace(var.domain, ".", "-")
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "storage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "dns.googleapis.com",
    "servicenetworking.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "redis.googleapis.com"
  ])

  service = each.value
  project = var.project_id

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Network Infrastructure
module "vpc" {
  source = "./modules/vpc"
  
  project_id   = var.project_id
  region       = var.region
  environment  = var.environment
  
  vpc_name                = var.vpc_name
  subnet_name            = var.subnet_name
  subnet_cidr            = var.subnet_cidr
  secondary_ranges       = var.secondary_ranges
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# IAM and Security
module "iam" {
  source = "./modules/iam"
  
  project_id  = var.project_id
  environment = var.environment
  
  gke_cluster_name = var.cluster_name
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# Cloud Storage
module "storage" {
  source = "./modules/storage"
  
  project_id  = var.project_id
  region      = var.region
  environment = var.environment
  
  media_bucket_name     = var.media_bucket_name
  mlflow_bucket_name    = var.mlflow_bucket_name
  terraform_bucket_name = var.terraform_bucket_name
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# Database Infrastructure
module "database" {
  source = "./modules/database"
  
  project_id  = var.project_id
  region      = var.region
  environment = var.environment
  
  vpc_network_id = module.vpc.vpc_network_id
  
  postgres_instance_name = var.postgres_instance_name
  postgres_version      = var.postgres_version
  postgres_tier         = var.postgres_tier
  
  redis_instance_name = var.redis_instance_name
  redis_memory_size   = var.redis_memory_size
  redis_version       = var.redis_version
  
  labels = local.common_labels
  
  depends_on = [
    google_project_service.apis,
    module.vpc
  ]
}

# GKE Cluster
module "gke" {
  source = "./modules/gke"
  
  project_id  = var.project_id
  region      = var.region
  zone        = var.zone
  environment = var.environment
  
  cluster_name          = var.cluster_name
  network              = module.vpc.vpc_network_name
  subnetwork           = module.vpc.subnet_name
  pods_range_name      = "pods"
  services_range_name  = "services"
  
  node_pools = var.node_pools
  
  service_account = module.iam.gke_service_account_email
  
  labels = local.common_labels
  
  depends_on = [
    google_project_service.apis,
    module.vpc,
    module.iam
  ]
}

# Load Balancer and External IP
module "networking" {
  source = "./modules/networking"
  
  project_id  = var.project_id
  region      = var.region
  environment = var.environment
  
  domain = var.domain
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# DNS Configuration
module "dns" {
  source = "./modules/dns"
  
  project_id  = var.project_id
  environment = var.environment
  
  domain           = var.domain
  dns_zone_name    = local.dns_zone_name
  api_ip_address   = module.networking.api_ip_address
  mlflow_ip_address = module.networking.mlflow_ip_address
  
  labels = local.common_labels
  
  depends_on = [
    google_project_service.apis,
    module.networking
  ]
}

# Secret Manager
module "secrets" {
  source = "./modules/secrets"
  
  project_id  = var.project_id
  environment = var.environment
  
  secrets = var.secrets
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# Monitoring Infrastructure
module "monitoring" {
  source = "./modules/monitoring"
  
  project_id  = var.project_id
  environment = var.environment
  
  notification_channels = var.notification_channels
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}
