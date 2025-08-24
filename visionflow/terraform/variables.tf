# VisionFlow Infrastructure Variables

# Project Configuration
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "domain" {
  description = "Primary domain for the application"
  type        = string
  default     = "visionflow.ai"
}

# Network Configuration
variable "vpc_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "visionflow-vpc"
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
  default     = "visionflow-subnet"
}

variable "subnet_cidr" {
  description = "CIDR block for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "secondary_ranges" {
  description = "Secondary IP ranges for pods and services"
  type = map(object({
    range_name    = string
    ip_cidr_range = string
  }))
  default = {
    pods = {
      range_name    = "pods"
      ip_cidr_range = "10.1.0.0/16"
    }
    services = {
      range_name    = "services"
      ip_cidr_range = "10.2.0.0/16"
    }
  }
}

# GKE Configuration
variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "visionflow-cluster"
}

variable "node_pools" {
  description = "GKE node pool configurations"
  type = map(object({
    machine_type     = string
    min_count       = number
    max_count       = number
    disk_size_gb    = number
    disk_type       = string
    image_type      = string
    auto_repair     = bool
    auto_upgrade    = bool
    preemptible     = bool
    accelerator_type = optional(string)
    accelerator_count = optional(number)
    taints = optional(list(object({
      key    = string
      value  = string
      effect = string
    })))
    labels = optional(map(string))
  }))
  
  default = {
    default = {
      machine_type     = "e2-standard-4"
      min_count       = 1
      max_count       = 5
      disk_size_gb    = 50
      disk_type       = "pd-standard"
      image_type      = "COS_CONTAINERD"
      auto_repair     = true
      auto_upgrade    = true
      preemptible     = false
    }
    
    cpu-intensive = {
      machine_type     = "c2-standard-8"
      min_count       = 0
      max_count       = 3
      disk_size_gb    = 100
      disk_type       = "pd-ssd"
      image_type      = "COS_CONTAINERD"
      auto_repair     = true
      auto_upgrade    = true
      preemptible     = false
      labels = {
        workload = "cpu-intensive"
      }
    }
    
    gpu = {
      machine_type      = "n1-standard-4"
      min_count        = 0
      max_count        = 3
      disk_size_gb     = 100
      disk_type        = "pd-ssd"
      image_type       = "COS_CONTAINERD"
      auto_repair      = true
      auto_upgrade     = true
      preemptible      = true
      accelerator_type = "nvidia-tesla-t4"
      accelerator_count = 1
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      labels = {
        workload = "gpu"
      }
    }
  }
}

# Storage Configuration
variable "media_bucket_name" {
  description = "Name of the media storage bucket"
  type        = string
  default     = "visionflow-media-bucket"
}

variable "mlflow_bucket_name" {
  description = "Name of the MLflow artifacts bucket"
  type        = string
  default     = "visionflow-mlflow-artifacts"
}

variable "terraform_bucket_name" {
  description = "Name of the Terraform state bucket"
  type        = string
  default     = "visionflow-terraform-state"
}

# Database Configuration
variable "postgres_instance_name" {
  description = "Name of the PostgreSQL instance"
  type        = string
  default     = "visionflow-postgres"
}

variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "POSTGRES_15"
}

variable "postgres_tier" {
  description = "PostgreSQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "redis_instance_name" {
  description = "Name of the Redis instance"
  type        = string
  default     = "visionflow-redis"
}

variable "redis_memory_size" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_7_0"
}

# Secrets Configuration
variable "secrets" {
  description = "Map of secrets to create in Secret Manager"
  type = map(object({
    secret_data = string
    labels      = optional(map(string))
  }))
  sensitive = true
  default = {
    langchain-api-key = {
      secret_data = "placeholder-langchain-key"
      labels = {
        service = "orchestration"
      }
    }
    openai-api-key = {
      secret_data = "placeholder-openai-key"
      labels = {
        service = "ai"
      }
    }
    huggingface-token = {
      secret_data = "placeholder-hf-token"
      labels = {
        service = "ai"
      }
    }
    postgres-password = {
      secret_data = "visionflow-postgres-password"
      labels = {
        service = "database"
      }
    }
    redis-password = {
      secret_data = "visionflow-redis-password"
      labels = {
        service = "cache"
      }
    }
    mlflow-auth = {
      secret_data = "mlflow:mlflow123"
      labels = {
        service = "mlflow"
      }
    }
  }
}

# Monitoring Configuration
variable "notification_channels" {
  description = "Notification channels for monitoring alerts"
  type = list(object({
    display_name = string
    type         = string
    labels       = map(string)
    user_labels  = optional(map(string))
  }))
  default = [
    {
      display_name = "VisionFlow Email Alerts"
      type         = "email"
      labels = {
        email_address = "alerts@visionflow.ai"
      }
      user_labels = {
        service = "visionflow"
      }
    }
  ]
}

# Resource Labels
variable "default_labels" {
  description = "Default labels to apply to all resources"
  type        = map(string)
  default = {
    project    = "visionflow"
    managed_by = "terraform"
  }
}
