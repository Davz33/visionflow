# Secrets Module for VisionFlow

# Create secrets in Google Secret Manager
resource "google_secret_manager_secret" "secrets" {
  for_each = var.secrets

  secret_id = each.key
  project   = var.project_id

  labels = merge(var.labels, each.value.labels != null ? each.value.labels : {})

  replication {
    auto {
      # Use customer-managed encryption keys for production
      dynamic "customer_managed_encryption" {
        for_each = var.environment == "prod" && var.kms_key_id != null ? [1] : []
        content {
          kms_key_name = var.kms_key_id
        }
      }
    }
  }

  depends_on = [google_project_service.secretmanager]
}

# Create secret versions
resource "google_secret_manager_secret_version" "secret_versions" {
  for_each = var.secrets

  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = each.value.secret_data

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Enable Secret Manager API
resource "google_project_service" "secretmanager" {
  project = var.project_id
  service = "secretmanager.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

# IAM bindings for secret access
resource "google_secret_manager_secret_iam_member" "secret_accessors" {
  for_each = var.secret_accessors

  project   = var.project_id
  secret_id = google_secret_manager_secret.secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value

  depends_on = [google_secret_manager_secret.secrets]
}

# Additional database credentials
resource "google_secret_manager_secret" "db_credentials" {
  secret_id = "visionflow-db-credentials-${var.environment}"
  project   = var.project_id

  labels = merge(var.labels, {
    service = "database"
    type    = "credentials"
  })

  replication {
    auto {
      dynamic "customer_managed_encryption" {
        for_each = var.environment == "prod" && var.kms_key_id != null ? [1] : []
        content {
          kms_key_name = var.kms_key_id
        }
      }
    }
  }
}

resource "google_secret_manager_secret_version" "db_credentials" {
  secret = google_secret_manager_secret.db_credentials.id
  secret_data = jsonencode({
    username = "visionflow"
    password = var.db_password
    host     = var.db_host
    port     = "5432"
    database = "visionflow"
  })

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Redis credentials
resource "google_secret_manager_secret" "redis_credentials" {
  secret_id = "visionflow-redis-credentials-${var.environment}"
  project   = var.project_id

  labels = merge(var.labels, {
    service = "redis"
    type    = "credentials"
  })

  replication {
    auto {
      dynamic "customer_managed_encryption" {
        for_each = var.environment == "prod" && var.kms_key_id != null ? [1] : []
        content {
          kms_key_name = var.kms_key_id
        }
      }
    }
  }
}

resource "google_secret_manager_secret_version" "redis_credentials" {
  secret = google_secret_manager_secret.redis_credentials.id
  secret_data = jsonencode({
    host     = var.redis_host
    port     = var.redis_port
    password = var.redis_password
    db       = "0"
  })

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# MLFlow configuration
resource "google_secret_manager_secret" "mlflow_config" {
  secret_id = "visionflow-mlflow-config-${var.environment}"
  project   = var.project_id

  labels = merge(var.labels, {
    service = "mlflow"
    type    = "configuration"
  })

  replication {
    auto {
      dynamic "customer_managed_encryption" {
        for_each = var.environment == "prod" && var.kms_key_id != null ? [1] : []
        content {
          kms_key_name = var.kms_key_id
        }
      }
    }
  }
}

resource "google_secret_manager_secret_version" "mlflow_config" {
  secret = google_secret_manager_secret.mlflow_config.id
  secret_data = jsonencode({
    tracking_uri       = "http://mlflow-service:5000"
    backend_store_uri  = "postgresql://mlflow:${var.mlflow_password}@${var.db_host}:5432/mlflow"
    artifact_root      = "gs://${var.mlflow_bucket_name}"
    registry_uri       = "http://mlflow-service:5000"
  })

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Google Cloud service account key (if needed)
resource "google_secret_manager_secret" "gcp_service_account_key" {
  count     = var.gcp_service_account_key != null ? 1 : 0
  secret_id = "visionflow-gcp-sa-key-${var.environment}"
  project   = var.project_id

  labels = merge(var.labels, {
    service = "gcp"
    type    = "service-account-key"
  })

  replication {
    auto {
      dynamic "customer_managed_encryption" {
        for_each = var.environment == "prod" && var.kms_key_id != null ? [1] : []
        content {
          kms_key_name = var.kms_key_id
        }
      }
    }
  }
}

resource "google_secret_manager_secret_version" "gcp_service_account_key" {
  count       = var.gcp_service_account_key != null ? 1 : 0
  secret      = google_secret_manager_secret.gcp_service_account_key[0].id
  secret_data = var.gcp_service_account_key

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# TLS certificates for internal services
resource "google_secret_manager_secret" "tls_certificates" {
  for_each = var.tls_certificates

  secret_id = "visionflow-tls-${each.key}-${var.environment}"
  project   = var.project_id

  labels = merge(var.labels, {
    service = each.key
    type    = "tls-certificate"
  })

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "tls_certificates" {
  for_each = var.tls_certificates

  secret      = google_secret_manager_secret.tls_certificates[each.key].id
  secret_data = each.value

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Secret rotation notification
resource "google_pubsub_topic" "secret_rotation" {
  name    = "visionflow-secret-rotation-${var.environment}"
  project = var.project_id

  labels = var.labels

  message_retention_duration = "604800s" # 7 days
}

# Cloud Function for secret rotation (placeholder)
resource "google_cloudfunctions_function" "secret_rotation" {
  count               = var.enable_secret_rotation ? 1 : 0
  name                = "visionflow-secret-rotation-${var.environment}"
  project             = var.project_id
  region              = var.region
  runtime             = "python39"
  available_memory_mb = 256
  timeout             = 60

  source_archive_bucket = var.function_source_bucket
  source_archive_object = "secret-rotation.zip"
  entry_point          = "rotate_secrets"

  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.secret_rotation.name
  }

  environment_variables = {
    PROJECT_ID   = var.project_id
    ENVIRONMENT  = var.environment
  }

  labels = var.labels
}
