# Storage Module for VisionFlow

# Media Storage Bucket
resource "google_storage_bucket" "media_bucket" {
  name          = var.media_bucket_name
  location      = var.region
  project       = var.project_id
  force_destroy = var.environment != "prod"

  # Versioning
  versioning {
    enabled = var.environment == "prod"
  }

  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }

  # CORS configuration for web access
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }

  # Uniform bucket-level access
  uniform_bucket_level_access = true

  # Public access prevention
  public_access_prevention = var.environment == "prod" ? "enforced" : "inherited"

  # Logging
  logging {
    log_bucket        = google_storage_bucket.logs_bucket.name
    log_object_prefix = "media-access-logs/"
  }

  labels = merge(var.labels, {
    purpose = "media-storage"
    tier    = "hot"
  })
}

# MLFlow Artifacts Bucket
resource "google_storage_bucket" "mlflow_bucket" {
  name          = var.mlflow_bucket_name
  location      = var.region
  project       = var.project_id
  force_destroy = var.environment != "prod"

  # Versioning for model artifacts
  versioning {
    enabled = true
  }

  # Lifecycle management for MLFlow artifacts
  lifecycle_rule {
    condition {
      age = 60
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 180
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Keep model artifacts for longer
  lifecycle_rule {
    condition {
      age                = 730  # 2 years
      matches_storage_class = ["COLDLINE"]
    }
    action {
      type = "Delete"
    }
  }

  # Uniform bucket-level access
  uniform_bucket_level_access = true

  # Public access prevention
  public_access_prevention = "enforced"

  # Logging
  logging {
    log_bucket        = google_storage_bucket.logs_bucket.name
    log_object_prefix = "mlflow-access-logs/"
  }

  labels = merge(var.labels, {
    purpose = "mlflow-artifacts"
    tier    = "archive"
  })
}

# Terraform State Bucket
resource "google_storage_bucket" "terraform_state_bucket" {
  name          = var.terraform_bucket_name
  location      = var.region
  project       = var.project_id
  force_destroy = false  # Never auto-delete terraform state

  # Versioning is critical for terraform state
  versioning {
    enabled = true
  }

  # Prevent deletion of terraform state
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Uniform bucket-level access
  uniform_bucket_level_access = true

  # Public access prevention
  public_access_prevention = "enforced"

  # Logging
  logging {
    log_bucket        = google_storage_bucket.logs_bucket.name
    log_object_prefix = "terraform-access-logs/"
  }

  labels = merge(var.labels, {
    purpose = "terraform-state"
    tier    = "critical"
  })
}

# Logs Bucket (for access logs)
resource "google_storage_bucket" "logs_bucket" {
  name          = "${var.media_bucket_name}-logs"
  location      = var.region
  project       = var.project_id
  force_destroy = var.environment != "prod"

  # Lifecycle management for logs
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  # Uniform bucket-level access
  uniform_bucket_level_access = true

  # Public access prevention
  public_access_prevention = "enforced"

  labels = merge(var.labels, {
    purpose = "access-logs"
    tier    = "logs"
  })
}

# Backup Bucket (for critical data backup)
resource "google_storage_bucket" "backup_bucket" {
  name          = "${var.media_bucket_name}-backup"
  location      = var.backup_region != null ? var.backup_region : var.region
  project       = var.project_id
  force_destroy = false

  # Versioning for backups
  versioning {
    enabled = true
  }

  # Long-term retention for backups
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "ARCHIVE"
    }
  }

  # Keep backups for 7 years
  lifecycle_rule {
    condition {
      age = 2555  # 7 years
    }
    action {
      type = "Delete"
    }
  }

  # Uniform bucket-level access
  uniform_bucket_level_access = true

  # Public access prevention
  public_access_prevention = "enforced"

  labels = merge(var.labels, {
    purpose = "backup"
    tier    = "archive"
  })
}

# IAM bindings for buckets
resource "google_storage_bucket_iam_member" "media_bucket_viewers" {
  bucket = google_storage_bucket.media_bucket.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"

  condition {
    title       = "Public read access"
    description = "Allow public read access to media files"
    expression  = "request.auth.access_levels != null"
  }
}

resource "google_storage_bucket_iam_member" "mlflow_bucket_admin" {
  bucket = google_storage_bucket.mlflow_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.mlflow_service_account}"
}

resource "google_storage_bucket_iam_member" "terraform_state_admin" {
  bucket = google_storage_bucket.terraform_state_bucket.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${var.terraform_service_account}"
}

# Notification for bucket events
resource "google_pubsub_topic" "bucket_notifications" {
  name    = "visionflow-bucket-notifications"
  project = var.project_id

  labels = var.labels
}

resource "google_storage_notification" "media_bucket_notification" {
  bucket         = google_storage_bucket.media_bucket.name
  payload_format = "JSON_API_V1"
  topic          = google_pubsub_topic.bucket_notifications.id
  event_types    = ["OBJECT_FINALIZE", "OBJECT_DELETE"]

  depends_on = [google_pubsub_topic_iam_member.bucket_notification_publisher]
}

resource "google_pubsub_topic_iam_member" "bucket_notification_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.bucket_notifications.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com"
}

data "google_project" "project" {
  project_id = var.project_id
}
