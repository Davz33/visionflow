# Database Module for VisionFlow

# Private services access for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = var.vpc_network_id
  project       = var.project_id
}

# Private service connection
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = var.vpc_network_id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# PostgreSQL instance
resource "google_sql_database_instance" "postgres" {
  name             = var.postgres_instance_name
  database_version = var.postgres_version
  region           = var.region
  project          = var.project_id

  settings {
    tier                        = var.postgres_tier
    availability_type          = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_type                  = "PD_SSD"
    disk_size                  = var.postgres_disk_size
    disk_autoresize            = true
    disk_autoresize_limit      = var.postgres_max_disk_size

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.vpc_network_id
      enable_private_path_for_google_cloud_services = true
      require_ssl                                   = true
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }

    database_flags {
      name  = "log_connections"
      value = "on"
    }

    database_flags {
      name  = "log_disconnections"
      value = "on"
    }

    database_flags {
      name  = "log_lock_waits"
      value = "on"
    }

    database_flags {
      name  = "log_temp_files"
      value = "0"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"  # Log statements taking longer than 1 second
    }

    maintenance_window {
      day          = 7    # Sunday
      hour         = 3    # 3 AM
      update_track = "stable"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }

    user_labels = merge(var.labels, {
      database = "postgres"
      service  = "visionflow"
    })
  }

  deletion_protection = var.environment == "prod" ? true : false

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# PostgreSQL databases
resource "google_sql_database" "visionflow_db" {
  name     = "visionflow"
  instance = google_sql_database_instance.postgres.name
  project  = var.project_id
}

resource "google_sql_database" "mlflow_db" {
  name     = "mlflow"
  instance = google_sql_database_instance.postgres.name
  project  = var.project_id
}

# PostgreSQL users
resource "google_sql_user" "visionflow_user" {
  name     = "visionflow"
  instance = google_sql_database_instance.postgres.name
  password = var.postgres_password
  project  = var.project_id
}

resource "google_sql_user" "mlflow_user" {
  name     = "mlflow"
  instance = google_sql_database_instance.postgres.name
  password = var.mlflow_password
  project  = var.project_id
}

# Redis instance
resource "google_redis_instance" "redis" {
  name           = var.redis_instance_name
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size
  region         = var.region
  project        = var.project_id

  location_id             = var.redis_location_id
  alternative_location_id = var.redis_alternative_location_id

  redis_version     = var.redis_version
  display_name      = "VisionFlow Redis ${var.environment}"
  reserved_ip_range = var.redis_reserved_ip_range

  auth_enabled   = true
  transit_encryption_mode = "SERVER_AUTH"

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }

  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "TWENTY_FOUR_HOURS"
    rdb_snapshot_start_time = "03:00"
  }

  authorized_network = var.vpc_network_id

  labels = merge(var.labels, {
    database = "redis"
    service  = "visionflow"
  })
}

# Cloud SQL Proxy for secure connections
resource "google_project_service" "sqladmin" {
  project = var.project_id
  service = "sqladmin.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Database monitoring
resource "google_monitoring_alert_policy" "postgres_cpu" {
  display_name = "PostgreSQL High CPU - ${var.environment}"
  project      = var.project_id

  conditions {
    display_name = "PostgreSQL CPU usage"

    condition_threshold {
      filter         = "resource.type=\"gce_instance\" AND resource.label.instance_id=\"${google_sql_database_instance.postgres.name}\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 80

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channels

  enabled = true
}

resource "google_monitoring_alert_policy" "redis_memory" {
  display_name = "Redis High Memory Usage - ${var.environment}"
  project      = var.project_id

  conditions {
    display_name = "Redis memory usage"

    condition_threshold {
      filter         = "resource.type=\"redis_instance\" AND resource.label.instance_id=\"${google_redis_instance.redis.name}\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 90

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channels

  enabled = true
}
