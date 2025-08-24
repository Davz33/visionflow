# Networking Module for VisionFlow

# Global static IP for API load balancer
resource "google_compute_global_address" "api_ip" {
  name         = "visionflow-api-ip-${var.environment}"
  project      = var.project_id
  address_type = "EXTERNAL"
  ip_version   = "IPV4"

  labels = merge(var.labels, {
    purpose = "api-load-balancer"
  })
}

# Global static IP for MLFlow
resource "google_compute_global_address" "mlflow_ip" {
  name         = "visionflow-mlflow-ip-${var.environment}"
  project      = var.project_id
  address_type = "EXTERNAL"
  ip_version   = "IPV4"

  labels = merge(var.labels, {
    purpose = "mlflow-load-balancer"
  })
}

# Regional static IP for internal load balancer
resource "google_compute_address" "internal_lb_ip" {
  name         = "visionflow-internal-lb-ip-${var.environment}"
  project      = var.project_id
  region       = var.region
  address_type = "INTERNAL"
  purpose      = "GCE_ENDPOINT"

  labels = merge(var.labels, {
    purpose = "internal-load-balancer"
  })
}

# SSL Certificate for API
resource "google_compute_managed_ssl_certificate" "api_ssl_cert" {
  name    = "visionflow-api-ssl-cert-${var.environment}"
  project = var.project_id

  managed {
    domains = [
      "api.${var.domain}",
      var.environment != "prod" ? "${var.environment}-api.${var.domain}" : null
    ]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# SSL Certificate for MLFlow
resource "google_compute_managed_ssl_certificate" "mlflow_ssl_cert" {
  name    = "visionflow-mlflow-ssl-cert-${var.environment}"
  project = var.project_id

  managed {
    domains = [
      "mlflow.${var.domain}",
      var.environment != "prod" ? "${var.environment}-mlflow.${var.domain}" : null
    ]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Health check for API services
resource "google_compute_health_check" "api_health_check" {
  name               = "visionflow-api-health-check-${var.environment}"
  project            = var.project_id
  check_interval_sec = 30
  timeout_sec        = 5
  healthy_threshold  = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 8000
    request_path = "/health"
  }

  log_config {
    enable = true
  }
}

# Health check for MLFlow
resource "google_compute_health_check" "mlflow_health_check" {
  name               = "visionflow-mlflow-health-check-${var.environment}"
  project            = var.project_id
  check_interval_sec = 30
  timeout_sec        = 5
  healthy_threshold  = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 5000
    request_path = "/health"
  }

  log_config {
    enable = true
  }
}

# Backend service for API
resource "google_compute_backend_service" "api_backend" {
  name                    = "visionflow-api-backend-${var.environment}"
  project                 = var.project_id
  protocol                = "HTTP"
  port_name               = "http"
  timeout_sec             = 30
  enable_cdn              = false
  load_balancing_scheme   = "EXTERNAL"
  health_checks           = [google_compute_health_check.api_health_check.id]

  backend {
    group                 = "zones/${var.region}-a/instanceGroups/gke-${var.cluster_name}-default-pool"
    balancing_mode        = "UTILIZATION"
    capacity_scaler       = 1.0
    max_utilization       = 0.8
  }

  iap {
    oauth2_client_id     = var.iap_oauth2_client_id
    oauth2_client_secret = var.iap_oauth2_client_secret
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }

  connection_draining_timeout_sec = 300
}

# Backend service for MLFlow
resource "google_compute_backend_service" "mlflow_backend" {
  name                    = "visionflow-mlflow-backend-${var.environment}"
  project                 = var.project_id
  protocol                = "HTTP"
  port_name               = "http"
  timeout_sec             = 30
  enable_cdn              = false
  load_balancing_scheme   = "EXTERNAL"
  health_checks           = [google_compute_health_check.mlflow_health_check.id]

  backend {
    group                 = "zones/${var.region}-a/instanceGroups/gke-${var.cluster_name}-default-pool"
    balancing_mode        = "UTILIZATION"
    capacity_scaler       = 1.0
    max_utilization       = 0.8
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }

  connection_draining_timeout_sec = 300
}

# URL map for API
resource "google_compute_url_map" "api_url_map" {
  name            = "visionflow-api-url-map-${var.environment}"
  project         = var.project_id
  default_service = google_compute_backend_service.api_backend.id

  host_rule {
    hosts        = ["api.${var.domain}"]
    path_matcher = "api-paths"
  }

  path_matcher {
    name            = "api-paths"
    default_service = google_compute_backend_service.api_backend.id

    path_rule {
      paths   = ["/api/*"]
      service = google_compute_backend_service.api_backend.id
    }

    path_rule {
      paths   = ["/health"]
      service = google_compute_backend_service.api_backend.id
    }

    path_rule {
      paths   = ["/metrics"]
      service = google_compute_backend_service.api_backend.id
    }
  }
}

# URL map for MLFlow
resource "google_compute_url_map" "mlflow_url_map" {
  name            = "visionflow-mlflow-url-map-${var.environment}"
  project         = var.project_id
  default_service = google_compute_backend_service.mlflow_backend.id

  host_rule {
    hosts        = ["mlflow.${var.domain}"]
    path_matcher = "mlflow-paths"
  }

  path_matcher {
    name            = "mlflow-paths"
    default_service = google_compute_backend_service.mlflow_backend.id
  }
}

# HTTPS proxy for API
resource "google_compute_target_https_proxy" "api_https_proxy" {
  name             = "visionflow-api-https-proxy-${var.environment}"
  project          = var.project_id
  url_map          = google_compute_url_map.api_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.api_ssl_cert.id]
}

# HTTPS proxy for MLFlow
resource "google_compute_target_https_proxy" "mlflow_https_proxy" {
  name             = "visionflow-mlflow-https-proxy-${var.environment}"
  project          = var.project_id
  url_map          = google_compute_url_map.mlflow_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.mlflow_ssl_cert.id]
}

# HTTP to HTTPS redirect for API
resource "google_compute_url_map" "api_http_redirect" {
  name    = "visionflow-api-http-redirect-${var.environment}"
  project = var.project_id

  default_url_redirect {
    https_redirect = true
    strip_query    = false
  }
}

resource "google_compute_target_http_proxy" "api_http_proxy" {
  name    = "visionflow-api-http-proxy-${var.environment}"
  project = var.project_id
  url_map = google_compute_url_map.api_http_redirect.id
}

# Global forwarding rule for API HTTPS
resource "google_compute_global_forwarding_rule" "api_https_forwarding_rule" {
  name                  = "visionflow-api-https-forwarding-rule-${var.environment}"
  project               = var.project_id
  ip_protocol           = "TCP"
  load_balancing_scheme = "EXTERNAL"
  port_range            = "443"
  target                = google_compute_target_https_proxy.api_https_proxy.id
  ip_address            = google_compute_global_address.api_ip.id
}

# Global forwarding rule for API HTTP (redirect)
resource "google_compute_global_forwarding_rule" "api_http_forwarding_rule" {
  name                  = "visionflow-api-http-forwarding-rule-${var.environment}"
  project               = var.project_id
  ip_protocol           = "TCP"
  load_balancing_scheme = "EXTERNAL"
  port_range            = "80"
  target                = google_compute_target_http_proxy.api_http_proxy.id
  ip_address            = google_compute_global_address.api_ip.id
}

# Global forwarding rule for MLFlow HTTPS
resource "google_compute_global_forwarding_rule" "mlflow_https_forwarding_rule" {
  name                  = "visionflow-mlflow-https-forwarding-rule-${var.environment}"
  project               = var.project_id
  ip_protocol           = "TCP"
  load_balancing_scheme = "EXTERNAL"
  port_range            = "443"
  target                = google_compute_target_https_proxy.mlflow_https_proxy.id
  ip_address            = google_compute_global_address.mlflow_ip.id
}

# Cloud Armor security policy
resource "google_compute_security_policy" "api_security_policy" {
  name    = "visionflow-api-security-policy-${var.environment}"
  project = var.project_id

  description = "Security policy for VisionFlow API"

  # Default rule - allow all traffic
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }

  # Block known bad actors
  rule {
    action   = "deny(403)"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = var.blocked_ip_ranges
      }
    }
    description = "Block known bad IP ranges"
  }

  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1001"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      ban_duration_sec = 600
    }
    description = "Rate limiting rule"
  }

  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable = true
    }
  }
}

# Attach security policy to backend service
resource "google_compute_backend_service" "api_backend_with_policy" {
  name                    = "visionflow-api-backend-secure-${var.environment}"
  project                 = var.project_id
  protocol                = "HTTP"
  port_name               = "http"
  timeout_sec             = 30
  enable_cdn              = false
  load_balancing_scheme   = "EXTERNAL"
  health_checks           = [google_compute_health_check.api_health_check.id]
  security_policy         = google_compute_security_policy.api_security_policy.id

  backend {
    group                 = "zones/${var.region}-a/instanceGroups/gke-${var.cluster_name}-default-pool"
    balancing_mode        = "UTILIZATION"
    capacity_scaler       = 1.0
    max_utilization       = 0.8
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }

  connection_draining_timeout_sec = 300
}
