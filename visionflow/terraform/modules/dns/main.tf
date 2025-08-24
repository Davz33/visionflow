# DNS Module for VisionFlow

# Create DNS zone
resource "google_dns_managed_zone" "main" {
  name        = var.dns_zone_name
  dns_name    = "${var.domain}."
  project     = var.project_id
  description = "DNS zone for VisionFlow ${var.environment} environment"

  labels = merge(var.labels, {
    purpose = "dns-zone"
  })

  dnssec_config {
    state         = "on"
    non_existence = "nsec3"
    
    default_key_specs {
      algorithm  = "rsasha256"
      key_length = 2048
      key_type   = "keySigning"
    }
    
    default_key_specs {
      algorithm  = "rsasha256"
      key_length = 1024
      key_type   = "zoneSigning"
    }
  }
}

# A record for API
resource "google_dns_record_set" "api" {
  name         = "api.${google_dns_managed_zone.main.dns_name}"
  managed_zone = google_dns_managed_zone.main.name
  type         = "A"
  ttl          = 300
  project      = var.project_id

  rrdatas = [var.api_ip_address]
}

# A record for MLFlow
resource "google_dns_record_set" "mlflow" {
  name         = "mlflow.${google_dns_managed_zone.main.dns_name}"
  managed_zone = google_dns_managed_zone.main.name
  type         = "A"
  ttl          = 300
  project      = var.project_id

  rrdatas = [var.mlflow_ip_address]
}

# CNAME for www (redirect to apex)
resource "google_dns_record_set" "www" {
  name         = "www.${google_dns_managed_zone.main.dns_name}"
  managed_zone = google_dns_managed_zone.main.name
  type         = "CNAME"
  ttl          = 300
  project      = var.project_id

  rrdatas = ["api.${var.domain}."]
}

# Environment-specific subdomains (for non-prod)
resource "google_dns_record_set" "env_api" {
  count        = var.environment != "prod" ? 1 : 0
  name         = "${var.environment}-api.${google_dns_managed_zone.main.dns_name}"
  managed_zone = google_dns_managed_zone.main.name
  type         = "A"
  ttl          = 300
  project      = var.project_id

  rrdatas = [var.api_ip_address]
}

resource "google_dns_record_set" "env_mlflow" {
  count        = var.environment != "prod" ? 1 : 0
  name         = "${var.environment}-mlflow.${google_dns_managed_zone.main.dns_name}"
  managed_zone = google_dns_managed_zone.main.name
  type         = "A"
  ttl          = 300
  project      = var.project_id

  rrdatas = [var.mlflow_ip_address]
}

# MX records for email (if needed)
resource "google_dns_record_set" "mx" {
  count        = length(var.mx_records) > 0 ? 1 : 0
  name         = google_dns_managed_zone.main.dns_name
  managed_zone = google_dns_managed_zone.main.name
  type         = "MX"
  ttl          = 3600
  project      = var.project_id

  rrdatas = var.mx_records
}

# TXT records (for verification, SPF, etc.)
resource "google_dns_record_set" "txt" {
  for_each = var.txt_records

  name         = each.key == "@" ? google_dns_managed_zone.main.dns_name : "${each.key}.${google_dns_managed_zone.main.dns_name}"
  managed_zone = google_dns_managed_zone.main.name
  type         = "TXT"
  ttl          = 300
  project      = var.project_id

  rrdatas = [each.value]
}

# CAA records for certificate authority authorization
resource "google_dns_record_set" "caa" {
  count        = length(var.caa_records) > 0 ? 1 : 0
  name         = google_dns_managed_zone.main.dns_name
  managed_zone = google_dns_managed_zone.main.name
  type         = "CAA"
  ttl          = 3600
  project      = var.project_id

  rrdatas = var.caa_records
}

# Health check for DNS resolution
resource "google_dns_policy" "default" {
  name                      = "visionflow-dns-policy-${var.environment}"
  project                   = var.project_id
  enable_inbound_forwarding = true
  enable_logging            = true

  networks {
    network_url = var.vpc_network_url
  }

  alternative_name_server_config {
    target_name_servers {
      ipv4_address    = "8.8.8.8"
      forwarding_path = "default"
    }
    target_name_servers {
      ipv4_address    = "8.8.4.4"
      forwarding_path = "default"
    }
  }
}

# DNS peering (if needed for hybrid connectivity)
resource "google_dns_peering_zone" "private" {
  count       = var.enable_private_zone ? 1 : 0
  name        = "visionflow-private-zone-${var.environment}"
  dns_name    = "private.${var.domain}."
  project     = var.project_id
  description = "Private DNS zone for internal services"

  target_network {
    network_url = var.vpc_network_url
  }

  labels = merge(var.labels, {
    purpose = "private-dns"
  })
}

# Internal DNS records for services
resource "google_dns_record_set" "internal_services" {
  for_each = var.internal_services

  name         = "${each.key}.private.${google_dns_managed_zone.main.dns_name}"
  managed_zone = var.enable_private_zone ? google_dns_peering_zone.private[0].name : google_dns_managed_zone.main.name
  type         = "A"
  ttl          = 300
  project      = var.project_id

  rrdatas = [each.value]
}
