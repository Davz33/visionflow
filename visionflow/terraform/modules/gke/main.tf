# GKE Cluster Module

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Network configuration
  network    = var.network
  subnetwork = var.subnetwork

  # IP allocation policy for VPC-native networking
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_range_name
    services_secondary_range_name = var.services_range_name
  }

  # Master auth configuration
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Network policy
  network_policy {
    enabled = true
  }

  # Addons configuration
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    network_policy_config {
      disabled = false
    }

    dns_cache_config {
      enabled = true
    }

    gcp_filestore_csi_driver_config {
      enabled = true
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"

    master_global_access_config {
      enabled = true
    }
  }

  # Master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "10.0.0.0/8"
      display_name = "Private networks"
    }
    cidr_blocks {
      cidr_block   = "172.16.0.0/12"
      display_name = "Private networks"
    }
    cidr_blocks {
      cidr_block   = "192.168.0.0/16"
      display_name = "Private networks"
    }
  }

  # Maintenance policy
  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T01:00:00Z"
      end_time   = "2024-01-01T05:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }

  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Monitoring and logging
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Enable shielded nodes
  enable_shielded_nodes = true

  # Resource labels
  resource_labels = merge(var.labels, {
    cluster = var.cluster_name
  })

  # Lifecycle management
  lifecycle {
    ignore_changes = [
      initial_node_count,
      remove_default_node_pool,
    ]
  }

  depends_on = [
    google_project_service.container,
    google_project_service.compute,
  ]
}

# Enable required services
resource "google_project_service" "container" {
  project = var.project_id
  service = "container.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "compute" {
  project = var.project_id
  service = "compute.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Node pools
resource "google_container_node_pool" "node_pools" {
  for_each = var.node_pools

  name     = each.key
  location = var.region
  cluster  = google_container_cluster.primary.name
  project  = var.project_id

  # Autoscaling configuration
  autoscaling {
    min_node_count = each.value.min_count
    max_node_count = each.value.max_count
  }

  # Node configuration
  node_config {
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size_gb
    disk_type    = each.value.disk_type
    image_type   = each.value.image_type

    # Service account
    service_account = var.service_account

    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ]

    # Security settings
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # GPU configuration (if accelerator is specified)
    dynamic "guest_accelerator" {
      for_each = each.value.accelerator_type != null ? [1] : []
      content {
        type  = each.value.accelerator_type
        count = each.value.accelerator_count
      }
    }

    # Taints (for GPU nodes)
    dynamic "taint" {
      for_each = each.value.taints != null ? each.value.taints : []
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    # Labels
    labels = merge(var.labels, each.value.labels != null ? each.value.labels : {})

    # Preemptible instances
    preemptible = each.value.preemptible

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  # Management settings
  management {
    auto_repair  = each.value.auto_repair
    auto_upgrade = each.value.auto_upgrade
  }

  # Lifecycle management
  lifecycle {
    ignore_changes = [
      initial_node_count,
    ]
  }

  depends_on = [google_container_cluster.primary]
}

# Install NVIDIA GPU drivers for GPU nodes
resource "google_container_node_pool" "gpu_driver_installer" {
  for_each = {
    for k, v in var.node_pools : k => v
    if v.accelerator_type != null
  }

  name     = "${each.key}-gpu-installer"
  location = var.region
  cluster  = google_container_cluster.primary.name
  project  = var.project_id

  initial_node_count = 0

  node_config {
    machine_type = "e2-micro"
    disk_size_gb = 10
    disk_type    = "pd-standard"
    image_type   = "COS_CONTAINERD"

    service_account = var.service_account

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = merge(var.labels, {
      "gpu-driver-installer" = "true"
    })

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  depends_on = [google_container_cluster.primary]
}
