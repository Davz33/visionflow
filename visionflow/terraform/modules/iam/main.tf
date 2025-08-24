# IAM Module for VisionFlow

# GKE Service Account
resource "google_service_account" "gke_service_account" {
  account_id   = "gke-${var.environment}-sa"
  display_name = "GKE Service Account for ${var.environment}"
  description  = "Service account for GKE nodes in ${var.environment} environment"
  project      = var.project_id
}

# Workload Identity Service Account
resource "google_service_account" "workload_identity_sa" {
  account_id   = "visionflow-workload-sa"
  display_name = "VisionFlow Workload Identity SA"
  description  = "Service account for VisionFlow workload identity"
  project      = var.project_id
}

# GKE Service Account IAM Bindings
resource "google_project_iam_member" "gke_service_account_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/artifactregistry.reader"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_service_account.email}"
}

# Workload Identity IAM Bindings
resource "google_project_iam_member" "workload_identity_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/secretmanager.secretAccessor",
    "roles/aiplatform.user",
    "roles/monitoring.metricWriter",
    "roles/cloudsql.client",
    "roles/redis.editor"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workload_identity_sa.email}"
}

# Workload Identity binding
resource "google_service_account_iam_binding" "workload_identity_binding" {
  service_account_id = google_service_account.workload_identity_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[visionflow/visionflow-sa]",
  ]
}

# Cloud Build Service Account (for CI/CD)
resource "google_service_account" "cloudbuild_sa" {
  account_id   = "cloudbuild-${var.environment}-sa"
  display_name = "Cloud Build Service Account for ${var.environment}"
  description  = "Service account for Cloud Build CI/CD pipelines"
  project      = var.project_id
}

resource "google_project_iam_member" "cloudbuild_roles" {
  for_each = toset([
    "roles/container.developer",
    "roles/storage.admin",
    "roles/cloudbuild.builds.builder",
    "roles/secretmanager.secretAccessor",
    "roles/source.reader"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloudbuild_sa.email}"
}

# MLFlow Service Account
resource "google_service_account" "mlflow_sa" {
  account_id   = "mlflow-${var.environment}-sa"
  display_name = "MLFlow Service Account for ${var.environment}"
  description  = "Service account for MLFlow tracking server"
  project      = var.project_id
}

resource "google_project_iam_member" "mlflow_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.mlflow_sa.email}"
}

# Custom IAM Role for VisionFlow
resource "google_project_iam_custom_role" "visionflow_role" {
  role_id     = "visionflow_${var.environment}_role"
  title       = "VisionFlow ${var.environment} Role"
  description = "Custom role for VisionFlow application"
  project     = var.project_id

  permissions = [
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.update",
    "storage.buckets.get",
    "aiplatform.endpoints.predict",
    "aiplatform.models.predict",
    "secretmanager.versions.access",
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.timeSeries.create",
    "redis.instances.get"
  ]
}

# Bind custom role to workload identity service account
resource "google_project_iam_member" "workload_identity_custom_role" {
  project = var.project_id
  role    = google_project_iam_custom_role.visionflow_role.name
  member  = "serviceAccount:${google_service_account.workload_identity_sa.email}"
}

# Security Admin Service Account (for infrastructure management)
resource "google_service_account" "security_admin_sa" {
  account_id   = "security-admin-${var.environment}"
  display_name = "Security Admin for ${var.environment}"
  description  = "Service account for security and infrastructure management"
  project      = var.project_id
}

resource "google_project_iam_member" "security_admin_roles" {
  for_each = toset([
    "roles/security.admin",
    "roles/iam.securityAdmin",
    "roles/compute.securityAdmin",
    "roles/container.clusterAdmin"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.security_admin_sa.email}"
}

# Service Account Keys (only if absolutely necessary)
resource "google_service_account_key" "workload_identity_key" {
  count              = var.create_service_account_keys ? 1 : 0
  service_account_id = google_service_account.workload_identity_sa.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# IAM Policy for the project (principle of least privilege)
data "google_iam_policy" "project_policy" {
  binding {
    role = "roles/viewer"
    members = [
      "serviceAccount:${google_service_account.gke_service_account.email}",
    ]
  }

  binding {
    role = "roles/container.nodeServiceAccount"
    members = [
      "serviceAccount:${google_service_account.gke_service_account.email}",
    ]
  }

  binding {
    role = google_project_iam_custom_role.visionflow_role.name
    members = [
      "serviceAccount:${google_service_account.workload_identity_sa.email}",
    ]
  }
}

# Organization policies (if managing at org level)
resource "google_organization_policy" "disable_sa_key_creation" {
  count      = var.organization_id != null ? 1 : 0
  org_id     = var.organization_id
  constraint = "iam.disableServiceAccountKeyCreation"

  boolean_policy {
    enforced = true
  }
}

resource "google_organization_policy" "require_shielded_vm" {
  count      = var.organization_id != null ? 1 : 0
  org_id     = var.organization_id
  constraint = "compute.requireShieldedVm"

  boolean_policy {
    enforced = true
  }
}
