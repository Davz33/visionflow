# GKE Module Outputs

output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.primary.name
}

output "cluster_id" {
  description = "ID of the GKE cluster"
  value       = google_container_cluster.primary.id
}

output "endpoint" {
  description = "Endpoint of the GKE cluster"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "ca_certificate" {
  description = "CA certificate of the GKE cluster"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "location" {
  description = "Location of the GKE cluster"
  value       = google_container_cluster.primary.location
}

output "node_pool_names" {
  description = "Names of the node pools"
  value       = [for pool in google_container_node_pool.node_pools : pool.name]
}

output "node_pool_zones" {
  description = "Zones of the node pools"
  value = {
    for k, pool in google_container_node_pool.node_pools :
    k => pool.node_locations
  }
}

output "workload_identity_pool" {
  description = "Workload Identity pool"
  value       = google_container_cluster.primary.workload_identity_config[0].workload_pool
}

output "cluster_ipv4_cidr" {
  description = "IPv4 CIDR of the cluster"
  value       = google_container_cluster.primary.cluster_ipv4_cidr
}

output "services_ipv4_cidr" {
  description = "IPv4 CIDR of the services"
  value       = google_container_cluster.primary.services_ipv4_cidr
}

output "master_version" {
  description = "Current master version"
  value       = google_container_cluster.primary.master_version
}
