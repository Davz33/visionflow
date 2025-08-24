# VPC Module Outputs

output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc.name
}

output "vpc_network_id" {
  description = "ID of the VPC network"
  value       = google_compute_network.vpc.id
}

output "vpc_network_self_link" {
  description = "Self link of the VPC network"
  value       = google_compute_network.vpc.self_link
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = google_compute_subnetwork.subnet.name
}

output "subnet_id" {
  description = "ID of the subnet"
  value       = google_compute_subnetwork.subnet.id
}

output "subnet_self_link" {
  description = "Self link of the subnet"
  value       = google_compute_subnetwork.subnet.self_link
}

output "subnet_cidr" {
  description = "CIDR of the subnet"
  value       = google_compute_subnetwork.subnet.ip_cidr_range
}

output "secondary_ranges" {
  description = "Secondary IP ranges"
  value = {
    for range in google_compute_subnetwork.subnet.secondary_ip_range :
    range.range_name => range.ip_cidr_range
  }
}

output "router_name" {
  description = "Name of the Cloud Router"
  value       = google_compute_router.router.name
}

output "nat_name" {
  description = "Name of the Cloud NAT"
  value       = google_compute_router_nat.nat.name
}

output "private_vpc_connection_id" {
  description = "ID of the private VPC connection"
  value       = google_service_networking_connection.private_vpc_connection.network
}
