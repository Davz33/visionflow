# Database Module Outputs

# PostgreSQL Outputs
output "postgres_connection_name" {
  description = "PostgreSQL connection name"
  value       = google_sql_database_instance.postgres.connection_name
}

output "postgres_private_ip" {
  description = "PostgreSQL private IP address"
  value       = google_sql_database_instance.postgres.private_ip_address
}

output "postgres_instance_name" {
  description = "PostgreSQL instance name"
  value       = google_sql_database_instance.postgres.name
}

output "postgres_self_link" {
  description = "PostgreSQL instance self link"
  value       = google_sql_database_instance.postgres.self_link
}

output "visionflow_database_name" {
  description = "VisionFlow database name"
  value       = google_sql_database.visionflow_db.name
}

output "mlflow_database_name" {
  description = "MLFlow database name"
  value       = google_sql_database.mlflow_db.name
}

# Redis Outputs
output "redis_host" {
  description = "Redis host address"
  value       = google_redis_instance.redis.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.redis.port
}

output "redis_auth_string" {
  description = "Redis auth string"
  value       = google_redis_instance.redis.auth_string
  sensitive   = true
}

output "redis_instance_id" {
  description = "Redis instance ID"
  value       = google_redis_instance.redis.id
}

output "redis_memory_size" {
  description = "Redis memory size"
  value       = google_redis_instance.redis.memory_size_gb
}

# Connection strings
output "postgres_connection_string" {
  description = "PostgreSQL connection string for VisionFlow"
  value       = "postgresql://visionflow:${var.postgres_password}@${google_sql_database_instance.postgres.private_ip_address}:5432/visionflow"
  sensitive   = true
}

output "mlflow_connection_string" {
  description = "PostgreSQL connection string for MLFlow"
  value       = "postgresql://mlflow:${var.mlflow_password}@${google_sql_database_instance.postgres.private_ip_address}:5432/mlflow"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://:${google_redis_instance.redis.auth_string}@${google_redis_instance.redis.host}:${google_redis_instance.redis.port}/0"
  sensitive   = true
}

# Private networking
output "private_ip_address" {
  description = "Private IP address range"
  value       = google_compute_global_address.private_ip_address.address
}

output "private_vpc_connection" {
  description = "Private VPC connection"
  value       = google_service_networking_connection.private_vpc_connection.network
}
