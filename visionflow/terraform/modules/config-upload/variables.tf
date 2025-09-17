variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., production, staging, development)"
  type        = string
}

variable "tags" {
  description = "A map of tags to assign to the resources"
  type        = map(string)
  default     = {}
}

# S3 Bucket Creation Control Variables
variable "create_config_bucket" {
  description = "Whether to create the configuration bucket for WAN2.1"
  type        = bool
  default     = true
}

# Optional: Override bucket name
variable "config_bucket_name" {
  description = "Name for the configuration bucket (optional, will be auto-generated if not provided)"
  type        = string
  default     = ""
}
