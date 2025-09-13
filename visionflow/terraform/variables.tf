# Minimal AWS Variables for EC2 + GPU Configuration

variable "aws_region" {
  description = "AWS region for EC2 instances"
  type        = string
  default     = "eu-west-1"
}

variable "aws_access_key_id" {
  description = "AWS Access Key ID"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS Secret Access Key"
  type        = string
  sensitive   = true
}

variable "aws_ami_id" {
  description = "AMI ID for EC2 instances"
  type        = string
  default     = "ami-0fb277f6ccd685814"  # Deep Learning AMI (Ubuntu 24.04)
}

variable "aws_instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g5.xlarge"
}

variable "aws_key_pair_name" {
  description = "AWS Key Pair name for SSH access"
  type        = string
}

variable "aws_security_group_id" {
  description = "Existing Security Group ID (optional)"
  type        = string
  default     = ""
}

variable "aws_subnet_id" {
  description = "Subnet ID for EC2 instances"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for resources"
  type        = string
}

variable "volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 100
}

variable "remote_wan_api_key" {
  description = "API key for WAN2.1 service"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    Project     = "visionflow"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}
