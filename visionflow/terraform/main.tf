# AWS EC2 + GPU Configuration for VisionFlow WAN2.1
# This configuration handles EC2 instance setup with external configuration files

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Use local state for simplicity
  backend "local" {
    path = "terraform.tfstate"
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# Configuration upload module
module "config_upload" {
  source = "./modules/config-upload"
  
  project_name = "visionflow"
  environment  = "production"
  
  # S3 bucket creation control
  create_config_bucket = var.create_config_bucket
  config_bucket_name   = var.config_bucket_name
  
  tags = var.tags
}

# Security Group for WAN2.1 Service
resource "aws_security_group" "wan2_1_sg" {
  name_prefix = "wan2-1-"
  description = "Security group for WAN2.1 service"
  vpc_id      = var.vpc_id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # WAN2.1 API access
  ingress {
    from_port   = 8002
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "wan2-1-security-group"
  })
}

# IAM Role for EC2 instance
resource "aws_iam_role" "wan2_1_role" {
  name = "wan2-1-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM Policy for EC2 instance
resource "aws_iam_role_policy" "wan2_1_policy" {
  name = "wan2-1-ec2-policy"
  role = aws_iam_role.wan2_1_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "wan2_1_profile" {
  name = "wan2-1-ec2-profile"
  role = aws_iam_role.wan2_1_role.name

  tags = var.tags
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "wan2_1_logs" {
  name              = "/aws/ec2/wan2-1"
  retention_in_days = 30

  tags = var.tags
}

# User data script with config source
locals {
  user_data = base64encode(templatefile("${path.module}/user-data.sh", {
    api_key       = var.remote_wan_api_key
    config_source = "s3"
    s3_bucket     = module.config_upload.config_bucket_name
  }))
}

# EC2 Instance
resource "aws_instance" "wan2_1_instance" {
  ami                    = var.aws_ami_id
  instance_type          = var.aws_instance_type
  key_name              = var.aws_key_pair_name
  vpc_security_group_ids = [aws_security_group.wan2_1_sg.id]
  subnet_id             = var.aws_subnet_id
  iam_instance_profile  = aws_iam_instance_profile.wan2_1_profile.name

  root_block_device {
    volume_type = "gp3"
    volume_size = var.volume_size
    encrypted   = true
  }

  user_data = local.user_data

  tags = merge(var.tags, {
    Name = "wan2-1-instance"
  })
}

# Elastic IP
resource "aws_eip" "wan2_1_eip" {
  instance = aws_instance.wan2_1_instance.id
  domain   = "vpc"

  tags = merge(var.tags, {
    Name = "wan2-1-elastic-ip"
  })
}

# Outputs
output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.wan2_1_instance.id
}

output "instance_public_ip" {
  description = "EC2 instance public IP"
  value       = aws_instance.wan2_1_instance.public_ip
}

output "instance_private_ip" {
  description = "EC2 instance private IP"
  value       = aws_instance.wan2_1_instance.private_ip
}

output "elastic_ip" {
  description = "Elastic IP address"
  value       = aws_eip.wan2_1_eip.public_ip
}

output "wan2_1_url" {
  description = "WAN2.1 service URL"
  value       = "http://${aws_eip.wan2_1_eip.public_ip}:8002"
}

output "security_group_id" {
  description = "Security Group ID"
  value       = aws_security_group.wan2_1_sg.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.aws_key_pair_name}.pem ubuntu@${aws_eip.wan2_1_eip.public_ip}"
}
