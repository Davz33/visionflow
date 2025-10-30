# Configuration file upload to S3
# This module uploads configuration files to S3 for use by EC2 instances

resource "aws_s3_bucket" "config_bucket" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = var.config_bucket_name != "" ? var.config_bucket_name : "${var.project_name}-wan2-1-configs-${random_string.bucket_suffix.result}"
  
  tags = merge(var.tags, {
    Name        = "WAN2.1 Configuration Files"
    Environment = var.environment
    Service     = "wan2-1"
  })
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "config_bucket_versioning" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "config_bucket_encryption" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "config_bucket_pab" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Upload configuration files to S3 (only if config bucket is created)
resource "aws_s3_object" "env_template" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id
  key    = "configs/wan2-1.env.template"
  source = "${path.root}/configs/wan2-1.env.template"
  etag   = filemd5("${path.root}/configs/wan2-1.env.template")
  
  tags = var.tags
}

resource "aws_s3_object" "docker_compose" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id
  key    = "configs/docker-compose.yml"
  source = "${path.root}/configs/docker-compose.yml"
  etag   = filemd5("${path.root}/configs/docker-compose.yml")
  
  tags = var.tags
}

resource "aws_s3_object" "systemd_service" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id
  key    = "configs/wan2-1.service"
  source = "${path.root}/configs/wan2-1.service"
  etag   = filemd5("${path.root}/configs/wan2-1.service")
  
  tags = var.tags
}

resource "aws_s3_object" "cloudwatch_config" {
  count  = var.create_config_bucket ? 1 : 0
  bucket = aws_s3_bucket.config_bucket[0].id
  key    = "configs/amazon-cloudwatch-agent.json"
  source = "${path.root}/configs/amazon-cloudwatch-agent.json"
  etag   = filemd5("${path.root}/configs/amazon-cloudwatch-agent.json")
  
  tags = var.tags
}

# IAM role for EC2 instances to access S3 config bucket
resource "aws_iam_role" "ec2_config_role" {
  name = "${var.project_name}-ec2-config-role"

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

resource "aws_iam_policy" "ec2_config_policy" {
  count       = var.create_config_bucket ? 1 : 0
  name        = "${var.project_name}-ec2-config-policy"
  description = "Policy for EC2 instances to access configuration files"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.config_bucket[0].arn,
          "${aws_s3_bucket.config_bucket[0].arn}/*"
        ]
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ec2_config_policy_attachment" {
  count      = var.create_config_bucket ? 1 : 0
  role       = aws_iam_role.ec2_config_role.name
  policy_arn = aws_iam_policy.ec2_config_policy[0].arn
}

resource "aws_iam_instance_profile" "ec2_config_profile" {
  name = "${var.project_name}-ec2-config-profile"
  role = aws_iam_role.ec2_config_role.name

  tags = var.tags
}

# Outputs
output "config_bucket_name" {
  description = "Name of the S3 bucket containing configuration files"
  value       = var.create_config_bucket ? aws_s3_bucket.config_bucket[0].bucket : null
}

output "config_bucket_arn" {
  description = "ARN of the S3 bucket containing configuration files"
  value       = var.create_config_bucket ? aws_s3_bucket.config_bucket[0].arn : null
}

output "ec2_config_instance_profile_arn" {
  description = "ARN of the IAM instance profile for EC2 config access"
  value       = aws_iam_instance_profile.ec2_config_profile.arn
}
