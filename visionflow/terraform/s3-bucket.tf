# Additional S3 Buckets for VisionFlow
# This file creates media and MLflow buckets for VisionFlow

# Random string for unique bucket names (reuse existing)
resource "random_string" "media_bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Note: Configuration bucket is defined in config-upload.tf

# Media storage bucket for VisionFlow
resource "aws_s3_bucket" "visionflow_media_bucket" {
  count  = var.create_media_bucket ? 1 : 0
  bucket = var.media_bucket_name != "" ? var.media_bucket_name : "visionflow-media-${random_string.media_bucket_suffix.result}"
  
  tags = merge(var.tags, {
    Name        = "VisionFlow Media Storage"
    Environment = "production"
    Service     = "visionflow"
    Purpose     = "media-storage"
  })
}

# Versioning for media bucket
resource "aws_s3_bucket_versioning" "visionflow_media_versioning" {
  count  = var.create_media_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_media_bucket[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# Encryption for media bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "visionflow_media_encryption" {
  count  = var.create_media_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_media_bucket[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Public access block for media bucket
resource "aws_s3_bucket_public_access_block" "visionflow_media_pab" {
  count  = var.create_media_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_media_bucket[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CORS configuration for media bucket (for web access)
resource "aws_s3_bucket_cors_configuration" "visionflow_media_cors" {
  count  = var.create_media_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_media_bucket[0].id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    allowed_origins = ["*"]  # Restrict this in production
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# MLflow artifacts bucket
resource "aws_s3_bucket" "visionflow_mlflow_bucket" {
  count  = var.create_mlflow_bucket ? 1 : 0
  bucket = var.mlflow_bucket_name != "" ? var.mlflow_bucket_name : "visionflow-mlflow-${random_string.media_bucket_suffix.result}"
  
  tags = merge(var.tags, {
    Name        = "VisionFlow MLflow Artifacts"
    Environment = "production"
    Service     = "mlflow"
    Purpose     = "ml-artifacts"
  })
}

# Versioning for MLflow bucket
resource "aws_s3_bucket_versioning" "visionflow_mlflow_versioning" {
  count  = var.create_mlflow_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_mlflow_bucket[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# Encryption for MLflow bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "visionflow_mlflow_encryption" {
  count  = var.create_mlflow_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_mlflow_bucket[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Public access block for MLflow bucket
resource "aws_s3_bucket_public_access_block" "visionflow_mlflow_pab" {
  count  = var.create_mlflow_bucket ? 1 : 0
  bucket = aws_s3_bucket.visionflow_mlflow_bucket[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Note: IAM roles and config file uploads are handled in config-upload.tf

# Outputs for additional buckets
output "media_bucket_name" {
  description = "Name of the S3 bucket for media storage"
  value       = var.create_media_bucket ? aws_s3_bucket.visionflow_media_bucket[0].bucket : null
}

output "media_bucket_arn" {
  description = "ARN of the S3 bucket for media storage"
  value       = var.create_media_bucket ? aws_s3_bucket.visionflow_media_bucket[0].arn : null
}

output "mlflow_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = var.create_mlflow_bucket ? aws_s3_bucket.visionflow_mlflow_bucket[0].bucket : null
}

output "mlflow_bucket_arn" {
  description = "ARN of the S3 bucket for MLflow artifacts"
  value       = var.create_mlflow_bucket ? aws_s3_bucket.visionflow_mlflow_bucket[0].arn : null
}
