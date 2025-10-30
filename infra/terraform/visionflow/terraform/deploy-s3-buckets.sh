#!/bin/bash

# Deploy S3 Buckets for VisionFlow
# This script creates S3 buckets using the standalone configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    if [ ! -f "aws.tfvars" ]; then
        error "aws.tfvars file not found in current directory"
        exit 1
    fi
    
    if [ ! -d "configs" ]; then
        error "configs directory not found in current directory"
        exit 1
    fi
    
    # Check if config files exist
    local missing_files=()
    for file in "configs/wan2-1.env.template" "configs/docker-compose.yml" "configs/wan2-1.service" "configs/amazon-cloudwatch-agent.json"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        error "Missing configuration files: ${missing_files[*]}"
        exit 1
    fi
    
    log "All prerequisites met"
}

# Initialize Terraform
init_terraform() {
    log "Initializing Terraform..."
    terraform init
}

# Plan the deployment
plan_deployment() {
    log "Planning S3 bucket deployment..."
    terraform plan -var-file="aws.tfvars" -out="s3-buckets.tfplan"
}

# Apply the deployment
apply_deployment() {
    log "Applying S3 bucket deployment..."
    terraform apply "s3-buckets.tfplan"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Get bucket names (handle null values)
    local config_bucket=$(terraform output -raw config_bucket_name 2>/dev/null || echo "null")
    local media_bucket=$(terraform output -raw media_bucket_name 2>/dev/null || echo "null")
    local mlflow_bucket=$(terraform output -raw mlflow_bucket_name 2>/dev/null || echo "null")
    
    log "Created buckets:"
    if [ "$config_bucket" != "null" ] && [ -n "$config_bucket" ]; then
        info "  Config bucket: $config_bucket"
    else
        info "  Config bucket: Not created"
    fi
    
    if [ "$media_bucket" != "null" ] && [ -n "$media_bucket" ]; then
        info "  Media bucket: $media_bucket"
    else
        info "  Media bucket: Not created"
    fi
    
    if [ "$mlflow_bucket" != "null" ] && [ -n "$mlflow_bucket" ]; then
        info "  MLflow bucket: $mlflow_bucket"
    else
        info "  MLflow bucket: Not created"
    fi
    
    # Check if configs were uploaded (only if config bucket exists)
    if [ "$config_bucket" != "null" ] && [ -n "$config_bucket" ]; then
        log "Checking uploaded configuration files..."
        aws s3 ls "s3://$config_bucket/configs/" --region eu-west-2
    fi
    
    # Test bucket access
    log "Testing bucket access..."
    
    if [ "$config_bucket" != "null" ] && [ -n "$config_bucket" ]; then
        if aws s3 ls "s3://$config_bucket/" --region eu-west-2 >/dev/null 2>&1; then
            log "✅ Config bucket accessible"
        else
            error "❌ Config bucket not accessible"
            return 1
        fi
    fi
    
    if [ "$media_bucket" != "null" ] && [ -n "$media_bucket" ]; then
        if aws s3 ls "s3://$media_bucket/" --region eu-west-2 >/dev/null 2>&1; then
            log "✅ Media bucket accessible"
        else
            error "❌ Media bucket not accessible"
            return 1
        fi
    fi
    
    if [ "$mlflow_bucket" != "null" ] && [ -n "$mlflow_bucket" ]; then
        if aws s3 ls "s3://$mlflow_bucket/" --region eu-west-2 >/dev/null 2>&1; then
            log "✅ MLflow bucket accessible"
        else
            error "❌ MLflow bucket not accessible"
            return 1
        fi
    fi
}

# Show usage information
show_usage() {
    log "S3 bucket deployment completed successfully!"
    echo ""
    info "Next steps:"
    echo "  1. Update your EC2 instance to use the new config bucket:"
    echo "     export S3_BUCKET=\$(terraform output -raw config_bucket_name)"
    echo "     ./update-running-instance.sh i-08fb48c8f6293cd97 eu-west-2 s3 \$S3_BUCKET"
    echo ""
    echo "  2. Or use the SSM update method:"
    echo "     ./ssm-update-instance.sh i-08fb48c8f6293cd97 eu-west-2 your_api_key s3 \$S3_BUCKET"
    echo ""
    echo "  3. Check bucket contents:"
    echo "     aws s3 ls s3://\$S3_BUCKET/configs/ --region eu-west-2"
    echo ""
    echo "  4. Get all bucket information:"
    echo "     terraform output"
}

# Cleanup function
cleanup() {
    if [ -f "s3-buckets.tfplan" ]; then
        rm -f "s3-buckets.tfplan"
    fi
}

# Main execution
main() {
    log "Starting S3 bucket deployment for VisionFlow..."
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Execute steps
    check_prerequisites
    init_terraform
    plan_deployment
    
    # Ask for confirmation
    echo ""
    warning "This will create S3 buckets in your AWS account. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi
    
    apply_deployment
    verify_deployment
    show_usage
    
    log "S3 bucket deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "plan")
        check_prerequisites
        init_terraform
        plan_deployment
        log "Plan completed. Review the output above."
        ;;
    "apply")
        check_prerequisites
        init_terraform
        plan_deployment
        apply_deployment
        verify_deployment
        show_usage
        ;;
    "destroy")
        warning "This will destroy all S3 buckets and their contents. Continue? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            terraform destroy -var-file="aws.tfvars"
            log "S3 buckets destroyed"
        else
            log "Destroy cancelled by user"
        fi
        ;;
    "verify")
        verify_deployment
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [plan|apply|destroy|verify|help]"
        echo ""
        echo "Commands:"
        echo "  plan     - Plan the S3 bucket deployment"
        echo "  apply    - Create the S3 buckets"
        echo "  destroy  - Destroy the S3 buckets"
        echo "  verify   - Verify the deployment"
        echo "  help     - Show this help message"
        echo ""
        echo "Default: Run full deployment (plan + apply + verify)"
        ;;
    "")
        main
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
