#!/bin/bash

# VisionFlow Terraform Initialization Script
# This script initializes Terraform and sets up the GCP backend

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"visionflow-gcp-project"}
REGION=${REGION:-"us-central1"}
TERRAFORM_BUCKET=${TERRAFORM_BUCKET:-"visionflow-terraform-state"}
ENVIRONMENT=${ENVIRONMENT:-"prod"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated with gcloud
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Set GCP project
set_project() {
    log_info "Setting GCP project to $PROJECT_ID..."
    
    if ! gcloud config set project "$PROJECT_ID"; then
        log_error "Failed to set project. Make sure the project exists and you have access."
        exit 1
    fi
    
    log_success "Project set to $PROJECT_ID"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required GCP APIs..."
    
    local apis=(
        "compute.googleapis.com"
        "container.googleapis.com"
        "storage.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "dns.googleapis.com"
        "servicenetworking.googleapis.com"
        "sqladmin.googleapis.com"
        "secretmanager.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "aiplatform.googleapis.com"
        "cloudbuild.googleapis.com"
        "artifactregistry.googleapis.com"
        "redis.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID"
    done
    
    log_success "APIs enabled successfully"
}

# Create Terraform state bucket
create_terraform_bucket() {
    log_info "Creating Terraform state bucket: $TERRAFORM_BUCKET..."
    
    # Check if bucket already exists
    if gsutil ls -b "gs://$TERRAFORM_BUCKET" &> /dev/null; then
        log_warning "Bucket $TERRAFORM_BUCKET already exists"
        return 0
    fi
    
    # Create bucket
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$TERRAFORM_BUCKET"
    
    # Enable versioning
    gsutil versioning set on "gs://$TERRAFORM_BUCKET"
    
    # Set uniform bucket-level access
    gsutil uniformbucketlevelaccess set on "gs://$TERRAFORM_BUCKET"
    
    # Set public access prevention
    gsutil pap set enforced "gs://$TERRAFORM_BUCKET"
    
    log_success "Terraform state bucket created successfully"
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."
    
    # Create backend configuration
    cat > backend.tf << EOF
terraform {
  backend "gcs" {
    bucket = "$TERRAFORM_BUCKET"
    prefix = "terraform/state"
  }
}
EOF
    
    # Initialize Terraform
    terraform init
    
    log_success "Terraform initialized successfully"
}

# Validate Terraform configuration
validate_terraform() {
    log_info "Validating Terraform configuration..."
    
    terraform validate
    
    log_success "Terraform configuration is valid"
}

# Create terraform.tfvars if it doesn't exist
create_tfvars() {
    if [[ ! -f "terraform.tfvars" ]]; then
        log_info "Creating terraform.tfvars from example..."
        cp terraform.tfvars.example terraform.tfvars
        
        # Update project_id in terraform.tfvars
        sed -i.bak "s/visionflow-gcp-project/$PROJECT_ID/g" terraform.tfvars
        rm terraform.tfvars.bak
        
        log_warning "Please edit terraform.tfvars and update the values before running terraform plan/apply"
    else
        log_info "terraform.tfvars already exists"
    fi
}

# Generate Terraform plan
plan_terraform() {
    log_info "Generating Terraform plan..."
    
    terraform plan -out=tfplan
    
    log_success "Terraform plan generated successfully"
    log_info "Review the plan and run 'terraform apply tfplan' to apply changes"
}

# Main execution
main() {
    log_info "Starting VisionFlow Terraform initialization..."
    
    check_prerequisites
    set_project
    enable_apis
    create_terraform_bucket
    create_tfvars
    init_terraform
    validate_terraform
    
    if [[ "${SKIP_PLAN:-false}" != "true" ]]; then
        plan_terraform
    fi
    
    log_success "VisionFlow Terraform initialization completed!"
    
    echo
    log_info "Next steps:"
    echo "1. Edit terraform.tfvars and update the configuration values"
    echo "2. Review the Terraform plan: terraform show tfplan"
    echo "3. Apply the changes: terraform apply tfplan"
    echo "4. Configure kubectl: gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
