#!/bin/bash

# VisionFlow Terraform Deployment Script
# This script deploys the complete VisionFlow infrastructure

set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-"prod"}
AUTO_APPROVE=${AUTO_APPROVE:-false}
PLAN_ONLY=${PLAN_ONLY:-false}
DESTROY=${DESTROY:-false}

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

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -e, --environment ENVIRONMENT    Environment (dev, staging, prod) [default: prod]"
    echo "  -a, --auto-approve              Auto-approve terraform apply"
    echo "  -p, --plan-only                 Only generate plan, don't apply"
    echo "  -d, --destroy                   Destroy infrastructure"
    echo "  -h, --help                      Show this help message"
    echo
    echo "Environment variables:"
    echo "  PROJECT_ID                      GCP project ID"
    echo "  REGION                          GCP region [default: us-central1]"
    echo "  ZONE                            GCP zone [default: us-central1-a]"
    echo
    echo "Examples:"
    echo "  $0 --environment dev --plan-only"
    echo "  $0 --environment prod --auto-approve"
    echo "  ENVIRONMENT=staging $0"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -a|--auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            -p|--plan-only)
                PLAN_ONLY=true
                shift
                ;;
            -d|--destroy)
                DESTROY=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
        exit 1
    fi
    
    log_info "Environment: $ENVIRONMENT"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    # Check if terraform.tfvars exists
    if [[ ! -f "terraform.tfvars" ]]; then
        log_error "terraform.tfvars not found. Please create it from terraform.tfvars.example"
        exit 1
    fi
    
    # Check if Terraform is initialized
    if [[ ! -d ".terraform" ]]; then
        log_error "Terraform not initialized. Please run './scripts/init.sh' first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate Terraform configuration
validate_terraform() {
    log_info "Validating Terraform configuration..."
    
    terraform validate
    
    log_success "Terraform configuration is valid"
}

# Format Terraform files
format_terraform() {
    log_info "Formatting Terraform files..."
    
    terraform fmt -recursive
    
    log_success "Terraform files formatted"
}

# Generate Terraform plan
plan_terraform() {
    log_info "Generating Terraform plan for $ENVIRONMENT environment..."
    
    local plan_file="tfplan-$ENVIRONMENT"
    
    if [[ "$DESTROY" == "true" ]]; then
        terraform plan -destroy -var="environment=$ENVIRONMENT" -out="$plan_file"
        log_warning "DESTROY plan generated. This will destroy all infrastructure!"
    else
        terraform plan -var="environment=$ENVIRONMENT" -out="$plan_file"
        log_success "Terraform plan generated successfully"
    fi
    
    echo
    log_info "Plan summary:"
    terraform show -no-color "$plan_file" | grep -E "Plan:|No changes"
    
    return 0
}

# Apply Terraform changes
apply_terraform() {
    local plan_file="tfplan-$ENVIRONMENT"
    
    if [[ "$PLAN_ONLY" == "true" ]]; then
        log_info "Plan-only mode enabled. Skipping apply."
        return 0
    fi
    
    if [[ "$AUTO_APPROVE" != "true" ]]; then
        echo
        if [[ "$DESTROY" == "true" ]]; then
            log_warning "This will DESTROY all infrastructure in $ENVIRONMENT environment!"
        else
            log_info "This will apply changes to $ENVIRONMENT environment."
        fi
        
        read -p "Do you want to proceed? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Deployment cancelled by user."
            exit 0
        fi
    fi
    
    log_info "Applying Terraform changes..."
    
    terraform apply "$plan_file"
    
    if [[ "$DESTROY" == "true" ]]; then
        log_success "Infrastructure destroyed successfully"
    else
        log_success "Infrastructure deployed successfully"
    fi
}

# Configure kubectl
configure_kubectl() {
    if [[ "$DESTROY" == "true" ]]; then
        return 0
    fi
    
    log_info "Configuring kubectl..."
    
    # Get cluster credentials
    local cluster_name
    local zone
    local project_id
    
    cluster_name=$(terraform output -raw gke_cluster_name 2>/dev/null || echo "visionflow-cluster")
    zone=$(terraform output -raw gke_cluster_location 2>/dev/null || echo "us-central1-a")
    project_id=$(terraform output -raw project_id 2>/dev/null || echo "$PROJECT_ID")
    
    if gcloud container clusters get-credentials "$cluster_name" --zone "$zone" --project "$project_id"; then
        log_success "kubectl configured successfully"
        
        # Test cluster connection
        if kubectl cluster-info &> /dev/null; then
            log_success "Cluster connection verified"
        else
            log_warning "Could not verify cluster connection"
        fi
    else
        log_warning "Failed to configure kubectl"
    fi
}

# Display deployment information
show_deployment_info() {
    if [[ "$DESTROY" == "true" ]]; then
        return 0
    fi
    
    log_info "Deployment Information:"
    echo
    
    # Get outputs
    local api_domain
    local mlflow_domain
    local api_ip
    local mlflow_ip
    
    api_domain=$(terraform output -raw api_domain 2>/dev/null || echo "N/A")
    mlflow_domain=$(terraform output -raw mlflow_domain 2>/dev/null || echo "N/A")
    api_ip=$(terraform output -raw api_ip_address 2>/dev/null || echo "N/A")
    mlflow_ip=$(terraform output -raw mlflow_ip_address 2>/dev/null || echo "N/A")
    
    echo "Environment: $ENVIRONMENT"
    echo "API Domain: $api_domain"
    echo "MLFlow Domain: $mlflow_domain"
    echo "API IP: $api_ip"
    echo "MLFlow IP: $mlflow_ip"
    echo
    
    log_info "Next steps:"
    echo "1. Update your DNS records to point to the IP addresses above"
    echo "2. Deploy the Kubernetes applications: kubectl apply -f ../k8s/"
    echo "3. Monitor the deployment: kubectl get pods -n visionflow"
    echo "4. Access the services once DNS propagates"
    
    # Save deployment info to file
    cat > "deployment-info-$ENVIRONMENT.txt" << EOF
VisionFlow Deployment Information
Environment: $ENVIRONMENT
Deployed: $(date)

API Domain: $api_domain
MLFlow Domain: $mlflow_domain
API IP: $api_ip
MLFlow IP: $mlflow_ip

Kubectl command:
$(terraform output -raw kubectl_config_command 2>/dev/null || echo "N/A")
EOF
    
    log_success "Deployment information saved to deployment-info-$ENVIRONMENT.txt"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
    fi
    
    # Remove plan files on exit
    rm -f tfplan-*
    
    exit $exit_code
}

# Main execution
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    log_info "Starting VisionFlow Terraform deployment..."
    echo "Environment: $ENVIRONMENT"
    echo "Auto-approve: $AUTO_APPROVE"
    echo "Plan-only: $PLAN_ONLY"
    echo "Destroy: $DESTROY"
    echo
    
    check_prerequisites
    validate_terraform
    format_terraform
    plan_terraform
    apply_terraform
    configure_kubectl
    show_deployment_info
    
    if [[ "$DESTROY" == "true" ]]; then
        log_success "VisionFlow infrastructure destroyed successfully!"
    else
        log_success "VisionFlow infrastructure deployed successfully!"
    fi
}

# Script execution
parse_args "$@"
validate_environment

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
