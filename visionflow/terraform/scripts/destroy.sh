#!/bin/bash

# VisionFlow Infrastructure Destruction Script
# This script safely destroys the VisionFlow infrastructure

set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-"prod"}
FORCE=${FORCE:-false}
BACKUP_DATA=${BACKUP_DATA:-true}

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
    echo "  -e, --environment ENVIRONMENT    Environment to destroy (dev, staging, prod)"
    echo "  -f, --force                      Force destruction without confirmation"
    echo "  --no-backup                      Skip data backup"
    echo "  -h, --help                       Show this help message"
    echo
    echo "DANGER: This will permanently destroy all infrastructure!"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --no-backup)
                BACKUP_DATA=false
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
}

# Show destruction warning
show_warning() {
    echo
    log_warning "╔══════════════════════════════════════════════════════════════════════════════╗"
    log_warning "║                                 DANGER                                      ║"
    log_warning "║                                                                              ║"
    log_warning "║  This will PERMANENTLY DESTROY all VisionFlow infrastructure in the        ║"
    log_warning "║  $ENVIRONMENT environment, including:                                    ║"
    log_warning "║                                                                              ║"
    log_warning "║  • GKE Cluster and all workloads                                            ║"
    log_warning "║  • Databases (PostgreSQL, Redis) and ALL DATA                               ║"
    log_warning "║  • Storage buckets and ALL FILES                                            ║"
    log_warning "║  • Load balancers and networking                                             ║"
    log_warning "║  • DNS records                                                               ║"
    log_warning "║  • Monitoring and alerting                                                   ║"
    log_warning "║  • Secrets and credentials                                                   ║"
    log_warning "║                                                                              ║"
    log_warning "║  THIS CANNOT BE UNDONE!                                                     ║"
    log_warning "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo
}

# Backup data before destruction
backup_data() {
    if [[ "$BACKUP_DATA" != "true" ]]; then
        log_info "Skipping data backup (--no-backup specified)"
        return 0
    fi
    
    log_info "Creating backup before destruction..."
    
    local backup_dir="backups/$(date +%Y%m%d-%H%M%S)-$ENVIRONMENT"
    mkdir -p "$backup_dir"
    
    # Backup Terraform state
    log_info "Backing up Terraform state..."
    if [[ -f "terraform.tfstate" ]]; then
        cp terraform.tfstate "$backup_dir/"
    fi
    
    # Backup Kubernetes resources
    log_info "Backing up Kubernetes resources..."
    if command -v kubectl &> /dev/null; then
        kubectl get all -n visionflow -o yaml > "$backup_dir/k8s-resources.yaml" 2>/dev/null || true
        kubectl get secrets -n visionflow -o yaml > "$backup_dir/k8s-secrets.yaml" 2>/dev/null || true
        kubectl get configmaps -n visionflow -o yaml > "$backup_dir/k8s-configmaps.yaml" 2>/dev/null || true
    fi
    
    # Backup database (if accessible)
    log_info "Attempting database backup..."
    local db_host
    local db_user
    local db_name
    
    if command -v terraform &> /dev/null && [[ -f "terraform.tfstate" ]]; then
        db_host=$(terraform output -raw postgres_private_ip 2>/dev/null || echo "")
        if [[ -n "$db_host" ]]; then
            log_info "Database backup would require manual connection to $db_host"
            echo "Manual database backup command:" > "$backup_dir/db-backup-commands.txt"
            echo "pg_dump -h $db_host -U visionflow -d visionflow > visionflow-backup.sql" >> "$backup_dir/db-backup-commands.txt"
            echo "pg_dump -h $db_host -U mlflow -d mlflow > mlflow-backup.sql" >> "$backup_dir/db-backup-commands.txt"
        fi
    fi
    
    # Backup important files
    log_info "Backing up configuration files..."
    cp terraform.tfvars "$backup_dir/" 2>/dev/null || true
    cp -r ../k8s "$backup_dir/" 2>/dev/null || true
    
    log_success "Backup created in $backup_dir"
    echo "Backup location: $(pwd)/$backup_dir"
}

# Get user confirmation
get_confirmation() {
    if [[ "$FORCE" == "true" ]]; then
        log_warning "Force mode enabled. Skipping confirmation."
        return 0
    fi
    
    echo
    read -p "Type 'destroy' to confirm destruction of $ENVIRONMENT environment: " -r
    if [[ "$REPLY" != "destroy" ]]; then
        log_info "Destruction cancelled by user."
        exit 0
    fi
    
    echo
    read -p "Are you absolutely sure? Type 'yes' to proceed: " -r
    if [[ "$REPLY" != "yes" ]]; then
        log_info "Destruction cancelled by user."
        exit 0
    fi
}

# Delete Kubernetes resources first
delete_kubernetes_resources() {
    log_info "Deleting Kubernetes resources..."
    
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl not found. Skipping Kubernetes cleanup."
        return 0
    fi
    
    # Try to delete the namespace (this will delete all resources in it)
    if kubectl get namespace visionflow &> /dev/null; then
        log_info "Deleting visionflow namespace..."
        kubectl delete namespace visionflow --timeout=300s || log_warning "Failed to delete namespace"
    else
        log_info "visionflow namespace not found"
    fi
    
    # Delete any cluster-level resources
    log_info "Cleaning up cluster-level resources..."
    kubectl delete clusterrolebinding visionflow-cluster-admin &> /dev/null || true
    kubectl delete clusterrole visionflow-cluster-role &> /dev/null || true
    
    log_success "Kubernetes resources deleted"
}

# Destroy infrastructure with Terraform
destroy_infrastructure() {
    log_info "Destroying infrastructure with Terraform..."
    
    # Check if Terraform is initialized
    if [[ ! -d ".terraform" ]]; then
        log_error "Terraform not initialized. Cannot proceed with destruction."
        exit 1
    fi
    
    # Generate destroy plan
    log_info "Generating destruction plan..."
    terraform plan -destroy -var="environment=$ENVIRONMENT" -out=destroy-plan
    
    # Show what will be destroyed
    echo
    log_info "Resources to be destroyed:"
    terraform show -no-color destroy-plan | grep -E "  # .* will be destroyed"
    
    # Apply destruction
    log_info "Applying destruction plan..."
    terraform apply destroy-plan
    
    # Clean up plan file
    rm -f destroy-plan
    
    log_success "Infrastructure destroyed successfully"
}

# Clean up local files
cleanup_local_files() {
    log_info "Cleaning up local files..."
    
    # Remove Terraform files
    rm -f terraform.tfstate.backup
    rm -f tfplan-*
    rm -f destroy-plan
    rm -f deployment-info-*.txt
    
    # Remove .terraform directory if requested
    read -p "Remove .terraform directory? This will require re-initialization. (y/n): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .terraform
        rm -f .terraform.lock.hcl
        log_info ".terraform directory removed"
    fi
    
    log_success "Local cleanup completed"
}

# Main execution
main() {
    log_info "Starting VisionFlow infrastructure destruction..."
    
    validate_environment
    show_warning
    get_confirmation
    backup_data
    delete_kubernetes_resources
    destroy_infrastructure
    cleanup_local_files
    
    log_success "VisionFlow infrastructure destruction completed!"
    
    echo
    log_info "What was destroyed:"
    echo "• All compute resources (GKE cluster, VMs)"
    echo "• All databases and stored data"
    echo "• All storage buckets and files"
    echo "• All networking components"
    echo "• All monitoring and alerting"
    echo "• All secrets and credentials"
    
    if [[ "$BACKUP_DATA" == "true" ]]; then
        log_info "Backup location: $(pwd)/backups/"
    fi
    
    echo
    log_warning "Remember to:"
    echo "• Update your DNS records"
    echo "• Cancel any recurring charges"
    echo "• Notify your team"
    echo "• Update documentation"
}

# Script execution
parse_args "$@"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
