#!/bin/bash

# Google Cloud Platform setup script for VisionFlow
# This script sets up the complete GCP environment

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"visionflow-gcp-project"}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"us-central1-a"}
CLUSTER_NAME=${CLUSTER_NAME:-"visionflow-cluster"}
BUCKET_NAME=${BUCKET_NAME:-"visionflow-media-bucket"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
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
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Setup GCP project
setup_project() {
    log_info "Setting up GCP project..."
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable \
        container.googleapis.com \
        storage.googleapis.com \
        aiplatform.googleapis.com \
        secretmanager.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com
    
    log_info "Project setup completed"
}

# Create GKE cluster
create_gke_cluster() {
    log_info "Creating GKE cluster..."
    
    # Check if cluster already exists
    if gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE &> /dev/null; then
        log_warn "Cluster $CLUSTER_NAME already exists"
        return 0
    fi
    
    # Create cluster with GPU support
    gcloud container clusters create $CLUSTER_NAME \
        --zone=$ZONE \
        --machine-type=e2-standard-4 \
        --num-nodes=3 \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=10 \
        --enable-autorepair \
        --enable-autoupgrade \
        --disk-size=50GB \
        --disk-type=pd-standard \
        --image-type=COS_CONTAINERD \
        --enable-ip-alias \
        --network=default \
        --subnetwork=default \
        --enable-stackdriver-kubernetes \
        --enable-autorepair \
        --enable-autoupgrade \
        --maintenance-window-start=2024-01-01T01:00:00Z \
        --maintenance-window-end=2024-01-01T05:00:00Z \
        --maintenance-window-recurrence="FREQ=WEEKLY;BYDAY=SA"
    
    # Create GPU node pool
    log_info "Creating GPU node pool..."
    gcloud container node-pools create gpu-pool \
        --cluster=$CLUSTER_NAME \
        --zone=$ZONE \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --num-nodes=0 \
        --enable-autoscaling \
        --min-nodes=0 \
        --max-nodes=3 \
        --disk-size=100GB \
        --disk-type=pd-ssd \
        --image-type=COS_CONTAINERD \
        --enable-autorepair \
        --enable-autoupgrade
    
    log_info "GKE cluster created successfully"
}

# Setup Cloud Storage
setup_storage() {
    log_info "Setting up Cloud Storage..."
    
    # Check if bucket exists
    if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
        log_warn "Bucket $BUCKET_NAME already exists"
        return 0
    fi
    
    # Create bucket
    gsutil mb -l $REGION gs://$BUCKET_NAME
    
    # Set bucket permissions (adjust as needed)
    gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME
    
    # Configure CORS for web access
    cat > cors.json << EOF
[
    {
        "origin": ["*"],
        "responseHeader": ["Content-Type"],
        "method": ["GET", "HEAD"],
        "maxAgeSeconds": 3600
    }
]
EOF
    
    gsutil cors set cors.json gs://$BUCKET_NAME
    rm cors.json
    
    # Set lifecycle policy
    cat > lifecycle.json << EOF
{
    "rule": [
        {
            "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
            "condition": {"age": 30}
        },
        {
            "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
            "condition": {"age": 90}
        },
        {
            "action": {"type": "Delete"},
            "condition": {"age": 365}
        }
    ]
}
EOF
    
    gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME
    rm lifecycle.json
    
    log_info "Cloud Storage setup completed"
}

# Setup service accounts and IAM
setup_iam() {
    log_info "Setting up IAM and service accounts..."
    
    # Create service account for workload identity
    gcloud iam service-accounts create visionflow-workload \
        --display-name="VisionFlow Workload Identity" \
        --description="Service account for VisionFlow Kubernetes workloads" || true
    
    # Grant necessary permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:visionflow-workload@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/storage.objectAdmin"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:visionflow-workload@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/aiplatform.user"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:visionflow-workload@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:visionflow-workload@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/monitoring.metricWriter"
    
    # Enable workload identity
    gcloud iam service-accounts add-iam-policy-binding \
        visionflow-workload@$PROJECT_ID.iam.gserviceaccount.com \
        --role="roles/iam.workloadIdentityUser" \
        --member="serviceAccount:$PROJECT_ID.svc.id.goog[visionflow/visionflow-sa]"
    
    log_info "IAM setup completed"
}

# Setup secrets
setup_secrets() {
    log_info "Setting up secrets..."
    
    # Create secrets (with placeholder values - replace with actual values)
    echo "placeholder_value" | gcloud secrets create langchain-api-key --data-file=- || true
    echo "placeholder_value" | gcloud secrets create openai-api-key --data-file=- || true
    echo "placeholder_value" | gcloud secrets create huggingface-token --data-file=- || true
    echo "placeholder_value" | gcloud secrets create db-password --data-file=- || true
    echo "placeholder_value" | gcloud secrets create redis-password --data-file=- || true
    
    log_warn "Placeholder secrets created. Please update them with actual values:"
    log_warn "  gcloud secrets versions add langchain-api-key --data-file=<path-to-key>"
    log_warn "  gcloud secrets versions add openai-api-key --data-file=<path-to-key>"
    log_warn "  gcloud secrets versions add huggingface-token --data-file=<path-to-token>"
    log_warn "  gcloud secrets versions add db-password --data-file=<path-to-password>"
    log_warn "  gcloud secrets versions add redis-password --data-file=<path-to-password>"
    
    log_info "Secrets setup completed"
}

# Get cluster credentials
get_credentials() {
    log_info "Getting cluster credentials..."
    gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
    log_info "Cluster credentials configured"
}

# Deploy initial setup
deploy_initial() {
    log_info "Deploying initial Kubernetes resources..."
    
    # Apply namespace and basic resources
    kubectl apply -f ../k8s/namespace.yaml
    kubectl apply -f ../k8s/rbac.yaml
    
    log_info "Initial deployment completed"
}

# Main execution
main() {
    log_info "Starting VisionFlow GCP setup..."
    
    check_prerequisites
    setup_project
    create_gke_cluster
    setup_storage
    setup_iam
    setup_secrets
    get_credentials
    deploy_initial
    
    log_info "GCP setup completed successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Update secrets with actual values"
    log_info "2. Build and push Docker images"
    log_info "3. Deploy the application using: kubectl apply -f k8s/"
    log_info ""
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Region: $REGION"
    log_info "Bucket: gs://$BUCKET_NAME"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
