#!/bin/bash

# SSM Update Instance Script
# This script uses AWS Systems Manager to update a running instance

set -e

# Configuration
INSTANCE_ID="${1:-i-08fb48c8f6293cd97}"
REGION="${2:-eu-west-2}"
API_KEY="${3:-}"
CONFIG_SOURCE="${4:-s3}"
S3_BUCKET="${5:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    if [ -z "$API_KEY" ]; then
        read -p "Enter your API key: " API_KEY
        if [ -z "$API_KEY" ]; then
            error "API key is required"
            exit 1
        fi
    fi
    
    if [ -z "$S3_BUCKET" ]; then
        read -p "Enter your S3 bucket name: " S3_BUCKET
        if [ -z "$S3_BUCKET" ]; then
            error "S3 bucket name is required"
            exit 1
        fi
    fi
}

# Create SSM document
create_ssm_document() {
    log "Creating SSM document..."
    
    DOCUMENT_NAME="wan2-1-update-$(date +%s)"
    
    aws ssm create-document \
        --region $REGION \
        --name $DOCUMENT_NAME \
        --document-type "Command" \
        --document-format "JSON" \
        --content file://ssm-update-document.json \
        --tags Key=Service,Value=wan2-1 Key=Type,Value=update-script
    
    echo $DOCUMENT_NAME
}

# Execute SSM command
execute_ssm_command() {
    local document_name=$1
    
    log "Executing SSM command on instance $INSTANCE_ID..."
    
    COMMAND_ID=$(aws ssm send-command \
        --region $REGION \
        --instance-ids $INSTANCE_ID \
        --document-name $document_name \
        --parameters "apiKey=$API_KEY,configSource=$CONFIG_SOURCE,s3Bucket=$S3_BUCKET" \
        --query 'Command.CommandId' \
        --output text)
    
    log "Command ID: $COMMAND_ID"
    echo "Command ID: $COMMAND_ID"
    
    # Wait for command to complete
    log "Waiting for command to complete..."
    aws ssm wait command-executed \
        --region $REGION \
        --command-id $COMMAND_ID \
        --instance-id $INSTANCE_ID
    
    # Get command output
    log "Getting command output..."
    aws ssm get-command-invocation \
        --region $REGION \
        --command-id $COMMAND_ID \
        --instance-id $INSTANCE_ID \
        --query 'StandardOutputContent' \
        --output text
    
    # Check for errors
    EXIT_CODE=$(aws ssm get-command-invocation \
        --region $REGION \
        --command-id $COMMAND_ID \
        --instance-id $INSTANCE_ID \
        --query 'ResponseCode' \
        --output text)
    
    if [ "$EXIT_CODE" != "0" ]; then
        error "Command failed with exit code: $EXIT_CODE"
        aws ssm get-command-invocation \
            --region $REGION \
            --command-id $COMMAND_ID \
            --instance-id $INSTANCE_ID \
            --query 'StandardErrorContent' \
            --output text
        exit 1
    fi
    
    log "Command executed successfully!"
}

# Clean up SSM document
cleanup_ssm_document() {
    local document_name=$1
    
    log "Cleaning up SSM document..."
    aws ssm delete-document \
        --region $REGION \
        --name $document_name \
        --query 'DocumentDescription.Status' \
        --output text
}

# Main execution
main() {
    log "Starting SSM-based instance update..."
    
    check_prerequisites
    
    # Check if instance is running
    STATUS=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)
    
    if [ "$STATUS" != "running" ]; then
        error "Instance $INSTANCE_ID is not running (status: $STATUS)"
        exit 1
    fi
    
    log "Instance $INSTANCE_ID is running"
    
    # Create and execute SSM document
    DOCUMENT_NAME=$(create_ssm_document)
    execute_ssm_command $DOCUMENT_NAME
    cleanup_ssm_document $DOCUMENT_NAME
    
    log "Instance update completed successfully!"
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <instance-id> [region] [api-key] [config-source] [s3-bucket]"
    echo ""
    echo "Examples:"
    echo "  $0 i-08fb48c8f6293cd97"
    echo "  $0 i-08fb48c8f6293cd97 eu-west-2 my-api-key s3 my-config-bucket"
    echo ""
    echo "Environment variables:"
    echo "  API_KEY - Your API key (will prompt if not provided)"
    echo "  S3_BUCKET - Your S3 bucket name (will prompt if not provided)"
    echo ""
    exit 1
fi

# Run main function
main "$@"
