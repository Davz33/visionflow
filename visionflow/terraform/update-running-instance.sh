#!/bin/bash

# Update Running Instance Script
# This script updates a running EC2 instance with new configuration

set -e

# Configuration
INSTANCE_ID="${1:-i-08fb48c8f6293cd97}"
REGION="${2:-eu-west-2}"
CONFIG_SOURCE="${3:-s3}"
S3_BUCKET="${4:-your-config-bucket}"

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

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if instance is running
check_instance_status() {
    log "Checking instance status..."
    STATUS=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)
    
    if [ "$STATUS" != "running" ]; then
        error "Instance $INSTANCE_ID is not running (status: $STATUS)"
        exit 1
    fi
    
    log "Instance $INSTANCE_ID is running"
}

# Get instance public IP
get_instance_ip() {
    log "Getting instance public IP..."
    PUBLIC_IP=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    
    if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
        error "Could not get public IP for instance $INSTANCE_ID"
        exit 1
    fi
    
    log "Instance public IP: $PUBLIC_IP"
}

# Create update script for the instance
create_update_script() {
    log "Creating update script for instance..."
    
    cat > /tmp/instance-update.sh << 'EOF'
#!/bin/bash

# Instance Update Script
set -e

# Configuration
CONFIG_DIR="/opt/wan2-1-configs"
APP_DIR="/opt/wan2-1"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/wan2-1-update.log
}

log "Starting WAN2.1 configuration update..."

# Stop the service
log "Stopping WAN2.1 service..."
systemctl stop wan2-1.service || true
cd $APP_DIR
docker-compose down || true

# Update configuration files
log "Updating configuration files..."

# Download new configs from S3
if [ "$CONFIG_SOURCE" = "s3" ] && [ -n "$S3_BUCKET" ]; then
    log "Downloading configuration files from S3..."
    aws s3 cp s3://$S3_BUCKET/configs/ $CONFIG_DIR/ --recursive
fi

# Verify configuration files exist
if [ ! -f "$CONFIG_DIR/wan2-1.env.template" ] || [ ! -f "$CONFIG_DIR/docker-compose.yml" ]; then
    log "Configuration files not found, creating fallback..."
    mkdir -p $CONFIG_DIR
    
    # Create fallback configs (same as in user-data.sh)
    cat > $CONFIG_DIR/wan2-1.env.template << 'ENVEOF'
API_KEY=${api_key}
API_HOST=0.0.0.0
API_PORT=8002
LOG_LEVEL=info
MODEL_DEVICE=cuda
MODEL_CACHE_DIR=/app/models
ENVEOF

    cat > $CONFIG_DIR/docker-compose.yml << 'COMPOSEEOF'
version: '3.8'

services:
  wan2-1:
    image: wan2-1:latest
    container_name: wan2-1
    restart: unless-stopped
    ports:
      - "8002:8002"
    environment:
      - API_KEY=${api_key}
      - API_HOST=0.0.0.0
      - API_PORT=8002
      - LOG_LEVEL=info
      - MODEL_DEVICE=cuda
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
COMPOSEEOF

    cat > $CONFIG_DIR/wan2-1.service << 'SERVICEEOF'
[Unit]
Description=WAN2.1 Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/wan2-1
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target
SERVICEEOF

    cat > $CONFIG_DIR/amazon-cloudwatch-agent.json << 'CLOUDWATCHEOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/opt/wan2-1/logs/*.log",
            "log_group_name": "/aws/ec2/wan2-1",
            "log_stream_name": "{instance_id}/wan2-1.log"
          }
        ]
      }
    }
  }
}
CLOUDWATCHEOF
fi

# Process environment file template
log "Processing environment configuration..."
if command -v envsubst >/dev/null 2>&1; then
    envsubst < $CONFIG_DIR/wan2-1.env.template > $APP_DIR/.env
else
    sed "s/\${api_key}/$api_key/g" $CONFIG_DIR/wan2-1.env.template > $APP_DIR/.env
fi

# Copy configuration files
log "Copying configuration files..."
cp $CONFIG_DIR/docker-compose.yml $APP_DIR/
cp $CONFIG_DIR/wan2-1.service /etc/systemd/system/wan2-1.service

# Create required directories
log "Creating application directories..."
mkdir -p $APP_DIR/models $APP_DIR/outputs $APP_DIR/logs

# Set proper permissions
log "Setting permissions..."
chown -R ubuntu:ubuntu $APP_DIR
chmod +x /usr/local/bin/docker-compose

# Reload systemd and restart service
log "Reloading systemd and restarting service..."
systemctl daemon-reload
systemctl enable wan2-1.service
systemctl start wan2-1.service

# Update CloudWatch agent config
log "Updating CloudWatch agent configuration..."
cp $CONFIG_DIR/amazon-cloudwatch-agent.json /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

# Verify service is running
sleep 10
if systemctl is-active --quiet wan2-1.service; then
    log "WAN2.1 service updated and started successfully"
else
    log "WARNING: WAN2.1 service may not have started properly"
    systemctl status wan2-1.service
fi

log "Configuration update completed successfully"
EOF

    chmod +x /tmp/instance-update.sh
}

# Upload and execute the update script
execute_update() {
    log "Uploading update script to instance..."
    
    # Copy script to instance
    scp -i visionflow-key.pem -o StrictHostKeyChecking=no /tmp/instance-update.sh ubuntu@$PUBLIC_IP:/tmp/
    
    # Execute the update script on the instance
    log "Executing update script on instance..."
    ssh -i visionflow-key.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << EOF
export api_key='$API_KEY'
export config_source='$CONFIG_SOURCE'
export s3_bucket='$S3_BUCKET'
/tmp/instance-update.sh
EOF
}

# Main execution
main() {
    log "Starting instance update process..."
    
    # Check prerequisites
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v ssh &> /dev/null; then
        error "SSH is not installed or not in PATH"
        exit 1
    fi
    
    if [ ! -f "visionflow-key.pem" ]; then
        error "SSH key file 'visionflow-key.pem' not found in current directory"
        exit 1
    fi
    
    # Set API key from environment or prompt
    if [ -z "$API_KEY" ]; then
        read -p "Enter your API key: " API_KEY
        if [ -z "$API_KEY" ]; then
            error "API key is required"
            exit 1
        fi
    fi
    
    # Execute steps
    check_instance_status
    get_instance_ip
    create_update_script
    execute_update
    
    log "Instance update completed successfully!"
    log "You can check the service status with: ssh -i visionflow-key.pem ubuntu@$PUBLIC_IP 'systemctl status wan2-1.service'"
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <instance-id> [region] [config-source] [s3-bucket]"
    echo ""
    echo "Examples:"
    echo "  $0 i-08fb48c8f6293cd97"
    echo "  $0 i-08fb48c8f6293cd97 eu-west-2 s3 my-config-bucket"
    echo ""
    echo "Environment variables:"
    echo "  API_KEY - Your API key (will prompt if not set)"
    echo ""
    exit 1
fi

# Run main function
main "$@"
