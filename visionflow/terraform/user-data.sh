#!/bin/bash

# WAN2.1 Service Setup Script for AWS EC2 (Advanced Version)
# This script sets up the WAN2.1 service using external configuration files

set -e

# Configuration
CONFIG_DIR="/opt/wan2-1-configs"
APP_DIR="/opt/wan2-1"
CONFIG_SOURCE="$${config_source:-s3}"  # s3, github, or local
S3_BUCKET="$${s3_bucket:-your-config-bucket}"
GITHUB_REPO="$${github_repo:-your-org/wan2-1-configs}"
GITHUB_BRANCH="$${github_branch:-main}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /var/log/wan2-1-setup.log
}

log "Starting WAN2.1 service setup..."

# Update system
log "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install required packages
log "Installing required packages..."
apt-get install -y curl wget git jq awscli

# Install Docker
log "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
log "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
log "Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Create configuration directory
mkdir -p $CONFIG_DIR

# Function to download configuration files
download_configs() {
    case $CONFIG_SOURCE in
        "s3")
            log "Downloading configuration files from S3..."
            aws s3 cp s3://$S3_BUCKET/configs/ $CONFIG_DIR/ --recursive
            ;;
        "github")
            log "Downloading configuration files from GitHub..."
            git clone -b $GITHUB_BRANCH https://github.com/$GITHUB_REPO.git $CONFIG_DIR
            ;;
        "local")
            log "Using local configuration files..."
            # Files should be pre-uploaded to the instance
            ;;
        *)
            log "Unknown config source: $CONFIG_SOURCE. Using fallback method..."
            create_fallback_configs
            ;;
    esac
}

# Fallback function to create configs if download fails
create_fallback_configs() {
    log "Creating fallback configuration files..."
    
    # Environment template
    cat > $CONFIG_DIR/wan2-1.env.template << 'EOF'
API_KEY=${api_key}
API_HOST=0.0.0.0
API_PORT=8002
LOG_LEVEL=info
MODEL_DEVICE=cuda
MODEL_CACHE_DIR=/app/models
EOF

    # Docker Compose
    cat > $CONFIG_DIR/docker-compose.yml << 'EOF'
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
EOF

    # Systemd service
    cat > $CONFIG_DIR/wan2-1.service << 'EOF'
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
EOF

    # CloudWatch agent config
    cat > $CONFIG_DIR/amazon-cloudwatch-agent.json << 'EOF'
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
EOF
}

# Download configuration files
download_configs

# Verify configuration files exist
if [ ! -f "$CONFIG_DIR/wan2-1.env.template" ] || [ ! -f "$CONFIG_DIR/docker-compose.yml" ]; then
    log "Configuration files not found, using fallback..."
    create_fallback_configs
fi

# Process environment file template with actual values
log "Processing environment configuration..."
if command -v envsubst >/dev/null 2>&1; then
    envsubst < $CONFIG_DIR/wan2-1.env.template > .env
else
    # Fallback if envsubst is not available
    sed "s/\${api_key}/$api_key/g" $CONFIG_DIR/wan2-1.env.template > .env
fi

# Copy configuration files to application directory
log "Copying configuration files..."
cp $CONFIG_DIR/docker-compose.yml .
cp $CONFIG_DIR/wan2-1.service /etc/systemd/system/wan2-1.service

# Create required directories
log "Creating application directories..."
mkdir -p models outputs logs

# Set proper permissions
log "Setting permissions..."
chown -R ubuntu:ubuntu $APP_DIR
chmod +x /usr/local/bin/docker-compose

# Enable and start the service
log "Enabling systemd service..."
systemctl daemon-reload
systemctl enable wan2-1.service

# Install CloudWatch agent for logging
log "Installing CloudWatch agent..."
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure CloudWatch agent
log "Configuring CloudWatch agent..."
cp $CONFIG_DIR/amazon-cloudwatch-agent.json /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Start CloudWatch agent
log "Starting CloudWatch agent..."
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

# Start the WAN2.1 service
log "Starting WAN2.1 service..."
systemctl start wan2-1.service

# Verify service is running
sleep 10
if systemctl is-active --quiet wan2-1.service; then
    log "WAN2.1 service started successfully"
else
    log "WARNING: WAN2.1 service may not have started properly"
    systemctl status wan2-1.service
fi

# Log completion
log "WAN2.1 setup completed successfully at $(date)"
