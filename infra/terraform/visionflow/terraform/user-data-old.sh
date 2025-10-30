#!/bin/bash

# WAN2.1 Service Setup Script for AWS EC2
# This script sets up the WAN2.1 service on a Deep Learning AMI

set -e

# Configuration files directory
CONFIG_DIR="/opt/wan2-1-configs"

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/wan2-1
cd /opt/wan2-1

# Create configuration directory and download config files
mkdir -p $CONFIG_DIR

# Download configuration files from S3 or GitHub (replace with your actual source)
# For now, we'll create them locally, but in production you'd download from a secure location
# curl -o $CONFIG_DIR/wan2-1.env.template https://your-bucket.s3.amazonaws.com/configs/wan2-1.env.template
# curl -o $CONFIG_DIR/docker-compose.yml https://your-bucket.s3.amazonaws.com/configs/docker-compose.yml
# curl -o $CONFIG_DIR/wan2-1.service https://your-bucket.s3.amazonaws.com/configs/wan2-1.service
# curl -o $CONFIG_DIR/amazon-cloudwatch-agent.json https://your-bucket.s3.amazonaws.com/configs/amazon-cloudwatch-agent.json

# For this example, we'll create the config files inline (in production, use the download method above)
cat > $CONFIG_DIR/wan2-1.env.template << 'EOF'
API_KEY=${api_key}
API_HOST=0.0.0.0
API_PORT=8002
LOG_LEVEL=info
MODEL_DEVICE=cuda
MODEL_CACHE_DIR=/app/models
EOF

cat > $CONFIG_DIR/docker-compose.yml << 'EOF'
version: '3.8'

services:
  wan2-1:
    image: wan2-1:latest  # Replace with your actual image
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

# Process environment file template with actual values
envsubst < $CONFIG_DIR/wan2-1.env.template > .env

# Copy configuration files to application directory
cp $CONFIG_DIR/docker-compose.yml .
cp $CONFIG_DIR/wan2-1.service /etc/systemd/system/wan2-1.service

# Create directories
mkdir -p models outputs

# Set permissions
chown -R ubuntu:ubuntu /opt/wan2-1

# Enable and start the service
systemctl daemon-reload
systemctl enable wan2-1.service

# Install CloudWatch agent for logging
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure CloudWatch agent using the config file
cp $CONFIG_DIR/amazon-cloudwatch-agent.json /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

# Log completion
echo "WAN2.1 setup completed at $(date)" >> /var/log/wan2-1-setup.log