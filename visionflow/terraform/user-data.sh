#!/bin/bash

# WAN2.1 Service Setup Script for AWS EC2
# This script sets up the WAN2.1 service on a Deep Learning AMI

set -e

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

# Create environment file
cat > .env << EOF
API_KEY=${api_key}
API_HOST=0.0.0.0
API_PORT=8002
LOG_LEVEL=info
MODEL_DEVICE=cuda
MODEL_CACHE_DIR=/app/models
EOF

# Create Docker Compose file
cat > docker-compose.yml << 'EOF'
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

# Create directories
mkdir -p models outputs

# Set permissions
chown -R ubuntu:ubuntu /opt/wan2-1

# Create systemd service for WAN2.1
cat > /etc/systemd/system/wan2-1.service << 'EOF'
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

# Enable and start the service
systemctl daemon-reload
systemctl enable wan2-1.service

# Install CloudWatch agent for logging
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
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

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

# Log completion
echo "WAN2.1 setup completed at $(date)" >> /var/log/wan2-1-setup.log
