#!/bin/bash

# VisionFlow Kubernetes Secrets Setup Script
# This script helps you encode your secrets and update the secrets.yaml file

set -e

echo "🔐 VisionFlow Kubernetes Secrets Setup"
echo "======================================"
echo ""

# Check if required tools are available
if ! command -v base64 &> /dev/null; then
    echo "❌ Error: base64 command not found"
    exit 1
fi

# Function to encode a secret value
encode_secret() {
    local value="$1"
    echo -n "$value" | base64
}

# Function to prompt for a secret value
prompt_secret() {
    local name="$1"
    local description="$2"
    local default="$3"
    
    echo ""
    echo "📝 $description"
    if [ -n "$default" ]; then
        echo "   Default: $default"
    fi
    echo -n "   Enter value: "
    read -s value
    echo ""
    
    if [ -z "$value" ] && [ -n "$default" ]; then
        value="$default"
    fi
    
    echo "$value"
}

echo "This script will help you encode your secrets and update the secrets.yaml file."
echo "Make sure you have the following information ready:"
echo "  - Database username and password"
echo "  - Redis password"
echo "  - Google Cloud service account JSON file path"
echo "  - LangChain API key"
echo "  - OpenAI API key (if using OpenAI models)"
echo "  - HuggingFace token"
echo "  - Grafana username and password"
echo ""

read -p "Press Enter to continue..."

# Get secret values
echo ""
echo "🔑 Collecting secret values..."
echo "=============================="

DB_USER=$(prompt_secret "DB_USER" "Database username" "visionflow")
DB_PASSWORD=$(prompt_secret "DB_PASSWORD" "Database password" "")
REDIS_PASSWORD=$(prompt_secret "REDIS_PASSWORD" "Redis password" "")
GRAFANA_USER=$(prompt_secret "GRAFANA_USER" "Grafana username" "admin")
GRAFANA_PASSWORD=$(prompt_secret "GRAFANA_PASSWORD" "Grafana password" "")

echo ""
echo "🌐 API Keys and Tokens"
echo "======================"

LANGCHAIN_API_KEY=$(prompt_secret "LANGCHAIN_API_KEY" "LangChain API key" "")
OPENAI_API_KEY=$(prompt_secret "OPENAI_API_KEY" "OpenAI API key (optional)" "")
HUGGINGFACE_TOKEN=$(prompt_secret "HUGGINGFACE_TOKEN" "HuggingFace token" "")

echo ""
echo "☁️ Google Cloud Configuration"
echo "============================"

SERVICE_ACCOUNT_PATH=$(prompt_secret "SERVICE_ACCOUNT_PATH" "Path to Google Cloud service account JSON file" "")

# Validate service account file
if [ ! -f "$SERVICE_ACCOUNT_PATH" ]; then
    echo "❌ Error: Service account file not found: $SERVICE_ACCOUNT_PATH"
    exit 1
fi

# Encode all secrets
echo ""
echo "🔒 Encoding secrets..."
echo "===================="

DB_USER_ENCODED=$(encode_secret "$DB_USER")
DB_PASSWORD_ENCODED=$(encode_secret "$DB_PASSWORD")
REDIS_PASSWORD_ENCODED=$(encode_secret "$REDIS_PASSWORD")
GRAFANA_USER_ENCODED=$(encode_secret "$GRAFANA_USER")
GRAFANA_PASSWORD_ENCODED=$(encode_secret "$GRAFANA_PASSWORD")
LANGCHAIN_API_KEY_ENCODED=$(encode_secret "$LANGCHAIN_API_KEY")
OPENAI_API_KEY_ENCODED=$(encode_secret "$OPENAI_API_KEY")
HUGGINGFACE_TOKEN_ENCODED=$(encode_secret "$HUGGINGFACE_TOKEN")
SERVICE_ACCOUNT_ENCODED=$(encode_secret "$(cat "$SERVICE_ACCOUNT_PATH")")

# Create backup of original secrets.yaml
BACKUP_FILE="secrets.yaml.backup.$(date +%Y%m%d_%H%M%S)"
cp secrets.yaml "$BACKUP_FILE"
echo "✅ Created backup: $BACKUP_FILE"

# Update secrets.yaml
echo ""
echo "📝 Updating secrets.yaml..."
echo "==========================="

# Use sed to replace the placeholder values
sed -i.bak \
    -e "s|<base64_encoded_db_user>|$DB_USER_ENCODED|g" \
    -e "s|<base64_encoded_db_password>|$DB_PASSWORD_ENCODED|g" \
    -e "s|<base64_encoded_redis_password>|$REDIS_PASSWORD_ENCODED|g" \
    -e "s|<base64_encoded_grafana_user>|$GRAFANA_USER_ENCODED|g" \
    -e "s|<base64_encoded_grafana_password>|$GRAFANA_PASSWORD_ENCODED|g" \
    -e "s|<base64_encoded_langchain_api_key>|$LANGCHAIN_API_KEY_ENCODED|g" \
    -e "s|<base64_encoded_openai_api_key>|$OPENAI_API_KEY_ENCODED|g" \
    -e "s|<base64_encoded_huggingface_token>|$HUGGINGFACE_TOKEN_ENCODED|g" \
    -e "s|<base64_encoded_service_account_json>|$SERVICE_ACCOUNT_ENCODED|g" \
    secrets.yaml

# Remove temporary backup
rm secrets.yaml.bak

echo "✅ Successfully updated secrets.yaml"
echo ""
echo "🔍 Verification"
echo "=============="
echo "The following secrets have been encoded and updated:"
echo "  ✓ Database credentials"
echo "  ✓ Redis password"
echo "  ✓ Grafana credentials"
echo "  ✓ LangChain API key"
echo "  ✓ OpenAI API key"
echo "  ✓ HuggingFace token"
echo "  ✓ Google Cloud service account"
echo ""
echo "📋 Next Steps"
echo "============="
echo "1. Review the updated secrets.yaml file"
echo "2. Apply the secrets to your Kubernetes cluster:"
echo "   kubectl apply -f secrets.yaml"
echo "3. Verify the secrets are created:"
echo "   kubectl get secrets -n visionflow"
echo ""
echo "⚠️  Security Notes"
echo "================="
echo "- Keep your secrets.yaml file secure and never commit it to version control"
echo "- Consider using Kubernetes external secrets or a secrets management solution"
echo "- Rotate your secrets regularly"
echo "- The backup file contains your original secrets - delete it after verification"
echo ""
echo "🎉 Setup complete! Your VisionFlow deployment is ready for secrets."
