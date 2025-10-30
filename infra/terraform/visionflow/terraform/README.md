# VisionFlow Terraform Configuration

This directory contains Terraform configuration files for deploying VisionFlow infrastructure on Google Cloud Platform (GCP) with optional AWS integration for WAN2.1 remote service.

## File Structure

### Core Configuration Files
- `main.tf` - Main Terraform configuration
- `terraform.tfvars` - Main variables file (GCP configuration)
- `variables.tf` - Variable definitions
- `outputs.tf` - Output definitions

### AWS Configuration Files
- `aws.tfvars` - AWS-specific configuration (credentials and settings)
- `aws-secrets.tfvars` - AWS secrets for Google Secret Manager

### Environment Files
- `terraform.tfvars.example` - Example configuration template
- `aws.tfvars.example` - Example AWS configuration template

## Security Notes

⚠️ **IMPORTANT SECURITY CONSIDERATIONS:**

1. **Never commit actual credentials** to version control
2. **Use environment variables** or secure secret management
3. **Rotate credentials regularly**
4. **Use least privilege access** for AWS and GCP services
5. **Enable audit logging** for all cloud resources

## Usage

### Basic GCP Deployment
```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="terraform.tfvars"

# Apply configuration
terraform apply -var-file="terraform.tfvars"
```

### With AWS Integration
```bash
# Plan with AWS configuration
terraform plan -var-file="terraform.tfvars" -var-file="aws.tfvars"

# Apply with AWS configuration
terraform apply -var-file="terraform.tfvars" -var-file="aws.tfvars"
```

### With AWS Secrets
```bash
# Plan with all configurations
terraform plan -var-file="terraform.tfvars" -var-file="aws.tfvars" -var-file="aws-secrets.tfvars"

# Apply with all configurations
terraform apply -var-file="terraform.tfvars" -var-file="aws.tfvars" -var-file="aws-secrets.tfvars"
```

## Configuration

### GCP Configuration (terraform.tfvars)
- Project ID and region settings
- GKE cluster configuration
- Database and storage settings
- Monitoring and notification channels

### AWS Configuration (aws.tfvars)
- AWS credentials and region
- EC2 instance configuration
- Network settings (security groups, subnets)
- AMI and instance type settings

### AWS Secrets (aws-secrets.tfvars)
- AWS credentials for Google Secret Manager
- SSH key pair information
- Service-specific labels

## Environment Variables

For enhanced security, you can use environment variables instead of hardcoded values:

```bash
export TF_VAR_aws_access_key_id="your-access-key"
export TF_VAR_aws_secret_access_key="your-secret-key"
export TF_VAR_project_id="your-gcp-project-id"
```

## Best Practices

1. **Use separate files** for different environments (dev, staging, prod)
2. **Implement proper IAM roles** with minimal required permissions
3. **Enable CloudTrail** for AWS and Cloud Audit Logs for GCP
4. **Use Terraform state encryption** and remote state storage
5. **Regular security reviews** of all configuration files
6. **Implement proper backup strategies** for state files

## Troubleshooting

### Common Issues
1. **Credential errors** - Verify AWS and GCP credentials
2. **Permission errors** - Check IAM roles and service accounts
3. **Resource conflicts** - Ensure unique resource names
4. **Network issues** - Verify security group and firewall rules

### Getting Help
- Check Terraform logs: `terraform plan -detailed-exitcode`
- Verify GCP resources in Console
- Check AWS resources in EC2 Console
- Review CloudWatch and Cloud Logging for errors
