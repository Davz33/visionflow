# Minimal AWS EC2 + GPU Configuration

This is a minimal, focused Terraform configuration that only handles the bare AWS EC2 instance setup with GPU support for VisionFlow WAN2.1.

## What This Configuration Includes

- ✅ **EC2 Instance** with GPU support (g5.xlarge)
- ✅ **Security Group** with SSH (22) and WAN2.1 API (8002) access
- ✅ **Elastic IP** for consistent public access
- ✅ **IAM Role** for CloudWatch logging
- ✅ **User Data Script** for automatic WAN2.1 setup
- ✅ **CloudWatch Logs** integration

## What This Configuration Does NOT Include

- ❌ GCP infrastructure (moved to separate configs)
- ❌ Complex monitoring and alerting
- ❌ Secret management
- ❌ DNS configuration
- ❌ Database setup
- ❌ Load balancers
- ❌ Auto-scaling groups

## Files

- `aws-minimal.tf` - Main Terraform configuration
- `aws-variables.tf` - Variable definitions
- `aws.tfvars` - Your specific values
- `user-data.sh` - EC2 initialization script

## Quick Start

1. **Update your values in `*.tfvars`(s)**:
   ```bash
   # Edit the file with your specific values
   vim aws.tfvars
   ```

2. **Initialize Terraform**:
   ```bash
   terraform init
   ```

3. **Plan the deployment using (all or some) var-files**:
   ```bash
   terraform plan -var-file="*.tfvars" -var-file="*.tfvars" -var-file="*.tfvars" 
   ```

4. **Deploy the infrastructure**:
   ```bash
   terraform apply -var-file="aws.tfvars"
   ```

5. **Verify aws cli is in the correct region**
   ```bash
   aws configure get region
   ````
And change to the region where the instance is deployed, if necessary

6. **Create and associate a key-pair**
   ```bash
   # Create new key pair and set permissions
   aws ec2 create-key-pair --region <region> --key-name <key-pair-name> --query 'KeyMaterial' --output text > <key-pair-name>.pem
   chmod 600 <key-pair-name>.pem
   # Add to .gitignore, in case you're tracking you're project
   echo "<key-pair-name>.pem" >> .gitignore
   git add .gitignore
   # Extract public key from private and send it to the instance using EC2 instance connect
   ssh-keygen -y -f <key-pair-name>.pem > <key-pair-name>.pub
   aws ec2-instance-connect send-ssh-public-key --region <region> --instance-id <ec2-instance-id> --instance-os-user <username> --ssh-public-key file://<key-pair-name>.pub
   ```

7. **Connect to your instance**:
   ```bash
   # Use the SSH command from the output
   # The standard username is ubuntu on ubuntu ec2 instances
   ssh -i <key-pair-name>.pem <username>@<elastic-ip>
   ```

## Outputs

After deployment, you'll get:
- `wan2_1_url` - The service URL (http://<ip>:8002)
- `ssh_command` - Ready-to-use SSH command
- `instance_id` - EC2 instance ID
- `elastic_ip` - Public IP address

## Next Steps

This minimal configuration gives you a working AWS EC2 instance with GPU support. You can then:

1. **Deploy your WAN2.1 application** to the instance
2. **Set up monitoring** using separate Terraform configurations
3. **Configure load balancing** if needed
4. **Add auto-scaling** for production use

## Modular Architecture

This approach allows you to:
- **Separate concerns** - AWS, GCP, monitoring, etc. in different configs
- **Deploy independently** - Each service can be deployed separately
- **Maintain easily** - Smaller, focused configurations
- **Scale teams** - Different teams can own different configurations




