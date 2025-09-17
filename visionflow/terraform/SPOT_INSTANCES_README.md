# AWS EC2 Spot Instances Configuration

This Terraform configuration supports both On-Demand and Spot Instances for cost optimization.

## What are Spot Instances?

Amazon EC2 Spot Instances let you take advantage of unused EC2 capacity in the AWS cloud and are available at up to a **90% discount** compared to On-Demand prices.

## Benefits

- **Cost Savings**: Up to 90% off On-Demand pricing
- **Same Performance**: Identical hardware and performance as On-Demand
- **Perfect for Fault-Tolerant Workloads**: Like our WAN2.1 service with S3 storage
- **No Bidding Required**: AWS sets the price automatically

## How to Enable Spot Instances

### Option 1: Update your `aws.tfvars` file

```hcl
# Enable Spot Instances
use_spot_instance = true
spot_max_price    = "0.10"  # Maximum price per hour (adjust as needed)
```

### Option 2: Use command line variables

```bash
terraform apply -var="use_spot_instance=true" -var="spot_max_price=0.10"
```

## Important Considerations

### Spot Interruption Handling
- AWS provides a **2-minute warning** before termination
- Your application should handle interruptions gracefully
- Data persistence is handled via S3 (already configured)

### Instance Availability
- Spot Instances may not always be available
- Consider using multiple instance types for better availability
- Monitor Spot Instance pricing in your region

### Best Practices
1. **Set appropriate max price**: Don't set too low, or you might not get instances
2. **Monitor costs**: Use CloudWatch to track spending
3. **Handle interruptions**: Ensure your application can restart cleanly
4. **Use multiple AZs**: For better availability

## Cost Comparison

| Instance Type | g5.xlarge (On-Demand) | g5.xlarge (Spot) | Savings |
|---------------|----------------------|------------------|---------|
| eu-west-1     | ~$1.20/hour          | ~$0.12-0.36/hour | 70-90%  |
| us-east-1     | ~$1.20/hour          | ~$0.12-0.36/hour | 70-90%  |

*Prices are approximate and vary by region and time*

## Monitoring

The configuration includes CloudWatch logging to monitor:
- Instance status
- Spot interruption warnings
- Application logs

## Switching Between Instance Types

You can easily switch between On-Demand and Spot Instances:

```bash
# Switch to Spot Instances
terraform apply -var="use_spot_instance=true"

# Switch back to On-Demand
terraform apply -var="use_spot_instance=false"
```

## Troubleshooting

### No Spot Capacity Available
- Try different instance types
- Check multiple availability zones
- Increase max price if needed

### Frequent Interruptions
- Consider using On-Demand for critical workloads
- Implement robust restart mechanisms
- Use multiple instance types in a fleet

## References

- [AWS Spot Instances Documentation](https://aws.amazon.com/ec2/spot/)
- [Spot Instance Best Practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html)
- [Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/)
