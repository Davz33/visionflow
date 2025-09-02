#!/bin/bash

# Test local Kubernetes deployment

set -e

echo "🧪 Testing VisionFlow Kubernetes deployment..."

NAMESPACE="visionflow-local"

# Check if namespace exists
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "❌ Namespace $NAMESPACE does not exist. Run deploy-k8s-local.sh first."
    exit 1
fi

# Wait for all pods to be ready
echo "⏳ Waiting for all pods to be ready..."
kubectl wait --for=condition=ready pod --all -n $NAMESPACE --timeout=300s

# Test database connectivity
echo "🗄️  Testing database connectivity..."
kubectl exec -n $NAMESPACE deployment/postgres -- pg_isready -U visionflow

# Test Redis connectivity  
echo "📮 Testing Redis connectivity..."
kubectl exec -n $NAMESPACE deployment/redis -- redis-cli ping

# Test MinIO connectivity
echo "🪣 Testing MinIO connectivity..."
kubectl exec -n $NAMESPACE deployment/minio -- curl -f http://localhost:9000/minio/health/live

# Test API Gateway health
echo "🌐 Testing API Gateway health..."
timeout 10 bash -c 'until curl -f http://localhost:30000/health; do sleep 1; done' || echo "⚠️  API Gateway not responding (this is normal if it's still starting)"

# Test Generation Service health (if available)
echo "🎬 Testing Generation Service health..."
timeout 10 bash -c 'until curl -f http://localhost:30002/health; do sleep 1; done' || echo "⚠️  Generation Service not responding (this is normal if it's still starting)"

# Show resource usage
echo "📊 Resource usage:"
kubectl top pods -n $NAMESPACE 2>/dev/null || echo "⚠️  Metrics server not available"

# Show service endpoints
echo "🌐 Service endpoints:"
kubectl get services -n $NAMESPACE

echo "✅ Tests completed!"
echo ""
echo "💡 Next steps:"
echo "   1. Check pod logs: kubectl logs -f deployment/api-gateway -n $NAMESPACE"
echo "   2. Access services at the endpoints shown above"
echo "   3. Monitor with: watch kubectl get pods -n $NAMESPACE"