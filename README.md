# VisionFlow: Advanced Video Generation & Evaluation Platform

VisionFlow is a production-ready platform for AI-powered video generation and automated quality assessment. Built with modern microservices architecture, it provides enterprise-grade video generation capabilities with comprehensive evaluation and monitoring.

## 🚀 Core Features

### Video Generation
- **WAN 2.1 Integration**: State-of-the-art video generation using multimodal AI models
- **Multi-Quality Output**: Support for low, medium, high, and ultra quality settings
- **Batch Processing**: Efficient handling of multiple video generation requests
- **Progress Tracking**: Real-time monitoring of generation progress

### Automated Evaluation System
- **6-Dimensional Quality Assessment**: Technical, Content, Aesthetic, UX, Performance, and Compliance metrics
- **Multi-Agent Orchestration**: Intelligent routing and processing using enhanced multi-agent systems
- **Statistical Analysis**: Comprehensive scoring with confidence intervals
- **Continuous Learning**: Adaptive evaluation strategies based on performance data

### Production Infrastructure
- **Kubernetes Deployment**: Scalable, production-ready deployment
- **Monitoring & Observability**: Prometheus metrics, Grafana dashboards, and comprehensive logging
- **Distributed Processing**: Celery-based task queue with horizontal scaling
- **Persistent Storage**: PostgreSQL for metadata, Redis for caching, MinIO for media storage

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Video Generation│    │   Evaluation    │
│   (FastAPI)     │◄──►│   Service        │◄──►│   Orchestrator  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Task Queue     │    │   Storage Layer │
│   (Prometheus)  │    │   (Celery)       │    │   (PostgreSQL)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔄 System Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   AI Services   │
│   (Flutter)     │◄──►│   (FastAPI)      │◄──►│   (WAN 2.1)     │
│                 │    │                  │    │   (Gemini Pro)  │
│ • Dashboard     │    │ • Request        │    │ • Video Gen     │
│ • Monitoring    │    │   Validation     │    │ • Evaluation    │
│ • Results View  │    │ • Job Queue      │    │ • Analysis      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Orchestration │    │   Task Queue     │    │   Storage       │
│   (Multi-Agent) │    │   (Celery)       │    │   (PostgreSQL)  │
│                 │    │                  │    │   (MinIO)       │
│ • Job Routing   │    │ • Async          │    │ • Metadata      │
│ • Load Balance  │    │   Processing     │    │ • Media Files   │
│ • Failover      │    │ • Worker Pool    │    │ • Results       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Implementation Status

### ✅ **Fully Implemented & Production Ready**

#### **Core Infrastructure**
- **API Gateway**: FastAPI-based REST API with comprehensive endpoints
- **Database Layer**: PostgreSQL with SQLAlchemy ORM and migrations
- **Caching System**: Redis integration for performance optimization
- **Task Queue**: Celery-based distributed task processing
- **Containerization**: Docker and Docker Compose for all services
- **Kubernetes Deployment**: Complete K8s manifests and Helm charts

#### **Video Generation Service**
- **WAN 2.1 Integration**: Full integration with state-of-the-art video model
- **Quality Settings**: Support for multiple quality levels (low, medium, high, ultra)
- **Batch Processing**: Efficient handling of multiple generation requests
- **Progress Tracking**: Real-time job status and progress monitoring
- **Resource Management**: GPU memory optimization and CPU offloading

#### **Evaluation System**
- **Multi-Agent Orchestration**: Intelligent routing and processing
- **Quality Metrics**: 6-dimensional assessment framework
- **Statistical Analysis**: Confidence intervals and scoring algorithms
- **Benchmark Integration**: Industry-standard evaluation criteria
- **Continuous Learning**: Adaptive evaluation strategies

#### **Monitoring & Observability**
- **Metrics Collection**: Prometheus integration with custom metrics
- **Dashboard**: Grafana dashboards for real-time monitoring
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Comprehensive health monitoring endpoints
- **Alerting**: Configurable alerting rules and notifications

### 🔄 **Work in Progress / Needs Improvement**

#### **Fine-Tuning System**
- **Status**: Partially implemented, needs completion
- **Current State**: Basic framework exists, training pipeline incomplete
- **Priority**: Medium - requires ML pipeline completion
- **Estimated Completion**: 2-3 weeks

#### **Advanced Analytics**
- **Status**: Basic implementation, needs enhancement
- **Current State**: Core metrics collection working
- **Priority**: Low - functional but could be enhanced
- **Estimated Completion**: 1-2 weeks

#### **Performance Optimization**
- **Status**: Ongoing optimization
- **Current State**: Good baseline performance
- **Priority**: Medium - continuous improvement
- **Estimated Completion**: Ongoing

### 🚧 **Planned / Future Development**

#### **Multi-Model Support**
- **Status**: Planned for next sprint
- **Description**: Support for additional video generation models
- **Priority**: High - increases platform flexibility
- **Estimated Start**: Next sprint

#### **Advanced Workflow Engine**
- **Status**: Design phase
- **Description**: Complex workflow orchestration with conditional logic
- **Priority**: Medium - enhances automation capabilities
- **Estimated Start**: Q4 2025

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, FastAPI, Celery
- **AI Models**: WAN 2.1, Gemini Pro Vision, CLIP
- **Databases**: PostgreSQL, Redis
- **Storage**: MinIO, Google Cloud Storage
- **Orchestration**: Kubernetes, Docker
- **Monitoring**: Prometheus, Grafana
- **MLOps**: MLflow, GCP Vertex AI

## 📋 Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Kubernetes cluster (for production deployment)
- Google Cloud Platform account (for GCP services)
- NVIDIA GPU with CUDA support (for local video generation)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Davz33/visionflow
   cd visionflow
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   # For development
   pip install -r requirements_ml_evaluation.txt
   ```

4. **Start services with Docker Compose**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

5. **Access the API**
   ```bash
   curl http://localhost:8000/health
   ```

### Production Deployment

#### **Kubernetes Deployment**

1. **Configure Kubernetes secrets**
   ```bash
   # Use the setup script to configure secrets
   cd visionflow/k8s/
   ./setup-secrets.sh
   
   # Or manually apply secrets
   kubectl apply -f secrets.yaml
   kubectl apply -f configmap.yaml
   ```

2. **Deploy the application**
   ```bash
   # Deploy all components
   kubectl apply -f k8s/
   
   # Verify deployment
   kubectl get pods -n visionflow
   kubectl get services -n visionflow
   ```

3. **Access the services**
   ```bash
   # Port forward for local access
   kubectl port-forward -n visionflow svc/visionflow-api 8000:8000
   kubectl port-forward -n visionflow svc/grafana 3000:3000
   ```

#### **Docker Deployment**

1. **Build and run with Docker Compose**
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

2. **Or use individual Dockerfiles**
   ```bash
   # Build API service
   docker build -f visionflow/docker/Dockerfile.api -t visionflow-api .
   
   # Run with proper environment
   docker run -d --name visionflow-api \
     -p 8000:8000 \
     --env-file .env.production \
     visionflow-api
   ```

## 🔧 Configuration

### Environment Variables

Key configuration options are available through environment variables:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=visionflow
DB_USER=visionflow
DB_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Storage Configuration
STORAGE_ENDPOINT=localhost:9000
STORAGE_ACCESS_KEY=your_access_key
STORAGE_SECRET_KEY=your_secret_key

# Model Configuration
WAN_MODEL_PATH=multimodalart/wan2-1-fast
MODEL_DEVICE=auto
MAX_MEMORY_GB=8

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

### Kubernetes Configuration

The platform uses Kubernetes ConfigMaps and Secrets for configuration management:

- **ConfigMap**: Non-sensitive configuration (hosts, ports, feature flags)
- **Secrets**: Sensitive data (passwords, API keys, service account credentials)

## 📊 API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /generate/video` - Video generation request
- `POST /evaluate/video` - Video evaluation request
- `GET /jobs/{job_id}` - Job status and results

### Video Generation

```bash
curl -X POST "http://localhost:8000/generate/video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "quality": "high",
    "duration": 10,
    "resolution": "512x512"
  }'
```

### Video Evaluation

```bash
curl -X POST "http://localhost:8000/evaluate/video" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "evaluation_criteria": ["technical", "content", "aesthetic"]
  }'
```

## 📈 Monitoring & Observability

### Metrics Dashboard

Access the Grafana dashboard at `http://localhost:3000` (default: admin/admin):

- **Video Generation Metrics**: Success rates, processing times, queue lengths
- **Evaluation Metrics**: Quality scores, confidence intervals, processing efficiency
- **System Metrics**: Resource utilization, API performance, error rates

### Key Metrics

- `video_generation_jobs_total` - Total video generation requests
- `video_evaluation_duration_seconds` - Evaluation processing time
- `job_queue_length` - Current queue length
- `success_rate` - Success rate by quality level

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify PostgreSQL is running and accessible
   - Check database credentials in environment variables
   - Ensure database exists and migrations are applied

2. **GPU Memory Issues**
   - Reduce `MAX_MEMORY_GB` in configuration
   - Enable CPU offloading with `ENABLE_CPU_OFFLOAD=true`
   - Use smaller model variants

3. **Kubernetes Deployment Issues**
   - Check pod logs: `kubectl logs -n visionflow <pod-name>`
   - Verify secrets and configmaps are properly applied
   - Check resource limits and requests

### Logs

Logs are available in structured JSON format:

```bash
# View API logs
kubectl logs -n visionflow -l app=visionflow-api

# View worker logs
kubectl logs -n visionflow -l app=visionflow-worker
```

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install -r requirements_ml_evaluation.txt

# Run tests
pytest visionflow/tests/

# Run with coverage
pytest --cov=visionflow visionflow/tests/
```

### Test Datasets

Sample evaluation datasets are available in `visionflow/evaluation_datasets/`:

- **Test Videos**: Small sample videos for testing
- **Evaluation Criteria**: Comprehensive quality assessment criteria
- **Benchmark Data**: Industry-standard evaluation benchmarks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For technical support or questions:

- Check the troubleshooting section above
- Review the logs for error details
- Open an issue in the repository
- Contact davide_vitiello@outlook.com

---

**VisionFlow** - Transforming video creation through intelligent AI orchestration and automated quality assessment.
