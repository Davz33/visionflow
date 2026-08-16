# VisionFlow: Advanced Video Generation & Evaluation Platform

VisionFlow is a production-ready platform for AI-powered video generation and automated quality assessment. Built with modern microservices architecture, it provides enterprise-grade video generation capabilities with comprehensive evaluation and monitoring.

## 🚀 Core Features

### Video Generation
- **Wan2.2 Integration**: Open-weight T2V / I2V / TI2V / S2V / Animate (plus Animate-2 distilled)
- **Size axis**: TI2V-5B (consumer 720P@24fps) vs T2V/I2V A14B MoE (~80GB)
- **Prompt-extend ablation**: official `--use_prompt_extend` on/off
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
│   (Flutter)     │◄──►│   (FastAPI)      │◄──►│   (Wan2.2)      │
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
- **Wan2.2 Integration**: Full integration with the official open-weight zoo (pinned 2026-08-15)
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
- **AI Models**: Wan2.2 (TI2V-5B, T2V-A14B, I2V-A14B, S2V-14B, Animate-14B / Animate-2), Gemini Pro Vision, CLIP
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

### Optional Dependencies

VisionFlow uses organized optional dependencies managed through `pyproject.toml`. Install additional features as needed:

```bash
# For ML evaluation features
pip install visionflow[ml_evaluation]

# For production deployment
pip install visionflow[production]

# For LLaVA model support
pip install visionflow[llava]

# For monitoring (Grafana/Prometheus)
pip install visionflow[monitoring]

# For Google Cloud integration
pip install visionflow[gcp]

# For development and testing
pip install visionflow[dev]

# For all optional dependencies
pip install visionflow[all]
```

### Using Pixi for Environment Management

VisionFlow includes native **pixi** support for flexible environment management. Pixi provides isolated, reproducible environments with native feature composition.

**Install Pixi:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Available Pixi Environments:**

```bash
# Development environment (default) with testing tools
pixi run --environment default test

# Production-only dependencies (no dev tools)
pixi shell --environment prod

# Production dependencies + test tools (same versions)
pixi run --environment prod-test test

# ML/AI evaluation environment
pixi run --environment ml python scripts/evaluate_large_scale_dataset.py

# LLaVA model support environment
pixi run --environment llava python

# Monitoring tools environment
pixi run --environment monitoring python

# GCP integration environment
pixi run --environment gcp python

# Full environment with all optional features
pixi run --environment full test
```

**Available Tasks in Environments:**

```bash
# Run tests
pixi run test

# Run linting
pixi run lint

# Format code
pixi run format

# Build documentation
pixi run --environment docs build-docs
```

**Install Dependencies for Specific Environment:**

```bash
# Install default environment
pixi install

# Install specific environment
pixi install --environment prod

# Install all environments
pixi install --all
```

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
   docker-compose -f infra/docker/docker-compose.dev.yml up -d
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
   cd infra/k8s/visionflow/k8s/
   ./setup-secrets.sh
   
   # Or manually apply secrets
   kubectl apply -f secrets.yaml
   kubectl apply -f configmap.yaml
   ```

2. **Deploy the application**
   ```bash
   # Deploy all components
   kubectl apply -f infra/k8s/visionflow/k8s/
   
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
   docker-compose -f infra/docker/visionflow/docker-compose.production.yml up -d
   ```

2. **Or use individual Dockerfiles**
   ```bash
   # Build API service
   docker build -f infra/docker/visionflow/Dockerfile.api -t visionflow-api .
   
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
WAN_MODEL_PATH=Wan-AI/Wan2.2-TI2V-5B-Diffusers
WAN_TASK=ti2v-5B
WAN_USE_PROMPT_EXTEND=false
WAN_SAMPLE_SOLVER=unipc
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

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /generate/video` - Video generation request
- `POST /evaluate/video` - Video evaluation request
- `GET /jobs/{job_id}` - Job status and results
- `GET /metadata/video` - Retrieve metadata for a video (e.g., duration, resolution, codec, from storage/DB)
- `GET /analytics/summary` - High-level analytics summary across jobs/generations (aggregated metrics)
- `GET /generate/video/status/{generation_id}` - Get current status and progress for a specific generation job
- `GET /generate/video/history` - List historical generation jobs with basic metadata (timestamps, status)
- `GET /generate/video/{generation_id}` - Retrieve details and results (URLs, metadata) for a completed generation
- `GET /models/status` - Return model availability and health (which models are loaded, device usage)

### Video Generation

```bash
curl -X POST "http://localhost:8000/generate/video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "quality": "high",
    "duration": 10,
    "resolution": "1280x704",
    "task": "ti2v-5B",
    "use_prompt_extend": false,
    "sample_solver": "unipc",
    "num_inference_steps": 50
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
pytest --cov=src.visionflow visionflow/tests/
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

## Wan2.2 eval harness (open weights, 2026-08-15)

Default generate path is **Wan2.2 TI2V-5B** (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`), not Wan2.1.
Upstream generate.py lives in [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) at git SHA `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` (2026-03-17). Wan 2.5 / 2.6 / 3.0 are Alibaba Cloud API products without official public weights; this repo does not wrap them.

```bash
# Print pinned zoo (HF ids, tasks, sample_steps)
python -m visionflow.eval.cli zoo

# Expand the benchmark matrix (task x size x prompt-extend x scheduler)
python -m visionflow.eval.cli matrix --suite config/wan_eval_suite.yaml

# Dry-run a generate.py-shaped request (no GPU)
python -m visionflow.eval.cli plan --task ti2v-5B --size 1280*704 --prompt "a cat" --base_seed 42
```

Eval axes in `config/wan_eval_suite.yaml`: TI2V / T2V / I2V / S2V / Animate / Animate-2, 5B vs A14B, prompt-extend on/off, 480P/720P (and TI2V 1280x704 @ 24fps), UniPC vs DPM++ vs few-step, plus official Animate-2 distilled 4-step weights. Result JSON records `model_revision`, `upstream_git_sha`, `hf_revision`, and `seed`.

See [`docs/COLAB_WAN22_BENCHMARK_REPORT.md`](docs/COLAB_WAN22_BENCHMARK_REPORT.md) for empirical hardware limits on free-tier runtimes (e.g., Colab Free 12.7 GB System RAM vs. High-RAM requirements for 5B models).

