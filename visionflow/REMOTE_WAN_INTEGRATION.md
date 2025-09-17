# Remote WAN2.1 Integration Guide

This guide explains how to configure your VisionFlow codebase to work with a remote WAN2.1 endpoint with minimal changes.

## Overview

The integration uses an adapter pattern that allows switching between local and remote WAN2.1 services based on configuration. This approach requires minimal changes to your existing codebase.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│  WAN Adapter     │───▶│  Local/Remote   │
│                 │    │                  │    │  WAN Service    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

The adapter automatically routes requests to either:
- **Local WAN Service**: Your existing local WAN2.1 implementation
- **Remote WAN Service**: External WAN2.1 endpoint via HTTP API

## Configuration

### Environment Variables

Add these environment variables to enable remote WAN mode:

```bash
# Enable remote WAN mode
USE_REMOTE_WAN=true

# Remote WAN endpoint configuration
REMOTE_WAN_URL=http://your-remote-wan-service:8002
REMOTE_WAN_API_KEY=your_api_key_here
REMOTE_WAN_TIMEOUT=600
REMOTE_WAN_MAX_RETRIES=3
```

### Configuration File

Copy `remote-wan.env.example` to `.env` and configure:

```bash
cp remote-wan.env.example .env
# Edit .env with your remote WAN endpoint details
```

## Usage

### 1. Local Mode (Default)

```bash
# Use local WAN2.1 service
docker-compose up
```

### 2. Remote WAN Mode

```bash
# Use remote WAN2.1 service
docker-compose -f docker-compose.yml -f docker-compose.remote-wan.yml up
```

### 3. Testing with Mock Endpoint

```bash
# Start mock remote WAN service for testing
python -m visionflow.services.generation.remote_wan_mock

# In another terminal, start main services in remote mode
USE_REMOTE_WAN=true REMOTE_WAN_URL=http://localhost:8002 docker-compose up
```

## API Integration

### Remote WAN Endpoint Requirements

Your remote WAN2.1 endpoint should implement these endpoints:

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "wan2.1"
}
```

#### Video Generation
```http
POST /generate
```

Request Body:
```json
{
  "prompt": "A cat playing with a ball",
  "original_prompt": "A cat playing with a ball",
  "generation_params": {
    "duration": 5,
    "fps": 24,
    "resolution": "512x512",
    "width": 512,
    "height": 512,
    "seed": 42,
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "quality": "medium"
  },
  "routing_decision": {
    "service": "wan2.1",
    "reason": "default"
  },
  "job_id": "uuid-string",
  "optimization": {}
}
```

Response:
```json
{
  "status": "completed",
  "job_id": "uuid-string",
  "generation_time": 45.2,
  "video_path": "/path/to/generated/video.mp4",
  "video_url": "http://example.com/video.mp4",
  "model": "wan2.1-14b",
  "quality": "medium",
  "resolution": "512x512"
}
```

## Code Changes Made

### 1. New Files Created

- `visionflow/services/generation/remote_wan_client.py` - Remote WAN HTTP client
- `visionflow/services/generation/wan_adapter.py` - Adapter for local/remote switching
- `visionflow/services/generation/remote_wan_mock.py` - Mock endpoint for testing
- `docker-compose.remote-wan.yml` - Docker Compose override for remote mode
- `remote-wan.env.example` - Environment configuration template

### 2. Files Modified

- `visionflow/shared/config.py` - Added remote WAN configuration options
- `visionflow/services/generation/service.py` - Updated to use adapter

### 3. Minimal Changes Required

The integration requires **zero changes** to your existing:
- API endpoints
- Database models
- Business logic
- Frontend code
- Celery tasks

## Error Handling

The remote client includes:
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Timeout Handling**: Configurable request timeouts
- **Health Checks**: Automatic endpoint health monitoring
- **Fallback Support**: Can fall back to local service if remote fails

## Monitoring

The adapter provides health check endpoints:

```bash
# Check adapter health
curl http://localhost:8000/health/wan

# Response
{
  "service": "remote_wan",
  "healthy": true,
  "endpoint": "http://your-remote-wan-service:8002"
}
```

## Performance Considerations

### Remote Mode
- **Network Latency**: Consider network latency to remote endpoint
- **Timeout Settings**: Adjust `REMOTE_WAN_TIMEOUT` based on your network
- **Retry Logic**: Configure `REMOTE_WAN_MAX_RETRIES` for reliability

### Local Mode
- **GPU Memory**: Local mode uses GPU memory for model loading
- **Resource Usage**: Higher local resource consumption

## Security

### API Key Authentication
```bash
REMOTE_WAN_API_KEY=your_secure_api_key
```

### Network Security
- Use HTTPS for production remote endpoints
- Implement proper authentication on remote WAN service
- Consider VPN or private network for internal services

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if remote endpoint is running
   curl http://your-remote-wan-service:8002/health
   ```

2. **Timeout Errors**
   ```bash
   # Increase timeout
   REMOTE_WAN_TIMEOUT=1200
   ```

3. **Authentication Errors**
   ```bash
   # Check API key
   echo $REMOTE_WAN_API_KEY
   ```

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=debug
```

## Migration Guide

### From Local to Remote

1. **Test with Mock**: Start with the mock endpoint
2. **Configure Remote**: Set up your remote WAN2.1 service
3. **Update Environment**: Set `USE_REMOTE_WAN=true`
4. **Deploy**: Use remote WAN Docker Compose configuration

### From Remote to Local

1. **Set Environment**: Set `USE_REMOTE_WAN=false`
2. **Deploy**: Use standard Docker Compose configuration

## Example Remote WAN Service

Here's a minimal example of a remote WAN2.1 service:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    generation_params: dict
    job_id: str

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/generate")
async def generate(request: GenerationRequest):
    # Your WAN2.1 generation logic here
    return {
        "status": "completed",
        "job_id": request.job_id,
        "video_path": "/path/to/video.mp4"
    }
```

## Support

For issues or questions:
1. Check the logs: `docker-compose logs generation-service`
2. Verify configuration: `docker-compose config`
3. Test connectivity: Use the health check endpoints

This integration provides a clean, minimal-change solution for using remote WAN2.1 endpoints while maintaining full compatibility with your existing local implementation.





