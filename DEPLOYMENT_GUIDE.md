# RoboVLA - Complete Deployment and Build Guide

This comprehensive guide covers everything you need to build, run, test, and deploy the RoboVLA system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Locally](#running-locally)
5. [Testing](#testing)
6. [Building Docker Images](#building-docker-images)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- NVIDIA GPU with 12GB VRAM (RTX 3060 or better)
- 16GB RAM
- 50GB free disk space
- 4 CPU cores

**Recommended:**
- NVIDIA GPU with 24GB VRAM (RTX 3090/4090, A5000)
- 32GB RAM
- 100GB SSD storage
- 8+ CPU cores

### Software Requirements

- **Operating System**: Linux (Ubuntu 22.04 recommended), Windows 10/11 with WSL2
- **Python**: 3.9, 3.10, or 3.11
- **CUDA**: 12.1 or later
- **cuDNN**: 8.9 or later
- **Docker**: 24.0+ (for containerized deployment)
- **Kubernetes**: 1.28+ (for production deployment)
- **Git**: 2.30+

### External Services (Optional)

- **Neo4j**: 5.14+ (for knowledge graph)
- **ChromaDB**: Self-hosted or embedded
- **Prometheus**: For metrics collection
- **Grafana**: For visualization

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/robo-vla.git
cd robo-vla
```

### Step 2: Create Virtual Environment

**Linux/Mac:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

**Option A: Install from requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Install package with pip**
```bash
pip install -e .
```

**Option C: Install with development dependencies**
```bash
pip install -e ".[dev]"
```

### Step 5: Verify Installation

```bash
# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Check GPU
nvidia-smi

# Run system check
robo-vla check
```

Expected output for `robo-vla check`:
```
========================================
SYSTEM REQUIREMENTS CHECK
========================================
✓ PASS   Python 3.9+              3.10.12
✓ PASS   PyTorch 2.1+             2.1.2
✓ PASS   CUDA Available           12.1
✓ PASS   GPU Memory (12GB+)       12.0GB
========================================
✓ All checks passed!
```

---

## Configuration

### Default Configuration

The main configuration file is located at `robo_vla/configs/config.yaml`.

### Key Configuration Sections

#### 1. System Settings

```yaml
system:
  device: "cuda:0"              # Device to use (cuda:0, cpu)
  mixed_precision: true         # Enable FP16 for faster inference
  compile_models: true          # Use torch.compile() for speed
```

#### 2. Hardware Settings

```yaml
hardware:
  gpu:
    vram_gb: 12                 # Available VRAM
    max_batch_size: 4           # Maximum batch size
  cpu:
    num_workers: 4              # Number of data loader workers
```

#### 3. Model Settings

See `robo_vla/configs/config.yaml` for complete model configurations including:
- Perception models (DINO, MiDaS, Gaussian Splatting)
- Language models (Llama 3.1, CLIP)
- VLA model architecture
- RL parameters

#### 4. Server Settings

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_concurrent_requests: 100
  timeout: 30
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model cache directory
MODEL_CACHE_DIR=/path/to/models

# Database connections
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_password

# Monitoring
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=robo-vla

# CUDA settings
CUDA_VISIBLE_DEVICES=0
```

---

## Running Locally

### Option 1: Using CLI Commands

#### Run Inference

```bash
# Using the CLI
robo-vla predict \
  --image examples/warehouse_scene.jpg \
  --command "Pick the red box from the shelf" \
  --device cuda:0
```

#### Start Server

```bash
# Using the server CLI
robo-vla-serve \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

Access the API:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

#### Test the API

```bash
# Using curl
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "rgb_image": "<base64_encoded_image>",
    "text_command": "Pick the red box",
    "robot_state": [0, 0, 0, 0, 0, 0, 0, 1]
  }'
```

### Option 2: Using Python API

Create a file `test_model.py`:

```python
import torch
import numpy as np
from PIL import Image
from robo_vla import VLAModel, load_config

# Load configuration
config = load_config()

# Initialize model
print("Loading model...")
model = VLAModel(config, device="cuda:0")

# Load test image
image = Image.open("examples/warehouse_scene.jpg").convert("RGB")
rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
rgb_tensor = rgb_tensor.cuda()

# Robot state (7 joints + gripper)
robot_state = torch.zeros(8).cuda()

# Run inference
print("Running inference...")
result = model.predict(
    rgb_image=rgb_tensor,
    text_command="Pick the red box from the shelf",
    robot_state=robot_state,
)

# Print results
print(f"\nJoint Commands: {result['joint_commands'].flatten().tolist()}")
print(f"Gripper Command: {result['gripper_command'].flatten()[0]:.4f}")
print(f"Success Probability: {result['success_probability']:.2%}")
print(f"Execution Time: {result['execution_time']:.3f}s")
```

Run it:
```bash
python test_model.py
```

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_perception.py -v

# Run with coverage
pytest --cov=robo_vla --cov-report=html tests/

# Open coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

### Integration Tests

```bash
# Test full pipeline
pytest tests/test_integration.py -v -s

# Test server endpoints
pytest tests/test_server.py -v
```

### Manual Testing

```bash
# Test perception pipeline
python examples/test_perception.py

# Test language grounding
python examples/test_language.py

# Test full VLA model
python examples/test_vla.py
```

---

## Building Docker Images

### Step 1: Build the Image

```bash
# Navigate to Docker directory
cd robo_vla/deployment/docker

# Build image
docker build -t robo-vla:latest -f Dockerfile ../..

# Build with specific tag
docker build -t robo-vla:v1.0.0 -f Dockerfile ../..
```

### Step 2: Test the Image Locally

```bash
# Run container
docker run -d \
  --name robo-vla-test \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e CUDA_VISIBLE_DEVICES=0 \
  robo-vla:latest

# Check logs
docker logs robo-vla-test -f

# Test health endpoint
curl http://localhost:8000/health

# Stop and remove
docker stop robo-vla-test
docker rm robo-vla-test
```

### Step 3: Push to Registry (Optional)

```bash
# Tag for registry
docker tag robo-vla:latest your-registry.com/robo-vla:v1.0.0

# Login to registry
docker login your-registry.com

# Push image
docker push your-registry.com/robo-vla:v1.0.0
```

### Docker Compose (Alternative)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  robo-vla:
    image: robo-vla:latest
    container_name: robo-vla
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NEO4J_URI=bolt://neo4j:7687
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:5.14
    container_name: neo4j
    environment:
      - NEO4J_AUTH=neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

Run with:
```bash
docker-compose up -d
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with NVIDIA GPU support
- kubectl configured
- NVIDIA GPU Operator installed

### Step 1: Create Namespace

```bash
kubectl create namespace robo-vla
```

### Step 2: Create Persistent Volume Claim

The deployment YAML already includes a PVC definition. Apply it:

```bash
kubectl apply -f robo_vla/deployment/kubernetes/deployment.yaml
```

### Step 3: Deploy Application

```bash
# Deploy main application
kubectl apply -f robo_vla/deployment/kubernetes/deployment.yaml

# Verify deployment
kubectl get deployments -n robo-vla
kubectl get pods -n robo-vla
kubectl get services -n robo-vla
```

### Step 4: Deploy Monitoring

```bash
# Deploy Prometheus and Grafana
kubectl apply -f robo_vla/deployment/kubernetes/monitoring.yaml

# Verify monitoring stack
kubectl get pods -n robo-vla | grep -E 'prometheus|grafana'
```

### Step 5: Access Services

#### Port Forward (Development)

```bash
# Forward VLA server
kubectl port-forward -n robo-vla svc/vla-server-service 8000:80

# Forward Grafana
kubectl port-forward -n robo-vla svc/grafana 3000:80

# Forward Prometheus
kubectl port-forward -n robo-vla svc/prometheus 9090:9090
```

#### Load Balancer (Production)

```bash
# Get external IP
kubectl get svc vla-server-service -n robo-vla

# Wait for EXTERNAL-IP to be assigned
# Access at http://<EXTERNAL-IP>
```

### Step 6: Scale Deployment

```bash
# Manual scaling
kubectl scale deployment robo-vla-server -n robo-vla --replicas=5

# Check HPA status (auto-scaling)
kubectl get hpa -n robo-vla

# Describe HPA
kubectl describe hpa vla-server-hpa -n robo-vla
```

### Step 7: Update Deployment

```bash
# Update image
kubectl set image deployment/robo-vla-server \
  vla-inference=robo-vla:v1.1.0 \
  -n robo-vla

# Check rollout status
kubectl rollout status deployment/robo-vla-server -n robo-vla

# Rollback if needed
kubectl rollout undo deployment/robo-vla-server -n robo-vla
```

---

## Monitoring and Logging

### Prometheus Metrics

Access Prometheus at http://localhost:9090

**Key Metrics:**
- `vla_requests_total{endpoint="/execute", status="success"}` - Total successful requests
- `vla_request_latency_seconds` - Request latency histogram
- `vla_inference_latency_seconds` - Inference latency histogram
- `container_gpu_utilization` - GPU utilization

**Example Queries:**
```promql
# Request rate (requests per second)
rate(vla_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(vla_request_latency_seconds_bucket[5m]))

# Error rate
rate(vla_requests_total{status="error"}[5m])
```

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/admin)

**Pre-configured Dashboards:**
1. System Overview - CPU, GPU, memory usage
2. Inference Performance - Latency, throughput, error rates
3. Model Accuracy - Success rates, task completion

### Application Logs

```bash
# View logs (Kubernetes)
kubectl logs -f deployment/robo-vla-server -n robo-vla

# View logs from specific pod
kubectl logs -f <pod-name> -n robo-vla

# View logs (Docker)
docker logs -f robo-vla

# Tail last 100 lines
kubectl logs --tail=100 deployment/robo-vla-server -n robo-vla
```

### Weights & Biases (Optional)

Enable W&B logging:

```python
# In code
import wandb
wandb.init(project="robo-vla", entity="your-team")

# Or via environment
export WANDB_API_KEY=your_key
export WANDB_PROJECT=robo-vla
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in config.yaml
- Enable gradient checkpointing
- Use smaller model variants
- Clear CUDA cache: `torch.cuda.empty_cache()`

```yaml
# In config.yaml
hardware:
  gpu:
    max_batch_size: 2  # Reduce from 4

vla_model:
  optimization:
    gradient_checkpointing: true
```

#### 2. Model Download Failures

**Symptoms:**
```
HTTPError: 403 Client Error
```

**Solutions:**
- Set HuggingFace token: `export HF_TOKEN=your_token`
- Use offline mode: `export TRANSFORMERS_OFFLINE=1`
- Pre-download models:

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"
```

#### 3. Neo4j Connection Error

**Symptoms:**
```
ServiceUnavailable: Unable to connect to Neo4j
```

**Solutions:**
- Check Neo4j is running: `docker ps | grep neo4j`
- Verify connection string in config.yaml
- Check credentials

```bash
# Start Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.14
```

#### 4. Slow Inference

**Causes & Solutions:**

| Issue | Solution |
|-------|----------|
| CPU mode | Use CUDA: `device="cuda:0"` |
| No mixed precision | Enable FP16 in config |
| Not compiled | Enable torch.compile |
| Large batch | Reduce batch size |

```yaml
system:
  device: "cuda:0"
  mixed_precision: true
  compile_models: true
```

#### 5. Pod Crashes in Kubernetes

```bash
# Check pod status
kubectl describe pod <pod-name> -n robo-vla

# Check resource limits
kubectl top pod <pod-name> -n robo-vla

# View events
kubectl get events -n robo-vla --sort-by='.lastTimestamp'
```

**Common fixes:**
- Increase memory limits
- Reduce replicas
- Check GPU availability: `kubectl get nodes -o yaml | grep nvidia.com/gpu`

---

## Performance Tuning

### 1. Optimize Model Loading

```python
# Enable model compilation (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# Use bfloat16 on Ampere GPUs (RTX 30xx+)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    result = model(inputs)
```

### 2. Batch Processing

```python
# Process multiple requests in batch
results = []
for batch in dataloader:
    with torch.no_grad():
        batch_results = model(batch)
        results.extend(batch_results)
```

### 3. Model Quantization

```python
# Use INT8 quantization for DINO
config["perception"]["dino"]["quantization"] = "int8"

# Use 4-bit for Llama
config["language"]["llama"]["quantization"]["load_in_4bit"] = True
```

### 4. Caching

```yaml
# Enable caching in config.yaml
data:
  cache:
    enabled: true
    max_size_gb: 50
    ttl_hours: 24
```

### 5. Connection Pooling

```yaml
# Neo4j connection pool
knowledge:
  neo4j:
    max_connection_pool_size: 50
```

### Expected Performance

| Configuration | Latency | Memory | Throughput |
|--------------|---------|--------|------------|
| RTX 3060 (FP16) | 85ms | 9.7GB | 12 FPS |
| RTX 3090 (FP16) | 45ms | 10.2GB | 22 FPS |
| A100 (BF16) | 30ms | 11.5GB | 33 FPS |

---

## Production Checklist

Before deploying to production:

- [ ] All tests passing
- [ ] Performance benchmarks acceptable
- [ ] Monitoring configured
- [ ] Logging configured
- [ ] Backup strategy in place
- [ ] Load testing completed
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Incident response plan ready

---

## Support

For issues and questions:

- **GitHub Issues**: https://github.com/yourusername/robo-vla/issues
- **Documentation**: See README.md and other docs
- **Email**: team@robovla.ai

---

## Appendix: Quick Reference

### Common Commands

```bash
# Check system requirements
robo-vla check

# Show system info
robo-vla info

# Run inference
robo-vla predict --image <path> --command "<text>"

# Start server
robo-vla-serve --port 8000

# Train model
robo-vla-train --data-dir ./data --output-dir ./outputs

# Run tests
pytest tests/ -v

# Build Docker image
docker build -t robo-vla:latest .

# Deploy to Kubernetes
kubectl apply -f robo_vla/deployment/kubernetes/

# Check pod status
kubectl get pods -n robo-vla

# View logs
kubectl logs -f deployment/robo-vla-server -n robo-vla
```

### Configuration Files

- Main config: `robo_vla/configs/config.yaml`
- Docker: `robo_vla/deployment/docker/Dockerfile`
- Kubernetes: `robo_vla/deployment/kubernetes/deployment.yaml`
- Monitoring: `robo_vla/deployment/kubernetes/monitoring.yaml`

### Important URLs (Local)

- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Neo4j: http://localhost:7474

---

**Last Updated:** 2024-11-18
**Version:** 1.0.0
