# ☁️ Quran ASR Kids — Cloud Deployment Guide

A complete guide to deploy and serve the Quran ASR API on cloud infrastructure.

---

## Table of Contents

1. [Quick Reference](#-quick-reference)
2. [Option A: Docker (Recommended)](#-option-a-docker-recommended)
3. [Option B: Direct VM Setup](#-option-b-direct-vm-setup)
4. [Nginx Reverse Proxy + SSL](#-nginx-reverse-proxy--ssl)
5. [Cloud Provider Guides](#-cloud-provider-guides)
   - [AWS EC2 + GPU](#aws-ec2--gpu)
   - [Google Cloud (GCP)](#google-cloud-gcp)
   - [Azure](#microsoft-azure)
   - [RunPod / Vast.ai (Budget GPU)](#runpod--vastai-budget-gpu)
6. [Systemd Service (Auto-Restart)](#-systemd-service-auto-restart)
7. [Scaling & Performance](#-scaling--performance)
8. [Monitoring & Logging](#-monitoring--logging)
9. [Security Checklist](#-security-checklist)
10. [Cost Estimates](#-cost-estimates)

---

## 📋 Quick Reference

| Item | Value |
|:-----|:------|
| **Model** | `NAMAA-Space/Cohere-Speech-Tashkeel-2B` (single 2B model: words + harakat) |
| **GPU** | **Required** — NVIDIA **T4 (16 GB) / L4 / A10G** (needs ~5–6 GB VRAM, bf16) |
| **CPU-only?** | **No** — impractical for a 2B model |
| **System RAM** | 8 GB+ |
| **Disk** | ~15 GB (model ~5 GB + deps + cache) |
| **Port** | 8000 (default) |
| **Python** | 3.11 |
| **CUDA** | 12.x |
| **transformers** | **≥ 5.4** (do not downgrade) · **bf16 required** (fp16 → garbage) |
| **Model download** | ~5 GB (baked into the Docker image at build; or first run from HuggingFace) |
| **Health check** | `GET /health` |

---

## 🐳 Option A: Docker (Recommended)

### Dockerfile

**Do not write your own** — a ready `delivery/Dockerfile` is in the repo, already configured for
the NAMAA model (CUDA 12.1 runtime, Python 3.11, `torch` cu121, `transformers>=5.4`, and it
**bakes the ~5 GB model into the image** at build time so startup is fast and offline-capable).
Just build it. (If you want to inspect it: `cat delivery/Dockerfile`.)

### Build & Run

```bash
cd delivery

# Build (downloads the model into the image — first build is large/slow, ~5 GB)
docker build -t quran-asr .

# Run with GPU (a CUDA GPU is REQUIRED — the 2B model is impractical on CPU)
docker run -d \
    --name quran-asr \
    --gpus all \
    -p 8000:8000 \
    -e AI_API_KEY="<shared secret>" \
    -e PUBLIC_BASE_URL="https://ai.yutlaquran.com" \
    -v $(pwd)/temp_storage:/app/temp_storage \
    --restart unless-stopped \
    quran-asr

docker logs -f quran-asr            # watch it load NAMAA (~10–30 s)
curl http://localhost:8000/health   # {"status":"ok","model_loaded":true}
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  quran-asr:
    build: .
    container_name: quran-asr
    ports:
      - "8000:8000"
    volumes:
      - ./temp_storage:/app/temp_storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker compose up -d
```

---

## 🖥️ Option B: Direct VM Setup

### Ubuntu 22.04 Setup Script

```bash
#!/bin/bash
# setup.sh — Run on a fresh Ubuntu 22.04 VM with NVIDIA GPU

set -e

echo "=== 1. System Packages ==="
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip ffmpeg git

echo "=== 2. NVIDIA Drivers (if not pre-installed) ==="
# Skip if your cloud VM already has drivers (most do)
# sudo apt-get install -y nvidia-driver-535

echo "=== 3. Application Setup ==="
cd /opt
sudo mkdir -p quran-asr
sudo chown $USER:$USER quran-asr

# Copy delivery folder contents here (via scp, git, etc.)
# scp -r delivery/* user@server:/opt/quran-asr/

cd /opt/quran-asr

echo "=== 4. Python Environment ==="
python3.10 -m venv venv
source venv/bin/activate

echo "=== 5. Install PyTorch + Dependencies ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

echo "=== 6. Test ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from core.grader import QuranGrader; print('Grader OK')"

echo "=== 7. Start Server (test) ==="
python main.py
```

### Upload Files to Server

```bash
# From your local machine:
scp -r delivery/* user@YOUR_SERVER_IP:/opt/quran-asr/
```

---

## 🔒 Nginx Reverse Proxy + SSL

### Install Nginx + Certbot

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

### Nginx Config

Create `/etc/nginx/sites-available/quran-asr`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    # SSL (managed by Certbot)
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Upload size (audio files can be large)
    client_max_body_size 50M;

    # Proxy to FastAPI
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts (ASR can take a few seconds)
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }

    # Serve feedback audio directly (faster)
    location /audio/ {
        alias /opt/quran-asr/temp_storage/;
        expires 1h;
    }
}
```

### Enable & SSL

```bash
sudo ln -s /etc/nginx/sites-available/quran-asr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get free SSL certificate
sudo certbot --nginx -d api.yourdomain.com
```

Now your API is at: `https://api.yourdomain.com/grade_recitation`

---

## ☁️ Cloud Provider Guides

### AWS EC2 + GPU

**Recommended Instance:** `g4dn.xlarge` (1x T4 GPU, 4 vCPU, 16 GB RAM) — ~$0.526/hr

```bash
# 1. Launch EC2 instance
#    AMI: "Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 22.04)"
#    Instance type: g4dn.xlarge
#    Storage: 30 GB gp3
#    Security Group: Allow TCP 80, 443, 22

# 2. SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com

# 3. Upload delivery folder
# (From local machine):
scp -i your-key.pem -r delivery/* ubuntu@ec2-xx-xx-xx-xx:/opt/quran-asr/

# 4. Setup (on server)
cd /opt/quran-asr
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 5. Run
python main.py
```

**Cost-Saving Tips:**
- Use **Spot Instances** for ~70% savings (but can be interrupted)
- Use **Reserved Instances** for steady workloads
- Use `g5.xlarge` (A10G GPU) for better performance at similar cost

---

### Google Cloud (GCP)

**Recommended:** `n1-standard-4` + 1x T4 GPU — ~$0.45/hr

```bash
# 1. Create VM
gcloud compute instances create quran-asr \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=30GB \
    --maintenance-policy=TERMINATE

# 2. SSH
gcloud compute ssh quran-asr --zone=us-central1-a

# 3. Upload & setup (same as above)
```

---

### Microsoft Azure

**Recommended:** `Standard_NC4as_T4_v3` (1x T4, 4 vCPU, 28 GB) — ~$0.526/hr

```bash
# 1. Create VM
az vm create \
    --resource-group myResourceGroup \
    --name quran-asr \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
    --size Standard_NC4as_T4_v3 \
    --admin-username azureuser \
    --generate-ssh-keys

# 2. Install NVIDIA drivers
az vm extension set \
    --resource-group myResourceGroup \
    --vm-name quran-asr \
    --name NvidiaGpuDriverLinux \
    --publisher Microsoft.HCPCompute

# 3. SSH, upload, setup (same as above)
```

---

### RunPod / Vast.ai (Budget GPU)

**Cheapest option** for testing or low-traffic. GPU VMs from ~$0.20/hr.

```bash
# RunPod: Create a pod with PyTorch template
# 1. Go to runpod.io → Create Pod
# 2. Select: RTX 3090 or A4000 ($0.20-0.35/hr)
# 3. Template: RunPod PyTorch 2.0
# 4. Disk: 20 GB

# Once inside the pod:
cd /workspace
# Upload delivery folder
pip install -r requirements.txt
python main.py
```

---

## 🔄 Systemd Service (Auto-Restart)

Create `/etc/systemd/system/quran-asr.service`:

```ini
[Unit]
Description=Quran ASR Kids API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/quran-asr
Environment="PATH=/opt/quran-asr/venv/bin:/usr/bin"
ExecStart=/opt/quran-asr/venv/bin/python main.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/quran-asr.log
StandardError=append:/var/log/quran-asr-error.log

# GPU memory management
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable quran-asr
sudo systemctl start quran-asr

# Check status
sudo systemctl status quran-asr

# View logs
sudo journalctl -u quran-asr -f
```

---

## 📈 Scaling & Performance

### Single Server Optimization

```python
# In main.py, replace the last line with:
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,          # Keep 1 for GPU (model is loaded per-worker)
        timeout_keep_alive=30,
        limit_concurrency=10,  # Max concurrent requests
    )
```

### Production with Gunicorn

```bash
# 1 worker per GPU (model loads once per worker)
gunicorn main:app \
    -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 30
```

### Performance Benchmarks

| Config | Requests/min | Latency (avg) |
|:-------|:------------|:-------------|
| CPU (4 core) | ~6 | ~10s |
| T4 GPU | ~30 | ~2s |
| A10G GPU | ~50 | ~1.2s |
| A100 GPU | ~80 | ~0.7s |

*Benchmarks based on 5-second audio files, single worker.*

### Handling High Traffic

For >50 concurrent users:

1. **Load Balancer** (AWS ALB / Nginx) in front of multiple GPU servers
2. **Queue System** (Redis + Celery) for async processing:
   ```
   Client → API → Redis Queue → GPU Worker → Response
   ```
3. **Auto-Scaling** with Kubernetes + GPU nodes

---

## 📊 Monitoring & Logging

### Health Check Endpoint

```bash
# Cron job every minute
* * * * * curl -sf http://localhost:8000/health || systemctl restart quran-asr
```

### Log Rotation

Create `/etc/logrotate.d/quran-asr`:

```
/var/log/quran-asr*.log {
    daily
    missingok
    rotate 14
    compress
    notifempty
    create 0640 ubuntu ubuntu
}
```

### Metrics Endpoint (Optional)

Add to `main.py`:

```python
import time
from collections import defaultdict

# Simple metrics counter
metrics = defaultdict(int)

@app.middleware("http")
async def track_metrics(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    metrics["total_requests"] += 1
    metrics["total_time"] += duration
    if response.status_code == 200:
        metrics["successful"] += 1
    return response

@app.get("/metrics")
async def get_metrics():
    return {
        "total_requests": metrics["total_requests"],
        "successful": metrics["successful"],
        "avg_latency": metrics["total_time"] / max(metrics["total_requests"], 1),
        "model_loaded": model is not None
    }
```

---

## 🔐 Security Checklist

| # | Item | How |
|:--|:-----|:----|
| 1 | **HTTPS only** | Nginx + Certbot (see above) |
| 2 | **Rate limiting** | Nginx: `limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;` |
| 3 | **CORS** | Add to main.py (see INTEGRATION_GUIDE.md) |
| 4 | **File size limit** | Nginx: `client_max_body_size 50M;` |
| 5 | **Firewall** | Only expose ports 80, 443. Block 8000 externally |
| 6 | **API Key** (optional) | Add header check in FastAPI middleware |
| 7 | **Temp cleanup** | Cron (30-day, see note below) — do **not** delete `feedback_*.mp3` early |
| 8 | **Non-root** | Run as non-root user (systemd User= directive) |

### Temp File Cleanup Cron

> ⚠️ **Do not delete `feedback_*.mp3` after 1 hour.** The backend references those TTS
> feedback URLs for **30 days** (recitation lifetime). Deleting them early breaks playback
> of past recitations. Clean only files older than 30 days:

```bash
# Clean temp files older than 30 days (once daily)
0 3 * * * find /opt/quran-asr/temp_storage -type f -mtime +30 -delete
```

### Optional: API Key Authentication

Add to `main.py`:

```python
from fastapi import Security, Depends
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get("QURAN_API_KEY", "your-secret-key-here")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# Then add to endpoints:
@app.post("/grade_recitation", dependencies=[Depends(verify_api_key)])
```

---

## 💰 Cost Estimates

### Monthly Cost (24/7 operation)

| Provider | Instance | GPU | Monthly Cost |
|:---------|:---------|:----|:------------|
| **AWS** | g4dn.xlarge | T4 | ~$383/mo |
| **AWS Spot** | g4dn.xlarge | T4 | ~$115/mo |
| **GCP** | n1-std-4 + T4 | T4 | ~$325/mo |
| **GCP Preemptible** | n1-std-4 + T4 | T4 | ~$130/mo |
| **Azure** | NC4as_T4_v3 | T4 | ~$383/mo |
| **RunPod** | Community | 3090 | ~$144/mo |
| **Vast.ai** | Various | 3090 | ~$100/mo |

### Cost-Saving Strategies

1. **Spot/Preemptible instances**: 60-70% cheaper, acceptable for non-critical
2. **Auto-shutdown**: Turn off GPU when not in use (e.g., night hours)
3. **CPU fallback**: Use CPU instance during low-traffic periods (~$30/mo)
4. **Serverless GPU**: Modal.com, Banana.dev — pay per request (~$0.001/request)

### Serverless Option (Pay-Per-Request)

For low traffic (<1000 requests/day), serverless GPU is cheapest:

```python
# Example: modal.com deployment
# modal_app.py
import modal

app = modal.App("quran-asr")
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@app.function(gpu="T4", image=image, timeout=120)
def grade(audio_bytes, target_text):
    # ... load model, process, return result
    pass
```

---

## 🗺️ Recommended Architecture

### Small Scale (< 100 users)

```
[Mobile App] → [HTTPS] → [Single GPU VM] → [Quran ASR API]
                              ↑
                        Nginx + Certbot
                        systemd service
```

### Medium Scale (100-1000 users)

```
[Mobile App] → [HTTPS] → [Load Balancer] → [GPU VM 1] → [Quran ASR API]
                              ↓            → [GPU VM 2] → [Quran ASR API]
                         Nginx / ALB
```

### Large Scale (1000+ users)

```
[Mobile App] → [HTTPS] → [API Gateway] → [Redis Queue] → [GPU Workers (auto-scale)]
                              ↓
                     Rate Limiting + Auth
                     Response Caching
```
