# 🚀 Deployment Guide

Complete guide for deploying the Mental Health Prediction system.

---

## 📦 Components

1. **Streamlit Web App** - Interactive UI for employees
2. **FastAPI REST API** - Backend for integrations
3. **ML Models** - XGBoost and Gradient Boosting
4. **SHAP Explainability** - Model interpretability

---

## 🛠️ Local Development

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/mental-health-ml-analysis.git
cd mental-health-ml-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_full.txt
```

### 2. Train Models

```bash
# Run Jupyter notebook to train models
jupyter notebook Mental_Health_Analysis_FINAL.ipynb

# Or use Python script
python train_models.py
```

Models will be saved in `models/` directory:
- `model_disorder.pkl`
- `model_treatment.pkl`
- `scaler.pkl`
- `feature_names.pkl`

### 3. Run Streamlit App

```bash
streamlit run app.py
```

Access at: http://localhost:8501

### 4. Run API Server

```bash
# Development
uvicorn api:app --reload --port 8000

# Production
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
```

Access at: http://localhost:8000
API Docs: http://localhost:8000/docs

---

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud (Easiest)

1. **Push to GitHub**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Deploy!

3. **Configuration**
   - Create `secrets.toml` for API keys (if needed)
   - Set Python version in `.streamlit/config.toml`

**Pros:**
- ✅ Free tier available
- ✅ Easy GitHub integration
- ✅ Automatic updates
- ✅ HTTPS included

**Cons:**
- ❌ Limited to Streamlit apps
- ❌ Resource constraints on free tier

---

### Option 2: Heroku

1. **Create `Procfile`**
```
web: sh setup.sh && streamlit run app.py
```

2. **Create `setup.sh`**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. **Deploy**
```bash
heroku create your-app-name
git push heroku main
```

---

### Option 3: AWS EC2

1. **Launch EC2 Instance**
   - Ubuntu 22.04 LTS
   - t2.medium or larger
   - Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. **Setup Server**
```bash
# SSH into instance
ssh ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv -y

# Clone repository
git clone https://github.com/yourusername/mental-health-ml-analysis.git
cd mental-health-ml-analysis

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_full.txt
```

3. **Setup Nginx**
```bash
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/mental-health-app

# Add configuration
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/mental-health-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

4. **Setup Systemd Service**
```bash
sudo nano /etc/systemd/system/mental-health-app.service

[Unit]
Description=Mental Health Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/mental-health-ml-analysis
Environment="PATH=/home/ubuntu/mental-health-ml-analysis/venv/bin"
ExecStart=/home/ubuntu/mental-health-ml-analysis/venv/bin/streamlit run app.py

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable mental-health-app
sudo systemctl start mental-health-app
```

5. **Setup SSL (Let's Encrypt)**
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

### Option 4: Docker

1. **Create `Dockerfile`**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements_full.txt .
RUN pip install --no-cache-dir -r requirements_full.txt

# Copy application
COPY . .

# Create models directory
RUN mkdir -p models

# Expose ports
EXPOSE 8501 8000

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Create `docker-compose.yml`**
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  api:
    build: .
    command: uvicorn api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

3. **Build and Run**
```bash
# Build
docker-compose build

# Run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

### Option 5: Google Cloud Run

1. **Create `app.yaml`**
```yaml
runtime: python39

entrypoint: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

automatic_scaling:
  min_instances: 0
  max_instances: 10
```

2. **Deploy**
```bash
gcloud app deploy
```

---

## 🔐 Security Considerations

### 1. Environment Variables
```bash
# Create .env file
API_KEY=your-secret-key
DATABASE_URL=your-db-url
SECRET_KEY=random-secret
```

```python
# Load in app
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')
```

### 2. HTTPS
Always use HTTPS in production:
- Streamlit Cloud: Automatic
- Custom domain: Let's Encrypt
- AWS: ACM Certificate
- Google Cloud: Automatic

### 3. Rate Limiting
```python
# In FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(...):
    ...
```

### 4. Authentication
```python
# Add API key authentication
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(response: SurveyResponse, api_key: str = Depends(verify_api_key)):
    ...
```

---

## 📊 Monitoring

### Application Monitoring
```bash
# Install monitoring tools
pip install prometheus-client

# Add to FastAPI
from prometheus_client import Counter, Histogram

requests_total = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
```

### Logs
```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### API Testing
```bash
# Install testing tools
pip install httpx pytest-asyncio

# Run API tests
pytest tests/test_api.py -v
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 📈 Scaling

### Horizontal Scaling
```yaml
# docker-compose with replicas
services:
  api:
    deploy:
      replicas: 3
```

### Load Balancer
```nginx
# Nginx load balancing
upstream api_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    location /api/ {
        proxy_pass http://api_backend;
    }
}
```

---

## 🔄 CI/CD

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          # Your deployment script
          ./deploy.sh
```

---

## 📝 Checklist

Before deploying to production:

- [ ] Models trained and saved
- [ ] Environment variables configured
- [ ] HTTPS enabled
- [ ] Rate limiting implemented
- [ ] Authentication added
- [ ] Logging configured
- [ ] Error handling tested
- [ ] Load testing completed
- [ ] Backup strategy in place
- [ ] Monitoring setup
- [ ] Documentation updated

---

## 🆘 Troubleshooting

### Streamlit Issues
```bash
# Clear cache
streamlit cache clear

# Check port
netstat -an | grep 8501
```

### API Issues
```bash
# Check logs
tail -f app.log

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### Model Loading Issues
```bash
# Verify models exist
ls -lh models/

# Test model loading
python -c "import joblib; model = joblib.load('models/model_disorder.pkl'); print('✓ Loaded')"
```

---

## 📞 Support

For deployment issues:
- Check logs first
- Review error messages
- Consult documentation
- Open GitHub issue

---

**Happy Deploying! 🚀**
