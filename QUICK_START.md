# Quick Start Guide - 0.00sec AWS System

This guide helps you get started quickly with the refactored 0.00sec system.

## Prerequisites

- Python 3.10+
- AWS account (for backend deployment)
- Git

---

## Local Development & Testing

### 1. Install Frontend Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Configure API Endpoint

```bash
# Copy example config
cp .env.example .env

# Edit .env
# Set API_BASE_URL to your EC2 instance when deployed
# For now, use localhost for testing
nano .env
```

Example `.env`:
```bash
API_BASE_URL=http://localhost:8000/api/v1
```

### 3. Test API Client (Optional)

```python
from frontend.api_client import create_client

client = create_client("http://localhost:8000/api/v1")
# Will connect when backend is running
```

---

## Backend Deployment (AWS)

### Step 1: Setup AWS Resources (15 minutes)

```bash
# Create S3 buckets
aws s3 mb s3://mvp-inputs-bucket
aws s3 mb s3://mvp-outputs-bucket

# Create SQS queue
aws sqs create-queue --queue-name zerosec-jobs-queue

# Create RDS (takes ~10 minutes)
# Follow: DEPLOYMENT_GUIDE.md Section 1.3
```

### Step 2: Deploy API Server (10 minutes)

```bash
# Launch EC2 instance (via AWS Console or CLI)
# SSH into instance
ssh ec2-user@<EC2-IP>

# Install dependencies
sudo yum update -y
sudo yum install python3.10 git -y

# Clone and setup
git clone <your-repo>
cd 0sec_dataanalysis_app/backend/api
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Edit with AWS values

# Run server
python main.py
```

### Step 3: Initialize Database (2 minutes)

```bash
cd ../migrations
python create_tables.py
```

### Step 4: Deploy Worker (15 minutes)

```bash
# Build Docker image
cd ../worker
docker build -t zerosec-worker .

# Push to ECR
# Follow: DEPLOYMENT_GUIDE.md Section 3

# Create ECS service
# Follow: DEPLOYMENT_GUIDE.md Section 3.5
```

### Step 5: Update Frontend Config

```bash
# Edit frontend/.env
API_BASE_URL=http://<EC2-PUBLIC-IP>:8000/api/v1
```

---

## Using the System

### Option A: Direct API Usage

```python
from frontend.api_client import create_client
from backend.shared.models import JobType
from pathlib import Path

# Create client
client = create_client("http://<EC2-IP>:8000/api/v1")

# Submit job and wait
output_files = client.submit_and_wait(
    file_path=Path("data.csv"),
    job_type=JobType.OPTIMIZATION,
    parameters={"num_points": 15},
    output_dir=Path("./results")
)

print(f"Downloaded {len(output_files)} files")
```

### Option B: Integrate with Existing GUI

See [frontend/integration_example.py](frontend/integration_example.py)

**Minimal changes needed:**

1. Add API config to `__init__`:
```python
from dotenv import load_dotenv
load_dotenv("frontend/.env")
self.api_base_url = os.getenv('API_BASE_URL')
```

2. Replace local workers with RemoteAnalysisWorker
3. Keep all UI code unchanged

---

## Quick Test

### 1. Test API Health

```bash
curl http://<EC2-IP>:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "database_connected": true
}
```

### 2. Test Complete Workflow

```bash
# Create test file
echo "col1,col2,col3\n1,2,3\n4,5,6" > test.csv

# Run test script
python test_api_workflow.py
```

---

## File Structure Overview

```
0sec_dataanalysis_app/
├── backend/
│   ├── shared/          # Models, config, database
│   ├── api/             # FastAPI server (runs on EC2)
│   ├── worker/          # Job processor (runs on ECS)
│   └── migrations/      # Database setup
│
├── frontend/
│   ├── api_client.py          # API communication
│   ├── integration_example.py # GUI integration guide
│   └── .env                   # Config (API endpoint)
│
├── 0sec.py              # Main GUI (to be modified)
├── dsaitekika.py        # D-optimization logic (unchanged)
├── linear_analysis_module.py  # Linear analysis (unchanged)
└── ...                  # Other existing files (unchanged)
```

---

## Common Tasks

### Run API Server Locally (Development)

```bash
cd backend/api
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### Run Worker Locally (Development)

```bash
cd backend/worker
source venv/bin/activate
python worker.py
```

### View Logs

```bash
# API logs (systemd)
sudo journalctl -u zerosec-api -f

# Worker logs (CloudWatch)
aws logs tail /ecs/zerosec-worker --follow
```

### Update Deployed Code

```bash
# API
ssh ec2-user@<EC2-IP>
cd 0sec_dataanalysis_app
git pull
sudo systemctl restart zerosec-api

# Worker
cd backend/worker
docker build -t zerosec-worker .
docker push <ECR-URL>
aws ecs update-service --cluster zerosec-cluster --service zerosec-worker-service --force-new-deployment
```

---

## Troubleshooting

### "Connection refused" when calling API

- Check EC2 security group allows port 8000
- Verify API server is running: `systemctl status zerosec-api`
- Check logs: `journalctl -u zerosec-api`

### Jobs stuck in "queued"

- Check worker is running: `aws ecs list-tasks --cluster zerosec-cluster`
- Check SQS queue: `aws sqs get-queue-attributes ...`
- View worker logs in CloudWatch

### Database errors

- Verify RDS security group allows EC2/ECS access
- Test connection: `psql -h <RDS-HOST> -U zerosec_admin -d zerosec_db`
- Check credentials in .env files

### Upload/download failures

- Verify S3 bucket names in .env
- Check S3 bucket permissions
- Verify presigned URL expiration (default 1 hour)

---

## Next Steps

1. ✅ Deploy backend to AWS
2. ✅ Test with sample data
3. 📝 Integrate with existing GUI
4. 🔒 Add authentication (production)
5. 📊 Setup monitoring (CloudWatch)
6. 🚀 Configure auto-scaling

---

## Documentation

- **Architecture:** [README_ARCHITECTURE.md](README_ARCHITECTURE.md)
- **API Reference:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Deployment:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Implementation:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Integration:** [frontend/integration_example.py](frontend/integration_example.py)

---

## Support Checklist

Before asking for help:

- [ ] Checked API health endpoint
- [ ] Reviewed logs (API, Worker, CloudWatch)
- [ ] Verified .env configurations
- [ ] Tested database connection
- [ ] Checked AWS resource status (EC2, RDS, ECS, S3, SQS)
- [ ] Reviewed error messages carefully

---

## Production Checklist

Before going to production:

- [ ] Add authentication (Cognito/IAM)
- [ ] Enable HTTPS (ALB + ACM certificate)
- [ ] Restrict CORS origins
- [ ] Enable S3 bucket encryption
- [ ] Setup CloudWatch alarms
- [ ] Configure backup retention
- [ ] Implement rate limiting
- [ ] Add monitoring dashboards
- [ ] Test disaster recovery
- [ ] Document runbooks

---

## Success Criteria

You're ready to use the system when:

✅ API health check returns "healthy"  
✅ Can upload file and create job  
✅ Worker processes jobs successfully  
✅ Can download results  
✅ GUI shows progress from remote worker  

**Congratulations! Your system is operational.**
