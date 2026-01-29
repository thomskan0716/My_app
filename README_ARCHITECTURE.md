# 0.00sec System - AWS Deployment Architecture

## Overview

This document describes the refactored architecture that separates the 0.00sec data analysis application into:

1. **Frontend** (Python Desktop App) - Local PC
2. **Backend API** (FastAPI on EC2) - AWS Cloud
3. **Worker** (ECS Container) - AWS Cloud

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER'S LOCAL PC                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Frontend (Python Desktop App - PySide6)                 │  │
│  │  - Same UI as original app                               │  │
│  │  - Uses API client to communicate with backend           │  │
│  │  - Displays progress from remote worker                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS (API calls)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          AWS CLOUD                              │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  EC2 Instance (API Server)                             │    │
│  │  - FastAPI application                                 │    │
│  │  - Endpoints: /presign/upload, /jobs, /presign/download│    │
│  │  - Generates presigned S3 URLs                         │    │
│  │  - Enqueues jobs to SQS                                │    │
│  │  - Queries job status from RDS                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ↓                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  RDS PostgreSQL                                        │    │
│  │  Tables: jobs, job_artifacts                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ↓                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  SQS Queue                                             │    │
│  │  - Job messages with parameters                        │    │
│  │  - DLQ for failed messages                             │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ↓                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  ECS Fargate (Worker Containers)                       │    │
│  │  - Polls SQS for jobs                                  │    │
│  │  - Downloads input from S3                             │    │
│  │  - Runs analysis (D-opt, linear, nonlinear, class)     │    │
│  │  - Uploads results to S3                               │    │
│  │  - Updates progress in RDS                             │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ↓                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  S3 Buckets                                            │    │
│  │  - mvp-inputs-bucket: Input CSV/Excel files            │    │
│  │  - mvp-outputs-bucket: Result files (Excel, PNG, etc)  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
0sec_dataanalysis_app/
├── backend/
│   ├── shared/                   # Shared models and utilities
│   │   ├── __init__.py
│   │   ├── models.py            # Pydantic models
│   │   ├── config.py            # Configuration management
│   │   └── database.py          # SQLAlchemy models
│   │
│   ├── api/                     # EC2 API Server
│   │   ├── main.py              # FastAPI application
│   │   ├── requirements.txt     # API dependencies
│   │   └── .env.example         # Environment template
│   │
│   ├── worker/                  # ECS Worker
│   │   ├── worker.py            # Main worker loop
│   │   ├── job_runners.py       # Analysis job runners
│   │   ├── Dockerfile           # Container definition
│   │   ├── requirements.txt     # Worker dependencies
│   │   └── .env.example         # Environment template
│   │
│   └── migrations/              # Database migrations
│       └── create_tables.py     # Initial schema
│
├── frontend/                    # Desktop Application
│   ├── api_client.py            # API communication client
│   ├── requirements.txt         # Frontend dependencies
│   └── .env.example             # Environment template
│
├── 0sec.py                      # Main GUI (to be updated)
├── dsaitekika.py                # D-optimization logic
├── linear_analysis_module.py    # Linear analysis logic
├── 01_model_builder.py          # Nonlinear model building
├── 02_prediction.py             # Prediction logic
├── 03_pareto_analyzer.py        # Pareto analysis
└── README_ARCHITECTURE.md       # This file
```

## Data Flow

### 1. Upload Input File

```
Frontend → API: POST /presign/upload
Frontend ← API: {job_id, upload_url}
Frontend → S3: PUT to upload_url (direct upload)
```

### 2. Submit Job

```
Frontend → API: POST /jobs
API → RDS: INSERT job record (status=queued)
API → SQS: Send job message
Frontend ← API: {job_id, status}
```

### 3. Process Job

```
Worker → SQS: Long poll for messages
Worker ← SQS: Receive job message
Worker → S3: Download input file
Worker → RDS: UPDATE status=running, progress=10%
Worker: Run analysis (with progress callbacks)
Worker → RDS: UPDATE progress=50% (periodic)
Worker → S3: Upload output files
Worker → RDS: INSERT artifact records
Worker → RDS: UPDATE status=completed, progress=100%
Worker → SQS: Delete message
```

### 4. Poll Status

```
Frontend → API: GET /jobs/{job_id}/status (every 2 seconds)
Frontend ← API: {status, progress, message}
(Repeat until status == completed or failed)
```

### 5. Download Results

```
Frontend → API: GET /jobs/{job_id}/artifacts
Frontend ← API: {artifacts: [file_key, file_name, ...]}
Frontend → API: POST /presign/download {file_key}
Frontend ← API: {download_url}
Frontend → S3: GET from download_url (direct download)
```

## Database Schema

### jobs table

```sql
CREATE TABLE jobs (
    job_id VARCHAR(36) PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    progress_percent INTEGER DEFAULT 0,
    status_message TEXT DEFAULT '',
    error_message TEXT,
    input_bucket VARCHAR(255) NOT NULL,
    input_key VARCHAR(1024) NOT NULL,
    output_bucket VARCHAR(255) NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);
```

### job_artifacts table

```sql
CREATE TABLE job_artifacts (
    artifact_id SERIAL PRIMARY KEY,
    job_id VARCHAR(36) NOT NULL,
    file_key VARCHAR(1024) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    size_bytes BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_artifacts_job_id ON job_artifacts(job_id);
```

## Job Types

1. **optimization** - D-optimization (dsaitekika.py)
2. **linear_analysis** - Linear regression analysis
3. **nonlinear_analysis** - ML models (RF, LightGBM, XGBoost) with Optuna
4. **classification** - Classification analysis

## Security Considerations

**Current Implementation (MVP):**
- No authentication (internal use only)
- Anyone with job_id can access results
- CORS allows all origins

**Production Recommendations:**
1. Add IAM authentication for API
2. Implement API keys or JWT tokens
3. Restrict CORS to specific origins
4. Enable S3 bucket encryption
5. Use VPC for EC2/RDS/ECS
6. Enable CloudWatch logging
7. Add request rate limiting

## Deployment Steps

### 1. Setup AWS Resources

```bash
# Create S3 buckets
aws s3 mb s3://mvp-inputs-bucket
aws s3 mb s3://mvp-outputs-bucket

# Create SQS queue
aws sqs create-queue --queue-name zerosec-jobs-queue

# Create RDS PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier zerosec-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username zerosec_admin \
    --master-user-password <password> \
    --allocated-storage 20
```

### 2. Deploy API Server (EC2)

```bash
# SSH into EC2 instance
ssh ec2-user@<ec2-ip>

# Clone repository
git clone <repo-url>
cd 0sec_dataanalysis_app/backend/api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Edit with actual AWS values

# Run API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Initialize Database

```bash
cd backend/migrations
python create_tables.py
```

### 4. Deploy Worker (ECS)

```bash
# Build Docker image
cd backend/worker
docker build -t zerosec-worker .

# Push to ECR
aws ecr create-repository --repository-name zerosec-worker
docker tag zerosec-worker:latest <account>.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest
docker push <account>.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest

# Create ECS task definition and service (via AWS Console or CLI)
```

### 5. Configure Frontend

```bash
cd frontend
cp .env.example .env
nano .env  # Set API_BASE_URL to EC2 public IP
```

## Monitoring

- **CloudWatch Logs**: API and Worker logs
- **CloudWatch Metrics**: Custom metrics for job processing
- **RDS Performance Insights**: Database performance
- **X-Ray**: Request tracing (optional)

## Testing

```bash
# Test API health
curl http://<ec2-ip>:8000/health

# Test frontend client
python -c "from frontend.api_client import create_client; client = create_client('http://<ec2-ip>:8000/api/v1'); print(client)"
```

## Troubleshooting

1. **Job stuck in queued**: Check SQS queue and worker logs
2. **Upload fails**: Verify S3 bucket permissions
3. **Database errors**: Check RDS security groups and credentials
4. **Worker crashes**: Check CloudWatch logs and memory limits

## Future Enhancements

1. Add authentication (Cognito/IAM)
2. Implement job cancellation
3. Add job priority queues
4. Implement batch job submission
5. Add WebSocket for real-time progress
6. Create web frontend (React/Vue)
7. Add job result caching
8. Implement auto-scaling for workers
