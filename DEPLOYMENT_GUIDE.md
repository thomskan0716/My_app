# Deployment Guide - 0.00sec AWS Backend

This guide walks through deploying the refactored 0.00sec system to AWS.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed (for ECS worker)
- Python 3.10+ installed
- Git repository with the codebase

---

## Phase 1: AWS Resource Setup

### 1.1 Create S3 Buckets

```bash
# Create inputs bucket
aws s3 mb s3://mvp-inputs-bucket --region ap-northeast-1

# Create outputs bucket
aws s3 mb s3://mvp-outputs-bucket --region ap-northeast-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
    --bucket mvp-inputs-bucket \
    --versioning-configuration Status=Enabled

aws s3api put-bucket-versioning \
    --bucket mvp-outputs-bucket \
    --versioning-configuration Status=Enabled

# Enable encryption (recommended)
aws s3api put-bucket-encryption \
    --bucket mvp-inputs-bucket \
    --server-side-encryption-configuration '{
      "Rules": [{
        "ApplyServerSideEncryptionByDefault": {
          "SSEAlgorithm": "AES256"
        }
      }]
    }'

aws s3api put-bucket-encryption \
    --bucket mvp-outputs-bucket \
    --server-side-encryption-configuration '{
      "Rules": [{
        "ApplyServerSideEncryptionByDefault": {
          "SSEAlgorithm": "AES256"
        }
      }]
    }'
```

### 1.2 Create SQS Queue

```bash
# Create main queue
aws sqs create-queue \
    --queue-name zerosec-jobs-queue \
    --region ap-northeast-1 \
    --attributes '{
      "VisibilityTimeout": "900",
      "MessageRetentionPeriod": "86400",
      "ReceiveMessageWaitTimeSeconds": "20"
    }'

# Create Dead Letter Queue (DLQ)
aws sqs create-queue \
    --queue-name zerosec-jobs-dlq \
    --region ap-northeast-1

# Get DLQ ARN
DLQ_ARN=$(aws sqs get-queue-attributes \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-dlq \
    --attribute-names QueueArn \
    --query 'Attributes.QueueArn' \
    --output text)

# Configure DLQ for main queue
aws sqs set-queue-attributes \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-queue \
    --attributes "{
      \"RedrivePolicy\": \"{\\\"deadLetterTargetArn\\\":\\\"$DLQ_ARN\\\",\\\"maxReceiveCount\\\":\\\"3\\\"}\"
    }"
```

### 1.3 Create RDS PostgreSQL Database

```bash
# Create DB subnet group (if not exists)
aws rds create-db-subnet-group \
    --db-subnet-group-name zerosec-db-subnet \
    --db-subnet-group-description "Subnet group for 0sec DB" \
    --subnet-ids subnet-xxxxx subnet-yyyyy \
    --region ap-northeast-1

# Create security group for RDS
aws ec2 create-security-group \
    --group-name zerosec-rds-sg \
    --description "Security group for 0sec RDS" \
    --vpc-id vpc-xxxxx \
    --region ap-northeast-1

# Allow PostgreSQL access (from EC2 and ECS)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 5432 \
    --source-group sg-yyyyy  # EC2/ECS security group

# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier zerosec-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.4 \
    --master-username zerosec_admin \
    --master-user-password 'YOUR_SECURE_PASSWORD' \
    --allocated-storage 20 \
    --storage-type gp3 \
    --db-subnet-group-name zerosec-db-subnet \
    --vpc-security-group-ids sg-xxxxx \
    --backup-retention-period 7 \
    --publicly-accessible false \
    --region ap-northeast-1

# Wait for DB to be available (takes ~10 minutes)
aws rds wait db-instance-available \
    --db-instance-identifier zerosec-db
```

---

## Phase 2: Deploy EC2 API Server

### 2.1 Launch EC2 Instance

```bash
# Create security group for EC2
aws ec2 create-security-group \
    --group-name zerosec-api-sg \
    --description "Security group for 0sec API" \
    --vpc-id vpc-xxxxx

# Allow HTTP traffic (port 8000)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Allow SSH (for deployment)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 22 \
    --cidr YOUR_IP/32

# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0bba69335379e17f8 \  # Amazon Linux 2023
    --instance-type t3.small \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxx \
    --subnet-id subnet-xxxxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=zerosec-api}]'
```

### 2.2 Install API Server

```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>

# Update system
sudo yum update -y

# Install Python 3.10
sudo yum install python3.10 python3.10-pip git -y

# Clone repository
git clone <YOUR_REPO_URL>
cd 0sec_dataanalysis_app/backend/api

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env
```

**Edit .env file:**
```bash
S3_INPUTS_BUCKET=mvp-inputs-bucket
S3_OUTPUTS_BUCKET=mvp-outputs-bucket
S3_REGION=ap-northeast-1

SQS_QUEUE_URL=https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-queue
SQS_DLQ_URL=https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-dlq

RDS_HOST=zerosec-db.xxxxxx.ap-northeast-1.rds.amazonaws.com
RDS_PORT=5432
RDS_DATABASE=zerosec_db
RDS_USERNAME=zerosec_admin
RDS_PASSWORD=YOUR_SECURE_PASSWORD

API_HOST=0.0.0.0
API_PORT=8000
```

### 2.3 Initialize Database

```bash
cd ../migrations
python create_tables.py
```

### 2.4 Run API Server

**Option A: Direct run (for testing)**
```bash
cd ../api
python main.py
```

**Option B: Production with systemd**

Create service file:
```bash
sudo nano /etc/systemd/system/zerosec-api.service
```

Content:
```ini
[Unit]
Description=0.00sec API Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/0sec_dataanalysis_app/backend/api
Environment="PATH=/home/ec2-user/0sec_dataanalysis_app/backend/api/venv/bin"
ExecStart=/home/ec2-user/0sec_dataanalysis_app/backend/api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable zerosec-api
sudo systemctl start zerosec-api
sudo systemctl status zerosec-api
```

### 2.5 Test API

```bash
# From your local machine
curl http://<EC2_PUBLIC_IP>:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database_connected": true,
  ...
}
```

---

## Phase 3: Deploy ECS Worker

### 3.1 Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
    --repository-name zerosec-worker \
    --region ap-northeast-1

# Get login credentials
aws ecr get-login-password --region ap-northeast-1 | \
    docker login --username AWS --password-stdin \
    YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com
```

### 3.2 Build and Push Docker Image

```bash
cd backend/worker

# Build image
docker build -t zerosec-worker .

# Tag image
docker tag zerosec-worker:latest \
    YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest

# Push image
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest
```

### 3.3 Create ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name zerosec-cluster \
    --region ap-northeast-1
```

### 3.4 Create Task Definition

Create file `task-definition.json`:

```json
{
  "family": "zerosec-worker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/zerosecWorkerRole",
  "containerDefinitions": [
    {
      "name": "zerosec-worker",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest",
      "essential": true,
      "environment": [
        {"name": "S3_INPUTS_BUCKET", "value": "mvp-inputs-bucket"},
        {"name": "S3_OUTPUTS_BUCKET", "value": "mvp-outputs-bucket"},
        {"name": "S3_REGION", "value": "ap-northeast-1"},
        {"name": "SQS_QUEUE_URL", "value": "https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-queue"},
        {"name": "RDS_HOST", "value": "zerosec-db.xxxxxx.ap-northeast-1.rds.amazonaws.com"},
        {"name": "RDS_PORT", "value": "5432"},
        {"name": "RDS_DATABASE", "value": "zerosec_db"},
        {"name": "RDS_USERNAME", "value": "zerosec_admin"},
        {"name": "RDS_PASSWORD", "value": "YOUR_SECURE_PASSWORD"},
        {"name": "WORKER_ID", "value": "worker-1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/zerosec-worker",
          "awslogs-region": "ap-northeast-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register task:
```bash
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json
```

### 3.5 Create ECS Service

```bash
aws ecs create-service \
    --cluster zerosec-cluster \
    --service-name zerosec-worker-service \
    --task-definition zerosec-worker \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

---

## Phase 4: Configure Frontend

### 4.1 Install Frontend Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 4.2 Configure API Endpoint

```bash
cp .env.example .env
nano .env
```

Edit:
```bash
API_BASE_URL=http://<EC2_PUBLIC_IP>:8000/api/v1
```

### 4.3 Test Frontend Client

```python
from frontend.api_client import create_client

client = create_client("http://<EC2_PUBLIC_IP>:8000/api/v1")
print("Client created successfully!")
```

---

## Phase 5: Verification & Testing

### 5.1 End-to-End Test

```bash
# Create test file
echo "test data" > test.csv

# Run complete workflow
python test_workflow.py
```

### 5.2 Monitor Logs

```bash
# API logs
sudo journalctl -u zerosec-api -f

# Worker logs (CloudWatch)
aws logs tail /ecs/zerosec-worker --follow
```

---

## Maintenance

### Update API Server

```bash
ssh ec2-user@<EC2_PUBLIC_IP>
cd 0sec_dataanalysis_app
git pull
sudo systemctl restart zerosec-api
```

### Update Worker

```bash
cd backend/worker
docker build -t zerosec-worker .
docker tag zerosec-worker:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/zerosec-worker:latest

# Update ECS service
aws ecs update-service \
    --cluster zerosec-cluster \
    --service zerosec-worker-service \
    --force-new-deployment
```

---

## Troubleshooting

### API Server Issues

```bash
# Check service status
sudo systemctl status zerosec-api

# Check logs
sudo journalctl -u zerosec-api -n 100

# Test database connection
psql -h zerosec-db.xxxxxx.ap-northeast-1.rds.amazonaws.com \
     -U zerosec_admin -d zerosec_db
```

### Worker Issues

```bash
# Check ECS service
aws ecs describe-services \
    --cluster zerosec-cluster \
    --services zerosec-worker-service

# Check task status
aws ecs list-tasks --cluster zerosec-cluster

# View logs
aws logs tail /ecs/zerosec-worker --since 1h
```

### SQS Issues

```bash
# Check queue depth
aws sqs get-queue-attributes \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-queue \
    --attribute-names ApproximateNumberOfMessages

# Check DLQ
aws sqs get-queue-attributes \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/YOUR_ACCOUNT_ID/zerosec-jobs-dlq \
    --attribute-names ApproximateNumberOfMessages
```

---

## Security Checklist

- [ ] RDS is in private subnet
- [ ] Security groups restrict access
- [ ] Database password is strong
- [ ] S3 buckets have encryption enabled
- [ ] IAM roles follow least privilege
- [ ] CloudWatch logging is enabled
- [ ] Backup retention is configured

---

## Cost Optimization

- Use t3.micro for API server (upgrade if needed)
- Use Fargate Spot for workers (50-70% cost savings)
- Set S3 lifecycle policies to delete old files
- Use RDS automated backups (7 days retention)
- Monitor CloudWatch costs

---

## Next Steps

1. Add authentication (Cognito/IAM)
2. Set up CloudWatch alarms
3. Configure auto-scaling for workers
4. Add CDN (CloudFront) for static assets
5. Implement job cancellation feature
