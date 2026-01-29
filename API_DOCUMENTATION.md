# API Documentation - 0.00sec Analysis API

## Base URL

```
http://<EC2_PUBLIC_IP>:8000/api/v1
```

## Authentication

**Current:** None (MVP - internal use only)  
**Future:** IAM authentication recommended for production

---

## Endpoints

### Health Check

#### `GET /health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T10:30:00Z",
  "version": "1.0.0",
  "database_connected": true
}
```

---

### Upload Workflow

#### `POST /api/v1/presign/upload`

Request presigned URL for file upload.

**Request Body:**
```json
{
  "filename": "test_data.csv",
  "content_type": "text/csv"
}
```

**Response:**
```json
{
  "job_id": "8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91",
  "bucket": "mvp-inputs-bucket",
  "key": "inputs/8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91/test_data.csv",
  "url": "https://mvp-inputs-bucket.s3.ap-northeast-1.amazonaws.com/...",
  "expires_in": 3600
}
```

**Next Step:** Use the `url` to upload file via HTTP PUT.

---

### Job Management

#### `POST /api/v1/jobs`

Create a new analysis job.

**Request Body:**
```json
{
  "job_id": "8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91",
  "job_type": "nonlinear_analysis",
  "input_bucket": "mvp-inputs-bucket",
  "input_key": "inputs/8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91/test_data.csv",
  "parameters": {
    "target_columns": ["摩耗量", "上面ダレ量"],
    "models_to_use": ["random_forest", "lightgbm", "xgboost"],
    "outer_splits": 10,
    "n_trials": 50,
    "enable_shap": true,
    "enable_pareto": true
  }
}
```

**Job Types:**
- `optimization` - D-optimization
- `linear_analysis` - Linear regression
- `nonlinear_analysis` - ML models with Optuna
- `classification` - Classification analysis

**Parameters by Job Type:**

##### Optimization Parameters
```json
{
  "objective": "minimize_wear",
  "bounds": {
    "speed": [1000, 5000],
    "feed": [0.1, 1.0]
  },
  "iterations": 50,
  "random_seed": 42,
  "num_points": 15
}
```

##### Linear Analysis Parameters
```json
{
  "target_column": "摩耗量",
  "feature_columns": ["送り速度", "回転速度", "切込量"],
  "standardize": true,
  "train_test_split": 0.2,
  "cv_splits": 5
}
```

##### Nonlinear Analysis Parameters
```json
{
  "target_columns": ["摩耗量", "上面ダレ量", "側面ダレ量"],
  "models_to_use": ["random_forest", "lightgbm", "xgboost"],
  "outer_splits": 10,
  "n_trials": 50,
  "train_test_split": 0.2,
  "enable_shap": true,
  "enable_pareto": true,
  "hyperparams": {}
}
```

##### Classification Parameters
```json
{
  "target_column": "バリ除去",
  "positive_label": "YES",
  "models_to_use": ["lightgbm", "random_forest"],
  "outer_splits": 10,
  "n_trials_inner": 50,
  "train_test_split": 0.2,
  "brush_type": "A13",
  "material": "Steel",
  "wire_length": 75,
  "wire_count": 6
}
```

**Response:**
```json
{
  "job_id": "8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91",
  "status": "queued",
  "message": "Job queued successfully"
}
```

---

#### `GET /api/v1/jobs/{job_id}/status`

Get job status and progress.

**Response:**
```json
{
  "job_id": "8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91",
  "status": "running",
  "progress_percent": 45,
  "status_message": "Training models (fold 5/10)",
  "updated_at": "2026-01-29T10:35:00Z",
  "error_message": null
}
```

**Status Values:**
- `queued` - Job is waiting in queue
- `running` - Job is being processed
- `uploading_outputs` - Uploading results to S3
- `completed` - Job completed successfully
- `failed` - Job failed with error
- `cancelled` - Job was cancelled

---

### Results & Download

#### `GET /api/v1/jobs/{job_id}/artifacts`

List output files for completed job.

**Response:**
```json
{
  "job_id": "8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91",
  "artifacts": [
    {
      "file_key": "outputs/8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91/summary.xlsx",
      "file_name": "summary.xlsx",
      "file_type": "xlsx",
      "size_bytes": 182394,
      "created_at": "2026-01-29T10:40:00Z"
    },
    {
      "file_key": "outputs/8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91/plots.png",
      "file_name": "plots.png",
      "file_type": "png",
      "size_bytes": 45678,
      "created_at": "2026-01-29T10:40:01Z"
    }
  ]
}
```

---

#### `POST /api/v1/presign/download`

Request presigned URL for file download.

**Request Body:**
```json
{
  "job_id": "8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91",
  "file_key": "outputs/8d8a4d9a-0e86-4d3b-9d4e-2a0a6f3a4d91/summary.xlsx"
}
```

**Response:**
```json
{
  "url": "https://mvp-outputs-bucket.s3.ap-northeast-1.amazonaws.com/...",
  "expires_in": 3600,
  "file_name": "summary.xlsx"
}
```

**Next Step:** Use the `url` to download file via HTTP GET.

---

## Error Responses

All errors follow this format:

```json
{
  "error": "ValidationError",
  "message": "Invalid job_id format",
  "details": {
    "field": "job_id",
    "expected": "UUID"
  },
  "timestamp": "2026-01-29T10:45:00Z"
}
```

**Common Error Codes:**

| Code | Error | Description |
|------|-------|-------------|
| 400 | BadRequest | Invalid request parameters |
| 404 | NotFound | Resource not found |
| 500 | InternalServerError | Server error |

---

## Complete Workflow Example

### 1. Upload File

```python
import requests

# Step 1: Request upload URL
response = requests.post(
    "http://ec2-ip:8000/api/v1/presign/upload",
    json={
        "filename": "data.csv",
        "content_type": "text/csv"
    }
)
data = response.json()
job_id = data["job_id"]
upload_url = data["url"]

# Step 2: Upload file
with open("data.csv", "rb") as f:
    requests.put(
        upload_url,
        data=f.read(),
        headers={"Content-Type": "text/csv"}
    )
```

### 2. Submit Job

```python
response = requests.post(
    "http://ec2-ip:8000/api/v1/jobs",
    json={
        "job_id": job_id,
        "job_type": "nonlinear_analysis",
        "input_bucket": data["bucket"],
        "input_key": data["key"],
        "parameters": {
            "target_columns": ["摩耗量"],
            "models_to_use": ["lightgbm"],
            "outer_splits": 5,
            "n_trials": 20
        }
    }
)
```

### 3. Poll Status

```python
import time

while True:
    response = requests.get(
        f"http://ec2-ip:8000/api/v1/jobs/{job_id}/status"
    )
    status = response.json()
    
    print(f"{status['progress_percent']}% - {status['status_message']}")
    
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error_message']}")
        break
    
    time.sleep(2)
```

### 4. Download Results

```python
# Get artifacts list
response = requests.get(
    f"http://ec2-ip:8000/api/v1/jobs/{job_id}/artifacts"
)
artifacts = response.json()["artifacts"]

# Download each file
for artifact in artifacts:
    # Get download URL
    response = requests.post(
        "http://ec2-ip:8000/api/v1/presign/download",
        json={
            "job_id": job_id,
            "file_key": artifact["file_key"]
        }
    )
    download_url = response.json()["url"]
    
    # Download file
    response = requests.get(download_url)
    with open(artifact["file_name"], "wb") as f:
        f.write(response.content)
```

---

## Rate Limits

**Current:** None (MVP)  
**Recommended for Production:** 60 requests/minute per client

---

## Support

For API issues or questions, contact the development team.
