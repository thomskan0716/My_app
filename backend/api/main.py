"""
FastAPI EC2 API Server
Production-grade REST API for job management.
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import logging
import uuid
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError
import json

from backend.shared.models import (
    PresignUploadRequest,
    PresignUploadResponse,
    CreateJobRequest,
    CreateJobResponse,
    JobStatusResponse,
    JobArtifactsResponse,
    ArtifactInfo,
    PresignDownloadRequest,
    PresignDownloadResponse,
    HealthCheckResponse,
    ErrorResponse,
    SQSJobMessage,
    JobStatus,
    FileType,
)
from backend.shared.config import get_aws_config, get_api_config
from backend.shared.database import (
    DatabaseManager,
    JobRepository,
    ArtifactRepository,
    init_database,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
db_manager: Optional[DatabaseManager] = None
job_repo: Optional[JobRepository] = None
artifact_repo: Optional[ArtifactRepository] = None
s3_client = None
sqs_client = None
aws_config = None
api_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_manager, job_repo, artifact_repo, s3_client, sqs_client, aws_config, api_config
    
    # Startup
    logger.info("Starting EC2 API server...")
    
    try:
        # Load configurations
        aws_config = get_aws_config()
        api_config = get_api_config()
        
        # Initialize database
        db_manager = init_database()
        job_repo = JobRepository(db_manager)
        artifact_repo = ArtifactRepository(db_manager)
        
        # Test database connection
        if not db_manager.test_connection():
            raise RuntimeError("Database connection failed")
        
        # Initialize AWS clients
        s3_client = boto3.client('s3', region_name=aws_config.S3_REGION)
        sqs_client = boto3.client('sqs', region_name=aws_config.S3_REGION)
        
        logger.info("EC2 API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EC2 API server...")


# Create FastAPI app
app = FastAPI(
    title="0.00sec Analysis API",
    description="Production API for data analysis job management",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Exception Handlers =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            message=str(exc.detail),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow()
        ).dict()
    )


# ===== Health Check =====

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    db_connected = db_manager.test_connection() if db_manager else False
    
    return HealthCheckResponse(
        status="healthy" if db_connected else "unhealthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        database_connected=db_connected
    )


# ===== Presigned Upload =====

@app.post("/api/v1/presign/upload", response_model=PresignUploadResponse)
async def presign_upload(request: PresignUploadRequest):
    """
    Generate presigned URL for file upload.
    
    Flow:
    1. Generate unique job_id
    2. Create S3 key: inputs/{job_id}/{filename}
    3. Generate presigned PUT URL
    4. Return job_id, bucket, key, and URL
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create S3 key
        bucket = aws_config.S3_INPUTS_BUCKET
        key = f"inputs/{job_id}/{request.filename}"
        
        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket,
                'Key': key,
                'ContentType': request.content_type,
            },
            ExpiresIn=aws_config.PRESIGNED_URL_EXPIRATION
        )
        
        logger.info(f"Generated presigned upload URL for job {job_id}")
        
        return PresignUploadResponse(
            job_id=job_id,
            bucket=bucket,
            key=key,
            url=url,
            expires_in=aws_config.PRESIGNED_URL_EXPIRATION
        )
        
    except ClientError as e:
        logger.error(f"S3 error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate presigned URL"
        )


# ===== Job Management =====

@app.post("/api/v1/jobs", response_model=CreateJobResponse)
async def create_job(request: CreateJobRequest):
    """
    Create a new analysis job.
    
    Flow:
    1. Validate job_id and parameters
    2. Insert job into database
    3. Send message to SQS
    4. Return job_id
    """
    try:
        # Validate parameters based on job type
        try:
            typed_params = request.get_typed_parameters()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameters: {e}"
            )
        
        # Create job in database
        job = job_repo.create_job(
            job_id=request.job_id,
            job_type=request.job_type.value,
            input_bucket=request.input_bucket,
            input_key=request.input_key,
            output_bucket=aws_config.S3_OUTPUTS_BUCKET,
            parameters=request.parameters
        )
        
        # Send SQS message
        sqs_message = SQSJobMessage(
            job_id=request.job_id,
            job_type=request.job_type,
            input={
                "bucket": request.input_bucket,
                "key": request.input_key
            },
            output={
                "bucket": aws_config.S3_OUTPUTS_BUCKET
            },
            parameters=request.parameters
        )
        
        sqs_client.send_message(
            QueueUrl=aws_config.SQS_QUEUE_URL,
            MessageBody=sqs_message.json()
        )
        
        logger.info(f"Created job {request.job_id} ({request.job_type.value})")
        
        return CreateJobResponse(
            job_id=request.job_id,
            status=JobStatus.QUEUED,
            message="Job queued successfully"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ClientError as e:
        logger.error(f"SQS error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue job"
        )


@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get job status.
    
    Returns current status, progress, and messages.
    """
    try:
        job = job_repo.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=JobStatus(job.status),
            progress_percent=job.progress_percent,
            status_message=job.status_message,
            updated_at=job.updated_at,
            error_message=job.error_message
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format"
        )


@app.get("/api/v1/jobs/{job_id}/artifacts", response_model=JobArtifactsResponse)
async def get_job_artifacts(job_id: str):
    """
    Get list of output artifacts for a job.
    
    Returns list of files available for download.
    """
    try:
        # Verify job exists
        job = job_repo.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Get artifacts
        artifacts = artifact_repo.list_artifacts(job_id)
        
        artifact_infos = [
            ArtifactInfo(
                file_key=a.file_key,
                file_name=a.file_name,
                file_type=FileType(a.file_type),
                size_bytes=a.size_bytes,
                created_at=a.created_at
            )
            for a in artifacts
        ]
        
        return JobArtifactsResponse(
            job_id=job_id,
            artifacts=artifact_infos
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format"
        )


# ===== Presigned Download =====

@app.post("/api/v1/presign/download", response_model=PresignDownloadResponse)
async def presign_download(request: PresignDownloadRequest):
    """
    Generate presigned URL for file download.
    
    Security: Validates that the file belongs to the job.
    """
    try:
        # Verify artifact exists and belongs to job
        artifact = artifact_repo.get_artifact(request.job_id, request.file_key)
        
        if not artifact:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact not found or access denied"
            )
        
        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': aws_config.S3_OUTPUTS_BUCKET,
                'Key': request.file_key,
            },
            ExpiresIn=aws_config.PRESIGNED_URL_EXPIRATION
        )
        
        logger.info(f"Generated presigned download URL for {request.file_key}")
        
        return PresignDownloadResponse(
            url=url,
            expires_in=aws_config.PRESIGNED_URL_EXPIRATION,
            file_name=artifact.file_name
        )
        
    except ClientError as e:
        logger.error(f"S3 error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate presigned URL"
        )


# ===== Admin Endpoints (Optional) =====

@app.get("/api/v1/admin/jobs", include_in_schema=False)
async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 100
):
    """List jobs (admin endpoint)"""
    jobs = job_repo.list_jobs(status=status, job_type=job_type, limit=limit)
    return [job.to_dict() for job in jobs]


if __name__ == "__main__":
    import uvicorn
    
    config = get_api_config()
    
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
