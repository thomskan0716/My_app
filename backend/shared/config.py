"""
Configuration management for backend components.
Handles AWS resources, database connections, and environment-specific settings.
"""
import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AWSConfig(BaseSettings):
    """AWS resource configuration"""
    # S3 Buckets
    S3_INPUTS_BUCKET: str = Field(..., env="S3_INPUTS_BUCKET")
    S3_OUTPUTS_BUCKET: str = Field(..., env="S3_OUTPUTS_BUCKET")
    S3_REGION: str = Field(default="ap-northeast-1", env="S3_REGION")
    
    # SQS
    SQS_QUEUE_URL: str = Field(..., env="SQS_QUEUE_URL")
    SQS_DLQ_URL: Optional[str] = Field(default=None, env="SQS_DLQ_URL")
    SQS_VISIBILITY_TIMEOUT: int = Field(default=900, env="SQS_VISIBILITY_TIMEOUT")  # 15 minutes
    SQS_WAIT_TIME_SECONDS: int = Field(default=20, env="SQS_WAIT_TIME_SECONDS")  # Long polling
    
    # RDS PostgreSQL
    RDS_HOST: str = Field(..., env="RDS_HOST")
    RDS_PORT: int = Field(default=5432, env="RDS_PORT")
    RDS_DATABASE: str = Field(default="zerosec_db", env="RDS_DATABASE")
    RDS_USERNAME: str = Field(..., env="RDS_USERNAME")
    RDS_PASSWORD: str = Field(..., env="RDS_PASSWORD")
    RDS_SSL_MODE: str = Field(default="require", env="RDS_SSL_MODE")
    
    # IAM Role (for ECS)
    AWS_ROLE_ARN: Optional[str] = Field(default=None, env="AWS_ROLE_ARN")
    
    # Presigned URL settings
    PRESIGNED_URL_EXPIRATION: int = Field(default=3600, env="PRESIGNED_URL_EXPIRATION")  # 1 hour
    
    class Config:
        env_file = ".env.aws"
        env_file_encoding = "utf-8"
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return (
            f"postgresql://{self.RDS_USERNAME}:{self.RDS_PASSWORD}@"
            f"{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DATABASE}"
            f"?sslmode={self.RDS_SSL_MODE}"
        )


class APIConfig(BaseSettings):
    """API server configuration"""
    # Server settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_VERSION: str = Field(default="v1", env="API_VERSION")
    
    # CORS settings
    CORS_ORIGINS: list = Field(default=["*"], env="CORS_ORIGINS")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Request limits
    MAX_UPLOAD_SIZE_MB: int = Field(default=100, env="MAX_UPLOAD_SIZE_MB")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    
    # Job limits
    MAX_JOBS_PER_USER: int = Field(default=10, env="MAX_JOBS_PER_USER")
    JOB_RETENTION_DAYS: int = Field(default=30, env="JOB_RETENTION_DAYS")
    
    class Config:
        env_file = ".env.api"
        env_file_encoding = "utf-8"
    
    @property
    def base_url(self) -> str:
        """Get base API URL path"""
        return f"/api/{self.API_VERSION}"


class WorkerConfig(BaseSettings):
    """Worker configuration"""
    
    # Worker behavior
    WORKER_ID: str = Field(default="worker-1", env="WORKER_ID")
    WORKER_CONCURRENCY: int = Field(default=1, env="WORKER_CONCURRENCY")  # Number of concurrent jobs
    WORKER_POLL_INTERVAL: int = Field(default=5, env="WORKER_POLL_INTERVAL")  # Seconds between polls
    
    # Processing settings
    MAX_PROCESSING_TIME_MINUTES: int = Field(default=60, env="MAX_PROCESSING_TIME_MINUTES")
    PROGRESS_UPDATE_INTERVAL: int = Field(default=5, env="PROGRESS_UPDATE_INTERVAL")  # Seconds
    
    # Retry settings
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    RETRY_BACKOFF_SECONDS: int = Field(default=30, env="RETRY_BACKOFF_SECONDS")
    
    # Temp directory for processing
    TEMP_DIR: str = Field(default="/tmp/zerosec", env="TEMP_DIR")
    CLEANUP_TEMP_ON_SUCCESS: bool = Field(default=True, env="CLEANUP_TEMP_ON_SUCCESS")
    CLEANUP_TEMP_ON_FAILURE: bool = Field(default=False, env="CLEANUP_TEMP_ON_FAILURE")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_TO_CLOUDWATCH: bool = Field(default=True, env="LOG_TO_CLOUDWATCH")
    
    # Memory monitoring
    MEMORY_LIMIT_MB: int = Field(default=4096, env="MEMORY_LIMIT_MB")
    MEMORY_CHECK_INTERVAL: int = Field(default=10, env="MEMORY_CHECK_INTERVAL")
    
    class Config:
        env_file = ".env.frontend"
        env_file_encoding = "utf-8"


class FrontendConfig(BaseSettings):
    """Frontend client configuration"""
    
    # API endpoint
    API_BASE_URL: str = Field(..., env="API_BASE_URL")  # e.g., "http://ec2-instance:8000/api/v1"
    
    # Polling settings
    STATUS_POLL_INTERVAL_SECONDS: int = Field(default=2, env="STATUS_POLL_INTERVAL_SECONDS")
    MAX_POLL_TIME_MINUTES: int = Field(default=120, env="MAX_POLL_TIME_MINUTES")  # 2 hours max
    
    # Request settings
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")
    
    # File handling
    DOWNLOAD_CHUNK_SIZE: int = Field(default=8192, env="DOWNLOAD_CHUNK_SIZE")  # 8KB chunks
    
    # UI settings
    ENABLE_PROGRESS_BAR: bool = Field(default=True, env="ENABLE_PROGRESS_BAR")
    ENABLE_CONSOLE_OUTPUT: bool = Field(default=True, env="ENABLE_CONSOLE_OUTPUT")
    
    class Config:
        env_file = ".env.frontend"
        env_file_encoding = "utf-8"


def get_aws_config() -> AWSConfig:
    """Get AWS configuration singleton"""
    return AWSConfig()


def get_api_config() -> APIConfig:
    """Get API configuration singleton"""
    return APIConfig()


def get_worker_config() -> WorkerConfig:
    """Get Worker configuration singleton"""
    return WorkerConfig()


def get_frontend_config() -> FrontendConfig:
    """Get Frontend configuration singleton"""
    return FrontendConfig()
