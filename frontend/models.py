"""
Shared data models for frontend API client.
This is a COPY of backend/shared/models.py to avoid backend dependencies on client machines.

NOTE: Keep this in sync with backend/shared/models.py when making changes.
"""
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Types of analysis jobs"""
    OPTIMIZATION = "optimization"
    LINEAR_ANALYSIS = "linear_analysis"
    NONLINEAR_ANALYSIS = "nonlinear_analysis"
    CLASSIFICATION = "classification"


class JobStatus(str, Enum):
    """Job execution states"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UPLOADING_OUTPUTS = "uploading_outputs"


class FileType(str, Enum):
    """Output file type enumeration"""
    XLSX = "xlsx"
    CSV = "csv"
    PNG = "png"
    JSON = "json"
    DB = "db"
    HTML = "html"
    PDF = "pdf"


# ============================================================================
# Request Models (Client → API)
# ============================================================================

class PresignUploadRequest(BaseModel):
    """Request presigned URL for file upload"""
    filename: str = Field(..., description="File name to upload", min_length=1)
    file_size_mb: Optional[float] = Field(None, description="File size in MB (for validation)")
    content_type: Optional[str] = Field(default="text/csv", description="MIME type")


class PresignDownloadRequest(BaseModel):
    """Request for presigned download URL"""
    job_id: str = Field(..., description="Job ID")
    file_key: str = Field(..., description="S3 key of file to download")


class CreateJobRequest(BaseModel):
    """Request to create a new analysis job"""
    job_id: str = Field(..., description="Job ID from presign upload")
    job_type: JobType = Field(..., description="Type of analysis to run")
    input_bucket: str = Field(..., description="S3 bucket containing input file")
    input_key: str = Field(..., description="S3 key of input file")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job-specific parameters")


# ============================================================================
# Response Models (API → Client)
# ============================================================================

class PresignUploadResponse(BaseModel):
    """Response with presigned upload URL"""
    job_id: str = Field(..., description="Generated job ID")
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key")
    url: str = Field(..., description="Presigned PUT URL")
    expires_in: int = Field(default=3600, description="URL expiration time in seconds")


class CreateJobResponse(BaseModel):
    """Response after job creation"""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Initial job status")
    message: str = Field(default="Job queued successfully", description="Status message")


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current job status")
    progress_percent: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    status_message: str = Field(default="", description="Human-readable status message")
    updated_at: datetime = Field(..., description="Last update timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ArtifactInfo(BaseModel):
    """Information about an output artifact"""
    file_key: str = Field(..., description="S3 key of the file")
    file_name: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="File type")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current job status")
    progress_percent: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    status_message: str = Field(default="", description="Human-readable status message")
    updated_at: datetime = Field(..., description="Last update timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobArtifactsResponse(BaseModel):
    """Response with job output artifacts"""
    job_id: str = Field(..., description="Job ID")
    artifacts: List[ArtifactInfo] = Field(default_factory=list, description="List of output files")


class PresignDownloadResponse(BaseModel):
    """Response with presigned download URL"""
    url: str = Field(..., description="Presigned GET URL")
    expires_in: int = Field(default=3600, description="URL expiration time in seconds")
    file_name: str = Field(..., description="Filename for download")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Worker-specific Models (SQS Messages)
# ============================================================================

class SQSJobMessage(BaseModel):
    """SQS message format for job processing"""
    job_id: str = Field(..., description="Job ID")
    job_type: JobType = Field(..., description="Job type")
    input: Dict[str, str] = Field(..., description="Input S3 location (bucket, key)")
    output: Dict[str, str] = Field(..., description="Output S3 location (bucket)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")


# ============================================================================
# Job-specific Parameter Models
# ============================================================================

class OptimizationParameters(BaseModel):
    """Parameters for optimization job"""
    objective: str = Field(default="minimize_wear", description="Optimization objective")
    bounds: Dict[str, List[float]] = Field(default_factory=dict, description="Parameter bounds")
    iterations: int = Field(default=50, ge=1, le=1000, description="Number of iterations")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    num_points: int = Field(default=15, ge=1, le=100, description="Number of points to generate")


class LinearAnalysisParameters(BaseModel):
    """Parameters for linear analysis job"""
    target_column: str = Field(..., description="Target column name")
    feature_columns: List[str] = Field(default_factory=list, description="Feature column names")
    standardize: bool = Field(default=True, description="Whether to standardize features")
    train_test_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Test set ratio")
    cv_splits: int = Field(default=5, ge=2, le=20, description="Cross-validation splits")


class NonlinearAnalysisParameters(BaseModel):
    """Parameters for nonlinear analysis job"""
    target_columns: List[str] = Field(default_factory=list, description="Target column names")
    models_to_use: List[str] = Field(
        default_factory=lambda: ["random_forest", "lightgbm", "xgboost"],
        description="Models to use"
    )
    outer_splits: int = Field(default=10, ge=2, le=20, description="Outer CV splits")
    n_trials: int = Field(default=50, ge=10, le=500, description="Optuna trials per fold")
    train_test_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Test set ratio")
    hyperparams: Dict[str, Any] = Field(default_factory=dict, description="Additional hyperparameters")
    enable_shap: bool = Field(default=True, description="Enable SHAP analysis")
    enable_pareto: bool = Field(default=True, description="Enable Pareto analysis")


class ClassificationParameters(BaseModel):
    """Parameters for classification job"""
    target_column: str = Field(..., description="Target column name")
    positive_label: str = Field(default="YES", description="Positive class label")
    models_to_use: List[str] = Field(
        default_factory=lambda: ["lightgbm", "random_forest"],
        description="Models to use"
    )
    outer_splits: int = Field(default=10, ge=2, le=20, description="Outer CV splits")
    n_trials_inner: int = Field(default=50, ge=10, le=500, description="Optuna trials")
    train_test_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Test set ratio")
    brush_type: Optional[str] = Field(default="A13", description="Brush type")
    material: Optional[str] = Field(default="Steel", description="Material")
    wire_length: Optional[int] = Field(default=75, ge=30, le=75, description="Wire length")
    wire_count: Optional[int] = Field(default=6, ge=1, description="Wire count")
