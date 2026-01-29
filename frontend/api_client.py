"""
Frontend API Client
Production-grade client for communicating with EC2 API.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import from local models (frontend-only, no backend dependency)
from frontend.models import (
    PresignUploadRequest,
    PresignUploadResponse,
    PresignDownloadRequest,
    PresignDownloadResponse,
    CreateJobRequest,
    CreateJobResponse,
    JobStatusResponse,
    JobArtifactsResponse,
    ArtifactInfo,
    JobType,
    JobStatus,
)

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Base exception for API client errors"""
    pass


class JobTimeoutError(APIClientError):
    """Raised when job polling times out"""
    pass


class APIClient:
    """
    Client for communicating with the EC2 API server.
    
    Features:
    - Automatic retries with exponential backoff
    - Request/response validation
    - Progress tracking
    - Error handling
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.3
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of API (e.g., "http://ec2-host:8000/api/v1")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Create session with retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"API client initialized with base URL: {self.base_url}")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            json_data: JSON request body
            params: Query parameters
        
        Returns:
            Response JSON
        
        Raises:
            APIClientError: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise APIClientError(f"API request failed: {e}")
    
    # ===== Upload Workflow =====
    
    def request_upload_url(
        self,
        filename: str,
        content_type: str = "text/csv"
    ) -> PresignUploadResponse:
        """
        Request presigned upload URL.
        
        Args:
            filename: Name of file to upload
            content_type: MIME type
        
        Returns:
            PresignUploadResponse with job_id and upload URL
        """
        request = PresignUploadRequest(
            filename=filename,
            content_type=content_type
        )
        
        response_data = self._request(
            method="POST",
            endpoint="/presign/upload",
            json_data=request.dict()
        )
        
        response = PresignUploadResponse(**response_data)
        
        logger.info(f"Obtained upload URL for job {response.job_id}")
        
        return response
    
    def upload_file(
        self,
        file_path: Path,
        upload_url: str,
        content_type: str = "text/csv",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Upload file to S3 using presigned URL.
        
        Args:
            file_path: Local file path
            upload_url: Presigned PUT URL
            content_type: MIME type
            progress_callback: Optional callback(bytes_uploaded, total_bytes)
        """
        file_size = file_path.stat().st_size
        
        logger.info(f"Uploading file: {file_path} ({file_size} bytes)")
        
        try:
            with open(file_path, 'rb') as f:
                # Read file data
                file_data = f.read()
            
            # Upload with PUT request
            response = requests.put(
                upload_url,
                data=file_data,
                headers={'Content-Type': content_type},
                timeout=300  # 5 minutes for upload
            )
            
            response.raise_for_status()
            
            if progress_callback:
                progress_callback(file_size, file_size)
            
            logger.info("File uploaded successfully")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"File upload failed: {e}")
            raise APIClientError(f"File upload failed: {e}")
    
    # ===== Job Management =====
    
    def create_job(
        self,
        job_id: str,
        job_type: JobType,
        input_bucket: str,
        input_key: str,
        parameters: Dict[str, Any]
    ) -> CreateJobResponse:
        """
        Create a new analysis job.
        
        Args:
            job_id: Job ID from upload
            job_type: Type of analysis
            input_bucket: S3 bucket name
            input_key: S3 object key
            parameters: Job-specific parameters
        
        Returns:
            CreateJobResponse
        """
        request = CreateJobRequest(
            job_id=job_id,
            job_type=job_type,
            input_bucket=input_bucket,
            input_key=input_key,
            parameters=parameters
        )
        
        response_data = self._request(
            method="POST",
            endpoint="/jobs",
            json_data=request.dict()
        )
        
        response = CreateJobResponse(**response_data)
        
        logger.info(f"Created job {job_id} ({job_type.value})")
        
        return response
    
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        """
        Get current job status.
        
        Args:
            job_id: Job ID
        
        Returns:
            JobStatusResponse
        """
        response_data = self._request(
            method="GET",
            endpoint=f"/jobs/{job_id}/status"
        )
        
        return JobStatusResponse(**response_data)
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 2,
        max_wait_minutes: int = 120,
        progress_callback: Optional[Callable[[JobStatusResponse], None]] = None
    ) -> JobStatusResponse:
        """
        Poll job status until completion or failure.
        
        Args:
            job_id: Job ID
            poll_interval: Seconds between polls
            max_wait_minutes: Maximum time to wait
            progress_callback: Optional callback(status_response)
        
        Returns:
            Final JobStatusResponse
        
        Raises:
            JobTimeoutError: If max_wait_minutes exceeded
            APIClientError: If job failed
        """
        start_time = datetime.now()
        max_wait = timedelta(minutes=max_wait_minutes)
        
        logger.info(f"Waiting for job {job_id} (max {max_wait_minutes} minutes)")
        
        while True:
            # Check timeout
            elapsed = datetime.now() - start_time
            if elapsed > max_wait:
                raise JobTimeoutError(
                    f"Job {job_id} exceeded maximum wait time of {max_wait_minutes} minutes"
                )
            
            # Get status
            status = self.get_job_status(job_id)
            
            # Call progress callback
            if progress_callback:
                progress_callback(status)
            
            # Check if done
            if status.status == JobStatus.COMPLETED:
                logger.info(f"Job {job_id} completed successfully")
                return status
            
            elif status.status == JobStatus.FAILED:
                error_msg = status.error_message or "Unknown error"
                logger.error(f"Job {job_id} failed: {error_msg}")
                raise APIClientError(f"Job failed: {error_msg}")
            
            elif status.status == JobStatus.CANCELLED:
                logger.warning(f"Job {job_id} was cancelled")
                raise APIClientError("Job was cancelled")
            
            # Continue polling
            time.sleep(poll_interval)
    
    # ===== Artifacts =====
    
    def list_artifacts(self, job_id: str) -> JobArtifactsResponse:
        """
        Get list of job output artifacts.
        
        Args:
            job_id: Job ID
        
        Returns:
            JobArtifactsResponse
        """
        response_data = self._request(
            method="GET",
            endpoint=f"/jobs/{job_id}/artifacts"
        )
        
        return JobArtifactsResponse(**response_data)
    
    def request_download_url(
        self,
        job_id: str,
        file_key: str
    ) -> PresignDownloadResponse:
        """
        Request presigned download URL.
        
        Args:
            job_id: Job ID
            file_key: S3 key of file to download
        
        Returns:
            PresignDownloadResponse
        """
        request = PresignDownloadRequest(
            job_id=job_id,
            file_key=file_key
        )
        
        response_data = self._request(
            method="POST",
            endpoint="/presign/download",
            json_data=request.dict()
        )
        
        return PresignDownloadResponse(**response_data)
    
    def download_file(
        self,
        download_url: str,
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Download file from S3 using presigned URL.
        
        Args:
            download_url: Presigned GET URL
            output_path: Local path to save file
            progress_callback: Optional callback(bytes_downloaded, total_bytes)
        """
        logger.info(f"Downloading file to: {output_path}")
        
        try:
            # Stream download
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded, total_size)
            
            logger.info("File downloaded successfully")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"File download failed: {e}")
            raise APIClientError(f"File download failed: {e}")
    
    def download_all_artifacts(
        self,
        job_id: str,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Path]:
        """
        Download all artifacts for a job.
        
        Args:
            job_id: Job ID
            output_dir: Directory to save files
            progress_callback: Optional callback(current_file, total_files, filename)
        
        Returns:
            List of downloaded file paths
        """
        # Get artifacts list
        artifacts_response = self.list_artifacts(job_id)
        artifacts = artifacts_response.artifacts
        
        if not artifacts:
            logger.warning(f"No artifacts found for job {job_id}")
            return []
        
        logger.info(f"Downloading {len(artifacts)} artifacts for job {job_id}")
        
        downloaded_files = []
        
        for i, artifact in enumerate(artifacts):
            # Update progress
            if progress_callback:
                progress_callback(i + 1, len(artifacts), artifact.file_name)
            
            # Get download URL
            download_response = self.request_download_url(job_id, artifact.file_key)
            
            # Download file
            output_path = output_dir / artifact.file_name
            self.download_file(download_response.url, output_path)
            
            downloaded_files.append(output_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {output_dir}")
        
        return downloaded_files
    
    # ===== High-level Workflow =====
    
    def submit_and_wait(
        self,
        file_path: Path,
        job_type: JobType,
        parameters: Dict[str, Any],
        output_dir: Path,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> List[Path]:
        """
        Complete workflow: upload, submit, wait, download.
        
        Args:
            file_path: Input file to upload
            job_type: Type of analysis
            parameters: Job parameters
            output_dir: Directory to save results
            progress_callback: Optional callback(message, progress_percent)
        
        Returns:
            List of downloaded output files
        """
        try:
            # Step 1: Request upload URL
            if progress_callback:
                progress_callback("Requesting upload URL", 5)
            
            upload_response = self.request_upload_url(
                filename=file_path.name,
                content_type="text/csv"
            )
            
            job_id = upload_response.job_id
            
            # Step 2: Upload file
            if progress_callback:
                progress_callback("Uploading input file", 10)
            
            self.upload_file(file_path, upload_response.url)
            
            # Step 3: Create job
            if progress_callback:
                progress_callback("Creating analysis job", 15)
            
            self.create_job(
                job_id=job_id,
                job_type=job_type,
                input_bucket=upload_response.bucket,
                input_key=upload_response.key,
                parameters=parameters
            )
            
            # Step 4: Wait for completion
            def status_callback(status: JobStatusResponse):
                if progress_callback:
                    # Map 0-100% job progress to 15-90% overall progress
                    overall_progress = 15 + int(status.progress_percent * 0.75)
                    progress_callback(status.status_message, overall_progress)
            
            self.wait_for_job(
                job_id=job_id,
                progress_callback=status_callback
            )
            
            # Step 5: Download results
            if progress_callback:
                progress_callback("Downloading results", 90)
            
            output_files = self.download_all_artifacts(job_id, output_dir)
            
            if progress_callback:
                progress_callback("Completed", 100)
            
            return output_files
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise


# ===== Convenience Functions =====

def create_client(api_base_url: str) -> APIClient:
    """
    Create a configured API client.
    
    Args:
        api_base_url: Base URL of the API
    
    Returns:
        APIClient instance
    """
    return APIClient(base_url=api_base_url)
