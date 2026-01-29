"""
ECS Worker - Polls SQS and processes analysis jobs.
Production-grade worker with error handling and progress tracking.
"""
import logging
import os
import time
import signal
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import traceback

import boto3
from botocore.exceptions import ClientError
import pandas as pd

from backend.shared.models import (
    SQSJobMessage,
    JobType,
    JobStatus,
    OptimizationParameters,
    LinearAnalysisParameters,
    NonlinearAnalysisParameters,
    ClassificationParameters,
)
from backend.shared.config import get_aws_config, get_worker_config
from backend.shared.database import (
    DatabaseManager,
    JobRepository,
    ArtifactRepository,
    init_database,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)


class JobProcessor:
    """Handles job processing logic"""
    
    def __init__(
        self,
        job_id: str,
        job_type: JobType,
        input_path: Path,
        output_dir: Path,
        parameters: Dict[str, Any],
        job_repo: JobRepository,
        artifact_repo: ArtifactRepository,
        s3_client,
        output_bucket: str
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.input_path = input_path
        self.output_dir = output_dir
        self.parameters = parameters
        self.job_repo = job_repo
        self.artifact_repo = artifact_repo
        self.s3_client = s3_client
        self.output_bucket = output_bucket
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def progress_callback(self, percent: int, message: str):
        """
        Progress callback for analysis modules.
        Updates job status in database.
        """
        try:
            self.job_repo.update_job_status(
                job_id=self.job_id,
                status=JobStatus.RUNNING.value,
                progress_percent=max(1, min(99, percent)),  # Keep 1-99 during processing
                status_message=message
            )
            logger.info(f"Job {self.job_id}: {percent}% - {message}")
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
    
    def run(self):
        """Execute the job based on type"""
        try:
            logger.info(f"Starting job {self.job_id} ({self.job_type.value})")
            
            # Update status to running
            self.progress_callback(1, "Starting analysis")
            
            # Dispatch to appropriate handler
            if self.job_type == JobType.OPTIMIZATION:
                self._run_optimization()
            elif self.job_type == JobType.LINEAR_ANALYSIS:
                self._run_linear_analysis()
            elif self.job_type == JobType.NONLINEAR_ANALYSIS:
                self._run_nonlinear_analysis()
            elif self.job_type == JobType.CLASSIFICATION:
                self._run_classification()
            else:
                raise ValueError(f"Unknown job type: {self.job_type}")
            
            # Upload outputs
            self._upload_outputs()
            
            # Mark as completed
            self.job_repo.update_job_status(
                job_id=self.job_id,
                status=JobStatus.COMPLETED.value,
                progress_percent=100,
                status_message="Completed successfully"
            )
            
            logger.info(f"Job {self.job_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Job failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Job {self.job_id} failed: {error_msg}")
            
            self.job_repo.update_job_status(
                job_id=self.job_id,
                status=JobStatus.FAILED.value,
                progress_percent=0,
                status_message="Failed",
                error_message=error_msg
            )
            
            raise
    
    def _run_optimization(self):
        """Run D-optimization"""
        from backend.worker.job_runners import run_optimization
        
        params = OptimizationParameters(**self.parameters)
        
        run_optimization(
            input_file=self.input_path,
            output_dir=self.output_dir,
            parameters=params,
            progress_callback=self.progress_callback
        )
    
    def _run_linear_analysis(self):
        """Run linear analysis"""
        from backend.worker.job_runners import run_linear_analysis
        
        params = LinearAnalysisParameters(**self.parameters)
        
        run_linear_analysis(
            input_file=self.input_path,
            output_dir=self.output_dir,
            parameters=params,
            progress_callback=self.progress_callback
        )
    
    def _run_nonlinear_analysis(self):
        """Run nonlinear analysis"""
        from backend.worker.job_runners import run_nonlinear_analysis
        
        params = NonlinearAnalysisParameters(**self.parameters)
        
        run_nonlinear_analysis(
            input_file=self.input_path,
            output_dir=self.output_dir,
            parameters=params,
            progress_callback=self.progress_callback
        )
    
    def _run_classification(self):
        """Run classification analysis"""
        from backend.worker.job_runners import run_classification
        
        params = ClassificationParameters(**self.parameters)
        
        run_classification(
            input_file=self.input_path,
            output_dir=self.output_dir,
            parameters=params,
            progress_callback=self.progress_callback
        )
    
    def _upload_outputs(self):
        """Upload all output files to S3 and register artifacts"""
        self.progress_callback(90, "Uploading results to S3")
        
        output_files = list(self.output_dir.rglob("*"))
        output_files = [f for f in output_files if f.is_file()]
        
        if not output_files:
            logger.warning(f"No output files found for job {self.job_id}")
            return
        
        logger.info(f"Uploading {len(output_files)} files for job {self.job_id}")
        
        for i, file_path in enumerate(output_files):
            try:
                # Calculate S3 key
                relative_path = file_path.relative_to(self.output_dir)
                s3_key = f"outputs/{self.job_id}/{relative_path}"
                
                # Upload to S3
                self.s3_client.upload_file(
                    str(file_path),
                    self.output_bucket,
                    s3_key
                )
                
                # Determine file type
                file_type = self._get_file_type(file_path)
                
                # Register artifact in database
                self.artifact_repo.create_artifact(
                    job_id=self.job_id,
                    file_key=s3_key,
                    file_name=file_path.name,
                    file_type=file_type,
                    size_bytes=file_path.stat().st_size
                )
                
                logger.info(f"Uploaded {file_path.name} -> {s3_key}")
                
                # Update progress
                progress = 90 + int((i + 1) / len(output_files) * 9)
                self.progress_callback(progress, f"Uploading outputs ({i+1}/{len(output_files)})")
                
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")
                # Continue with other files
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        ext = file_path.suffix.lower()
        
        type_map = {
            '.xlsx': 'xlsx',
            '.xls': 'xlsx',
            '.csv': 'csv',
            '.png': 'png',
            '.jpg': 'png',
            '.jpeg': 'png',
            '.json': 'json',
            '.db': 'db',
            '.html': 'html',
            '.pdf': 'pdf',
        }
        
        return type_map.get(ext, 'xlsx')  # Default to xlsx


class Worker:
    """Main worker class - polls SQS and processes jobs"""
    
    def __init__(self):
        self.running = True
        self.aws_config = get_aws_config()
        self.worker_config = get_worker_config()
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.aws_config.S3_REGION)
        self.sqs_client = boto3.client('sqs', region_name=self.aws_config.S3_REGION)
        
        # Initialize database
        self.db_manager = init_database()
        self.job_repo = JobRepository(self.db_manager)
        self.artifact_repo = ArtifactRepository(self.db_manager)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"Worker {self.worker_config.WORKER_ID} initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main worker loop"""
        logger.info("Worker started, polling for jobs...")
        
        while self.running:
            try:
                self._poll_and_process()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(5)  # Brief pause before retrying
        
        logger.info("Worker stopped")
    
    def _poll_and_process(self):
        """Poll SQS for messages and process them"""
        try:
            # Long poll SQS
            response = self.sqs_client.receive_message(
                QueueUrl=self.aws_config.SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=self.aws_config.SQS_WAIT_TIME_SECONDS,
                VisibilityTimeout=self.aws_config.SQS_VISIBILITY_TIMEOUT
            )
            
            messages = response.get('Messages', [])
            
            if not messages:
                # No messages, continue polling
                return
            
            for message in messages:
                try:
                    # Process the message
                    self._process_message(message)
                    
                    # Delete message from queue on success
                    self.sqs_client.delete_message(
                        QueueUrl=self.aws_config.SQS_QUEUE_URL,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    
                    logger.info(f"Message processed and deleted successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to process message: {e}", exc_info=True)
                    # Don't delete - let it retry or go to DLQ
                    
        except ClientError as e:
            logger.error(f"SQS error: {e}")
    
    def _process_message(self, message: Dict[str, Any]):
        """Process a single SQS message"""
        # Parse message body
        body = json.loads(message['Body'])
        job_message = SQSJobMessage(**body)
        
        logger.info(f"Processing job {job_message.job_id} ({job_message.job_type.value})")
        
        # Create temp directory for this job
        temp_dir = Path(self.worker_config.TEMP_DIR) / job_message.job_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download input file from S3
            input_file = temp_dir / "input_data.csv"
            
            logger.info(f"Downloading input: s3://{job_message.input['bucket']}/{job_message.input['key']}")
            
            self.s3_client.download_file(
                job_message.input['bucket'],
                job_message.input['key'],
                str(input_file)
            )
            
            # Create output directory
            output_dir = temp_dir / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            # Process the job
            processor = JobProcessor(
                job_id=job_message.job_id,
                job_type=job_message.job_type,
                input_path=input_file,
                output_dir=output_dir,
                parameters=job_message.parameters,
                job_repo=self.job_repo,
                artifact_repo=self.artifact_repo,
                s3_client=self.s3_client,
                output_bucket=job_message.output['bucket']
            )
            
            processor.run()
            
        finally:
            # Cleanup temp directory
            if self.worker_config.CLEANUP_TEMP_ON_SUCCESS:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")


def main():
    """Main entry point"""
    logger.info("Starting ECS Worker...")
    
    try:
        worker = Worker()
        worker.run()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
