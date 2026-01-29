"""
Database models and connection management for RDS PostgreSQL.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, 
    Text, JSON, Boolean, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .config import get_aws_config

logger = logging.getLogger(__name__)

Base = declarative_base()


# ===== Database Models =====

class Job(Base):
    """Jobs table - tracks all analysis jobs"""
    __tablename__ = "jobs"
    
    job_id = Column(String(36), primary_key=True, index=True)  # UUID
    job_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    progress_percent = Column(Integer, default=0)
    status_message = Column(Text, default="")
    error_message = Column(Text, nullable=True)
    
    input_bucket = Column(String(255), nullable=False)
    input_key = Column(String(1024), nullable=False)
    output_bucket = Column(String(255), nullable=False)
    
    parameters = Column(JSON, default={})
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "status_message": self.status_message,
            "error_message": self.error_message,
            "input_bucket": self.input_bucket,
            "input_key": self.input_key,
            "output_bucket": self.output_bucket,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class JobArtifact(Base):
    """Job artifacts table - tracks output files"""
    __tablename__ = "job_artifacts"
    
    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), nullable=False, index=True)
    
    file_key = Column(String(1024), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "job_id": self.job_id,
            "file_key": self.file_key,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ===== Database Connection Manager =====

class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL. If None, uses config.
        """
        if database_url is None:
            config = get_aws_config()
            database_url = config.database_url
        
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set to True for SQL debugging
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database manager initialized")
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session context manager.
        
        Usage:
            with db_manager.get_session() as session:
                job = session.query(Job).filter_by(job_id=job_id).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


# ===== Database Operations =====

class JobRepository:
    """Repository for job operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_job(
        self,
        job_id: str,
        job_type: str,
        input_bucket: str,
        input_key: str,
        output_bucket: str,
        parameters: Dict[str, Any]
    ) -> Job:
        """Create a new job"""
        with self.db.get_session() as session:
            job = Job(
                job_id=job_id,
                job_type=job_type,
                status="queued",
                progress_percent=0,
                status_message="Queued",
                input_bucket=input_bucket,
                input_key=input_key,
                output_bucket=output_bucket,
                parameters=parameters,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            logger.info(f"Created job {job_id} ({job_type})")
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.db.get_session() as session:
            return session.query(Job).filter_by(job_id=job_id).first()
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress_percent: int = None,
        status_message: str = None,
        error_message: str = None
    ):
        """Update job status"""
        with self.db.get_session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            job.status = status
            if progress_percent is not None:
                job.progress_percent = progress_percent
            if status_message is not None:
                job.status_message = status_message
            if error_message is not None:
                job.error_message = error_message
            
            job.updated_at = datetime.utcnow()
            
            if status == "completed":
                job.completed_at = datetime.utcnow()
            
            session.commit()
            logger.info(f"Updated job {job_id}: {status} ({progress_percent}%)")
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with optional filtering"""
        with self.db.get_session() as session:
            query = session.query(Job)
            
            if status:
                query = query.filter_by(status=status)
            if job_type:
                query = query.filter_by(job_type=job_type)
            
            query = query.order_by(Job.created_at.desc()).limit(limit)
            
            return query.all()
    
    def delete_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than specified days"""
        with self.db.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            deleted = session.query(Job).filter(
                Job.created_at < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Deleted {deleted} jobs older than {days} days")
            return deleted


class ArtifactRepository:
    """Repository for artifact operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_artifact(
        self,
        job_id: str,
        file_key: str,
        file_name: str,
        file_type: str,
        size_bytes: int
    ) -> JobArtifact:
        """Create a new artifact"""
        with self.db.get_session() as session:
            artifact = JobArtifact(
                job_id=job_id,
                file_key=file_key,
                file_name=file_name,
                file_type=file_type,
                size_bytes=size_bytes,
            )
            session.add(artifact)
            session.commit()
            session.refresh(artifact)
            logger.info(f"Created artifact for job {job_id}: {file_name}")
            return artifact
    
    def list_artifacts(self, job_id: str) -> List[JobArtifact]:
        """List all artifacts for a job"""
        with self.db.get_session() as session:
            return session.query(JobArtifact).filter_by(job_id=job_id).all()
    
    def get_artifact(self, job_id: str, file_key: str) -> Optional[JobArtifact]:
        """Get specific artifact"""
        with self.db.get_session() as session:
            return session.query(JobArtifact).filter_by(
                job_id=job_id,
                file_key=file_key
            ).first()
    
    def delete_artifacts(self, job_id: str) -> int:
        """Delete all artifacts for a job"""
        with self.db.get_session() as session:
            deleted = session.query(JobArtifact).filter_by(job_id=job_id).delete()
            session.commit()
            logger.info(f"Deleted {deleted} artifacts for job {job_id}")
            return deleted


# ===== Utility Functions =====

from datetime import timedelta

def init_database(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Initialize database and create tables.
    
    Args:
        database_url: PostgreSQL connection URL
    
    Returns:
        DatabaseManager instance
    """
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager
