"""
Database migration script - Create initial tables.
Run this to set up the RDS database.
"""
from backend.shared.database import init_database
from backend.shared.config import get_aws_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Create database tables"""
    logger.info("Starting database migration...")
    
    # Initialize database
    config = get_aws_config()
    logger.info(f"Connecting to: {config.RDS_HOST}:{config.RDS_PORT}/{config.RDS_DATABASE}")
    
    db_manager = init_database()
    
    logger.info("Database tables created successfully!")
    
    # Test connection
    if db_manager.test_connection():
        logger.info("✓ Database connection test passed")
    else:
        logger.error("✗ Database connection test failed")
        return 1
    
    logger.info("Migration completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
