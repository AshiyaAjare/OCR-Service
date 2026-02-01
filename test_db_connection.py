#!/usr/bin/env python3
"""
Quick script to test database connection and verify table creation.
Run this before running migrations to ensure everything is configured correctly.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app.database import engine, Base
from app.config import settings
from app.models.db_models import (
    IngestedEmailModel,
    EmailAttachmentModel,
    ExtractedDocumentModel,
)

def test_connection():
    """Test database connection."""
    print("Testing database connection...")
    print(f"Database URL: {settings.DATABASE_URL.replace(settings.DATABASE_PASSWORD, '***')}")
    
    try:
        # Try to connect
        with engine.connect() as conn:
            print("✓ Database connection successful!")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_table_names():
    """Verify table names have the correct prefix."""
    print("\nVerifying table names...")
    prefix = settings.DATABASE_TABLE_PREFIX
    
    tables = {
        "IngestedEmailModel": IngestedEmailModel.__tablename__,
        "EmailAttachmentModel": EmailAttachmentModel.__tablename__,
        "ExtractedDocumentModel": ExtractedDocumentModel.__tablename__,
    }
    
    all_prefixed = True
    for model_name, table_name in tables.items():
        if table_name.startswith(prefix):
            print(f"✓ {model_name}: {table_name}")
        else:
            print(f"✗ {model_name}: {table_name} (missing prefix!)")
            all_prefixed = False
    
    return all_prefixed

def main():
    print("=" * 60)
    print("Database Connection Test")
    print("=" * 60)
    
    # Test connection
    if not test_connection():
        print("\nPlease check your database configuration in .env file.")
        sys.exit(1)
    
    # Test table names
    if not test_table_names():
        print("\nWarning: Some tables are missing the prefix!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All tests passed! You can now run migrations:")
    print("  alembic revision --autogenerate -m 'Initial migration'")
    print("  alembic upgrade head")
    print("=" * 60)

if __name__ == "__main__":
    main()

