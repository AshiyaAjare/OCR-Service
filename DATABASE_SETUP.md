# Database Setup Guide

This project uses PostgreSQL with SQLAlchemy and Alembic for migrations. The database tables are prefixed with `ocr_` by default to avoid conflicts with the Django project's tables in the same database.

## Configuration

Database connection details are configured in `.env` file:

```bash
DATABASE_NAME=fortress_db
DATABASE_USER=ashiya
DATABASE_PASSWORD=123456
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_TABLE_PREFIX=ocr_  # Optional, defaults to 'ocr_'
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure PostgreSQL is running and the database `fortress_db` exists:
```bash
# Connect to PostgreSQL
psql -U ashiya -d fortress_db

# Verify the database exists
\l
```

## Running Migrations

### Initial Setup

1. Create the initial migration:
```bash
cd pdf_extractor
alembic revision --autogenerate -m "Initial migration"
```

2. Review the generated migration file in `alembic/versions/` to ensure it only creates your project's tables (prefixed with `ocr_`).

3. Apply the migration:
```bash
alembic upgrade head
```

### Future Migrations

1. After modifying models in `app/models/db_models.py`, create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```

2. Review and apply:
```bash
alembic upgrade head
```

### Rollback

To rollback the last migration:
```bash
alembic downgrade -1
```

## Safety Features

### Table Prefix
- All tables are prefixed with `ocr_` by default (configurable via `DATABASE_TABLE_PREFIX`)
- This ensures your tables won't conflict with Django project's tables
- Example: `ocr_ingested_email`, `ocr_extracted_document`

### Migration Safety
- Alembic only creates/modifies tables that match your SQLAlchemy models
- Django's tables are untouched because they're not defined in your models
- Always review migration files before applying them

## Using the Database in Code

### Example: Saving an Ingested Email

```python
from app.database import get_db
from app.models.db_models import IngestedEmailModel
from sqlalchemy.orm import Session
from fastapi import Depends

@app.post("/ingest-email")
def ingest_email(db: Session = Depends(get_db)):
    email = IngestedEmailModel(
        message_id="<test@example.com>",
        subject="Test Email",
        sender="sender@example.com",
        date="Mon, 1 Dec 2025 12:00:00 +0000",
        body_text="Email body",
        urls=["https://example.com"]
    )
    db.add(email)
    db.commit()
    db.refresh(email)
    return email
```

### Example: Querying Emails

```python
from app.database import get_db
from app.models.db_models import IngestedEmailModel
from sqlalchemy.orm import Session
from fastapi import Depends

@app.get("/emails")
def get_emails(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    emails = db.query(IngestedEmailModel).offset(skip).limit(limit).all()
    return emails
```

## Database Models

### IngestedEmailModel
- Stores ingested emails from Gmail
- Table: `ocr_ingested_email`
- Fields: message_id, subject, sender, date, body_text, body_html, urls, etc.

### EmailAttachmentModel
- Stores email attachment metadata
- Table: `ocr_email_attachment`
- Fields: email_id, filename, content_type, file_size, file_path

### ExtractedDocumentModel
- Stores extracted documents (PDFs, etc.)
- Table: `ocr_extracted_document`
- Fields: source_email_id, source_url, file_path, extraction results, etc.

## Verifying Tables

After running migrations, verify your tables exist:

```sql
-- Connect to PostgreSQL
psql -U ashiya -d fortress_db

-- List all tables with 'ocr_' prefix
\dt ocr_*

-- Or list all tables to see both Django and OCR tables
\dt
```

You should see:
- Django project's tables (without prefix)
- Your OCR project's tables (with `ocr_` prefix)

## Troubleshooting

### Migration fails with "table already exists"
- This usually means the table was created manually or by a previous migration
- Check if the table exists: `\dt ocr_*`
- If it exists and you want to recreate it, drop it first: `DROP TABLE ocr_ingested_email;`

### Connection refused
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check connection details in `.env` file
- Test connection: `psql -U ashiya -d fortress_db -h localhost`

### Permission denied
- Ensure the database user has CREATE TABLE permissions
- Grant permissions if needed: `GRANT ALL PRIVILEGES ON DATABASE fortress_db TO ashiya;`

