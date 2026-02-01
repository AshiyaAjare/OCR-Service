"""Add enriched metadata columns to ocr_extracted_document

Revision ID: 73fd725c2805
Revises: 9263f5280eec
Create Date: 2025-12-01 17:44:29.724134

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '73fd725c2805'
down_revision = '9263f5280eec'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add nullable, indexed metadata columns to ocr_extracted_document
    op.add_column(
        'ocr_extracted_document',
        sa.Column('ticker', sa.String(length=100), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_ticker',
        'ocr_extracted_document',
        ['ticker'],
    )

    op.add_column(
        'ocr_extracted_document',
        sa.Column('company_name', sa.String(length=500), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_company_name',
        'ocr_extracted_document',
        ['company_name'],
    )

    op.add_column(
        'ocr_extracted_document',
        sa.Column('report_date', sa.Date(), nullable=True),
    )
    op.add_column(
        'ocr_extracted_document',
        sa.Column('period', sa.String(length=100), nullable=True),
    )

    op.add_column(
        'ocr_extracted_document',
        sa.Column('document_type', sa.String(length=100), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_document_type',
        'ocr_extracted_document',
        ['document_type'],
    )

    op.add_column(
        'ocr_extracted_document',
        sa.Column('source_domain', sa.String(length=255), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_source_domain',
        'ocr_extracted_document',
        ['source_domain'],
    )

    op.add_column(
        'ocr_extracted_document',
        sa.Column('language', sa.String(length=10), nullable=True),
    )
    op.add_column(
        'ocr_extracted_document',
        sa.Column('ocr_confidence', sa.Float(), nullable=True),
    )

    op.add_column(
        'ocr_extracted_document',
        sa.Column('ingestion_method', sa.String(length=50), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_ingestion_method',
        'ocr_extracted_document',
        ['ingestion_method'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_ocr_extracted_document_ingestion_method',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'ingestion_method')

    op.drop_column('ocr_extracted_document', 'ocr_confidence')
    op.drop_column('ocr_extracted_document', 'language')

    op.drop_index(
        'ix_ocr_extracted_document_source_domain',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'source_domain')

    op.drop_index(
        'ix_ocr_extracted_document_document_type',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'document_type')

    op.drop_column('ocr_extracted_document', 'period')
    op.drop_column('ocr_extracted_document', 'report_date')

    op.drop_index(
        'ix_ocr_extracted_document_company_name',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'company_name')

    op.drop_index(
        'ix_ocr_extracted_document_ticker',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'ticker')

