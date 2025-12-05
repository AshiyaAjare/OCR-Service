"""add_broker_estimate_column

Revision ID: 84f76bfd6b14
Revises: 26fc17420c9b
Create Date: 2025-12-04 23:23:50.086320

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '84f76bfd6b14'
down_revision = '26fc17420c9b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add broker_estimate column
    op.add_column(
        'ocr_extracted_document',
        sa.Column('broker_estimate', sa.Float(), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_broker_estimate',
        'ocr_extracted_document',
        ['broker_estimate'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_ocr_extracted_document_broker_estimate',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'broker_estimate')

