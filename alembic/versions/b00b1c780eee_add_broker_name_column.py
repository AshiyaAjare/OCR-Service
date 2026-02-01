"""add_broker_name_column

Revision ID: b00b1c780eee
Revises: 84f76bfd6b14
Create Date: 2025-12-05 15:41:51.382862

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b00b1c780eee'
down_revision = '84f76bfd6b14'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add broker_name column after company_name
    # Note: PostgreSQL doesn't support AFTER clause, so column will be added at the end
    # The model file should be updated to reflect the desired column order
    op.add_column(
        'ocr_extracted_document',
        sa.Column('broker_name', sa.String(length=500), nullable=True),
    )
    op.create_index(
        'ix_ocr_extracted_document_broker_name',
        'ocr_extracted_document',
        ['broker_name'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_ocr_extracted_document_broker_name',
        table_name='ocr_extracted_document',
    )
    op.drop_column('ocr_extracted_document', 'broker_name')

