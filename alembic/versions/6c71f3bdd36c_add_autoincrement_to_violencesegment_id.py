"""Add autoincrement to ViolenceSegment id

Revision ID: 6c71f3bdd36c
Revises: 
Create Date: 2024-07-26 21:04:51.019008

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6c71f3bdd36c'
#이 아래만 복사 붙여넣기 하면 됨
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute('ALTER TABLE violence_segment MODIFY id INT AUTO_INCREMENT')


def downgrade():
    op.execute('ALTER TABLE violence_segment MODIFY id INT')