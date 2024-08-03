"""Manual migration to update done column and add autoincrement to violence_segment id

Revision ID: e2118bdc2cc7
Revises:
Create Date: 2024-08-03 07:14:09.945087

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = 'e2118bdc2cc7'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # evidence_entity 테이블의 done 필드를 tinyint로 변경
    op.alter_column('EvidenceEntity', 'done', existing_type=mysql.BIT, type_=sa.SmallInteger)

    # violence_segment 테이블의 id 필드를 자동 증가로 설정
    op.execute('ALTER TABLE violence_segment MODIFY id INT AUTO_INCREMENT')

def downgrade():
    # 변경 사항을 되돌리는 작업
    op.alter_column('EvidenceEntity', 'done', existing_type=sa.SmallInteger, type_=mysql.BIT)
    op.execute('ALTER TABLE violence_segment MODIFY id INT')
