"""convert timestamps to timezone aware

Revision ID: 7e9f088f7517
Revises: ec85c9059503
Create Date: 2026-08-03 19:03:18.680902

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7e9f088f7517'
down_revision: Union[str, Sequence[str], None] = 'ec85c9059503'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "content_items",
        "timestamp",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        postgresql_using="timestamp AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        "prices",
        "timestamp",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        postgresql_using="timestamp AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        "signals",
        "timestamp",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        postgresql_using="timestamp AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        "sentiment",
        "created_at",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "sentiment",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        "signals",
        "timestamp",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        postgresql_using="timestamp AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        "prices",
        "timestamp",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        postgresql_using="timestamp AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
    op.alter_column(
        "content_items",
        "timestamp",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        postgresql_using="timestamp AT TIME ZONE 'UTC'",
        existing_nullable=False,
    )
