"""Initial schema with User, Exam, AnswerKey, Result

Revision ID: 001
Revises:
Create Date: 2024-01-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("email", sa.String(255), unique=True, index=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_subscribed", sa.Boolean(), default=False),
        sa.Column("subscription_plan", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "exams",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("teacher_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("subject_code", sa.String(50), nullable=False),
        sa.Column("total_questions", sa.Integer(), default=60),
        sa.Column("date_created", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "answer_keys",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("exams.id"), nullable=False),
        sa.Column("set_code", sa.String(10), nullable=False),
        sa.Column("answers", JSONB, nullable=False),
    )
    op.create_table(
        "results",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("exams.id"), nullable=False),
        sa.Column("roll_number", sa.String(50), nullable=False),
        sa.Column("set_code", sa.String(10), nullable=False),
        sa.Column("marks_obtained", sa.Integer(), default=0),
        sa.Column("wrong_answers", JSONB, server_default="[]"),
        sa.Column("percentage", sa.Float(), default=0.0),
        sa.Column("image_url", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("results")
    op.drop_table("answer_keys")
    op.drop_table("exams")
    op.drop_table("users")
