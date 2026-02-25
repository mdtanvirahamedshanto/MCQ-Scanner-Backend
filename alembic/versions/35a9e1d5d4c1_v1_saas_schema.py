"""Add v1 SaaS schema (profile, wallet, ledger, jobs, results)

Revision ID: 35a9e1d5d4c1
Revises: 24b76ed8cdff
Create Date: 2026-02-25

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "35a9e1d5d4c1"
down_revision: Union[str, None] = "24b76ed8cdff"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_sub", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=True, server_default=sa.text("true")))
    op.add_column("users", sa.Column("updated_at", sa.DateTime(), nullable=True))
    op.create_index("ix_users_google_sub", "users", ["google_sub"], unique=True)

    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("institute_name", sa.String(length=255), nullable=True),
        sa.Column("institute_address", sa.String(length=500), nullable=True),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("website", sa.String(length=255), nullable=True),
        sa.Column("profile_completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )

    op.create_table(
        "plans",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(length=100), nullable=False, unique=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("price_amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="USD"),
        sa.Column("billing_cycle", sa.String(length=20), nullable=False, server_default="monthly"),
        sa.Column("tokens_included", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=True, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )

    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("plan_id", sa.Integer(), sa.ForeignKey("plans.id"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("provider", sa.String(length=20), nullable=False, server_default="manual"),
        sa.Column("provider_ref", sa.String(length=255), nullable=True),
        sa.Column("starts_at", sa.DateTime(), nullable=True),
        sa.Column("ends_at", sa.DateTime(), nullable=True),
        sa.Column("activated_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_index("idx_sub_user_status", "subscriptions", ["user_id", "status"], unique=False)

    op.create_table(
        "token_wallets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False, unique=True),
        sa.Column("balance", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )

    op.create_table(
        "token_ledger",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("wallet_id", sa.Integer(), sa.ForeignKey("token_wallets.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("direction", sa.String(length=10), nullable=False),
        sa.Column("reason", sa.String(length=50), nullable=False),
        sa.Column("reference_type", sa.String(length=50), nullable=False),
        sa.Column("reference_id", sa.String(length=255), nullable=False),
        sa.Column("delta", sa.Integer(), nullable=False),
        sa.Column("before_balance", sa.Integer(), nullable=False),
        sa.Column("after_balance", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True, unique=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_index("idx_ledger_user_created", "token_ledger", ["user_id", "created_at"], unique=False)
    op.create_unique_constraint("uq_ledger_reason_ref", "token_ledger", ["user_id", "reason", "reference_type", "reference_id"])

    op.add_column("exams", sa.Column("user_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_exams_user_id", "exams", "users", ["user_id"], ["id"])
    op.add_column("exams", sa.Column("exam_name", sa.String(length=255), nullable=True))
    op.add_column("exams", sa.Column("subject_name", sa.String(length=255), nullable=True))
    op.add_column("exams", sa.Column("has_set", sa.Boolean(), nullable=True, server_default=sa.text("false")))
    op.add_column("exams", sa.Column("set_count", sa.Integer(), nullable=True, server_default="1"))
    op.add_column("exams", sa.Column("options_per_question", sa.Integer(), nullable=True, server_default="4"))
    op.add_column("exams", sa.Column("negative_marking", sa.Boolean(), nullable=True, server_default=sa.text("false")))
    op.add_column("exams", sa.Column("negative_value", sa.Float(), nullable=True, server_default="0"))
    op.add_column("exams", sa.Column("mark_per_question", sa.Float(), nullable=True, server_default="1"))
    op.add_column("exams", sa.Column("status", sa.String(length=20), nullable=True, server_default="draft"))
    op.add_column("exams", sa.Column("updated_at", sa.DateTime(), nullable=True))

    op.create_index("idx_exams_user_created", "exams", ["teacher_id", "date_created"], unique=False)

    op.create_table(
        "exam_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("exams.id"), nullable=False),
        sa.Column("set_label", sa.String(length=20), nullable=False),
        sa.Column("set_order", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_unique_constraint("uq_exam_set_label", "exam_sets", ["exam_id", "set_label"])

    op.add_column("answer_keys", sa.Column("set_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_answer_keys_set_id", "answer_keys", "exam_sets", ["set_id"], ["id"])
    op.add_column("answer_keys", sa.Column("mapping", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("answer_keys", sa.Column("version", sa.Integer(), nullable=True, server_default="1"))
    op.add_column("answer_keys", sa.Column("is_published", sa.Boolean(), nullable=True, server_default=sa.text("true")))
    op.add_column("answer_keys", sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()))
    op.add_column("answer_keys", sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()))
    op.create_unique_constraint("uq_exam_set_version", "answer_keys", ["exam_id", "set_id", "version"])

    op.create_table(
        "scan_batches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("exams.id"), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="created"),
        sa.Column("total_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_files", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_unique_constraint("uq_scan_batch_idempotency", "scan_batches", ["user_id", "exam_id", "idempotency_key"])

    op.create_table(
        "scan_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("batch_id", sa.Integer(), sa.ForeignKey("scan_batches.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("exams.id"), nullable=False),
        sa.Column("source_file_key", sa.String(length=500), nullable=False),
        sa.Column("source_file_hash", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("task_id", sa.String(length=255), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=1000), nullable=True),
        sa.Column("token_charged", sa.Boolean(), nullable=True, server_default=sa.text("false")),
        sa.Column("token_ledger_id", sa.Integer(), sa.ForeignKey("token_ledger.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_index("idx_jobs_batch_status", "scan_jobs", ["batch_id", "status"], unique=False)
    op.create_index("idx_jobs_user_created", "scan_jobs", ["user_id", "created_at"], unique=False)

    op.create_table(
        "scanned_sheets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("scan_job_id", sa.Integer(), sa.ForeignKey("scan_jobs.id"), nullable=False, unique=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("exam_id", sa.Integer(), sa.ForeignKey("exams.id"), nullable=False),
        sa.Column("student_identifier", sa.String(length=100), nullable=True),
        sa.Column("detected_identifier", sa.String(length=100), nullable=True),
        sa.Column("set_label_detected", sa.String(length=20), nullable=True),
        sa.Column("set_label_final", sa.String(length=20), nullable=True),
        sa.Column("manual_override", sa.Boolean(), nullable=True, server_default=sa.text("false")),
        sa.Column("correct_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("wrong_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("unanswered_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("invalid_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("raw_score", sa.Float(), nullable=True, server_default="0"),
        sa.Column("final_score", sa.Float(), nullable=True, server_default="0"),
        sa.Column("percentage", sa.Float(), nullable=True, server_default="0"),
        sa.Column("evaluated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_index("idx_sheet_exam_eval", "scanned_sheets", ["exam_id", "evaluated_at"], unique=False)
    op.create_index("idx_sheet_user_eval", "scanned_sheets", ["user_id", "evaluated_at"], unique=False)

    op.create_table(
        "sheet_answers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sheet_id", sa.Integer(), sa.ForeignKey("scanned_sheets.id"), nullable=False),
        sa.Column("question_no", sa.Integer(), nullable=False),
        sa.Column("selected_option", sa.String(length=10), nullable=True),
        sa.Column("correct_option", sa.String(length=10), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("mark_awarded", sa.Float(), nullable=False, server_default="0"),
        sa.Column("is_overridden", sa.Boolean(), nullable=True, server_default=sa.text("false")),
    )
    op.create_unique_constraint("uq_sheet_question", "sheet_answers", ["sheet_id", "question_no"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=100), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=True),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.create_index("idx_audit_actor_created", "audit_logs", ["actor_user_id", "created_at"], unique=False)
    op.create_index("idx_audit_entity", "audit_logs", ["entity_type", "entity_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_audit_entity", table_name="audit_logs")
    op.drop_index("idx_audit_actor_created", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_constraint("uq_sheet_question", "sheet_answers", type_="unique")
    op.drop_table("sheet_answers")

    op.drop_index("idx_sheet_user_eval", table_name="scanned_sheets")
    op.drop_index("idx_sheet_exam_eval", table_name="scanned_sheets")
    op.drop_table("scanned_sheets")

    op.drop_index("idx_jobs_user_created", table_name="scan_jobs")
    op.drop_index("idx_jobs_batch_status", table_name="scan_jobs")
    op.drop_table("scan_jobs")

    op.drop_constraint("uq_scan_batch_idempotency", "scan_batches", type_="unique")
    op.drop_table("scan_batches")

    op.drop_constraint("uq_exam_set_version", "answer_keys", type_="unique")
    op.drop_column("answer_keys", "updated_at")
    op.drop_column("answer_keys", "created_at")
    op.drop_column("answer_keys", "is_published")
    op.drop_column("answer_keys", "version")
    op.drop_column("answer_keys", "mapping")
    op.drop_constraint("fk_answer_keys_set_id", "answer_keys", type_="foreignkey")
    op.drop_column("answer_keys", "set_id")

    op.drop_constraint("uq_exam_set_label", "exam_sets", type_="unique")
    op.drop_table("exam_sets")

    op.drop_index("idx_exams_user_created", table_name="exams")
    op.drop_column("exams", "updated_at")
    op.drop_column("exams", "status")
    op.drop_column("exams", "mark_per_question")
    op.drop_column("exams", "negative_value")
    op.drop_column("exams", "negative_marking")
    op.drop_column("exams", "options_per_question")
    op.drop_column("exams", "set_count")
    op.drop_column("exams", "has_set")
    op.drop_column("exams", "subject_name")
    op.drop_column("exams", "exam_name")
    op.drop_constraint("fk_exams_user_id", "exams", type_="foreignkey")
    op.drop_column("exams", "user_id")

    op.drop_constraint("uq_ledger_reason_ref", "token_ledger", type_="unique")
    op.drop_index("idx_ledger_user_created", table_name="token_ledger")
    op.drop_table("token_ledger")

    op.drop_table("token_wallets")

    op.drop_index("idx_sub_user_status", table_name="subscriptions")
    op.drop_table("subscriptions")

    op.drop_table("plans")

    op.drop_table("user_profiles")

    op.drop_index("ix_users_google_sub", table_name="users")
    op.drop_column("users", "updated_at")
    op.drop_column("users", "is_active")
    op.drop_column("users", "google_sub")
