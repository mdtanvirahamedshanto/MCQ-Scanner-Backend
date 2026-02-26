"""SQLAlchemy database models for OptiMark."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """User model for teachers/admins/superadmins."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    google_sub = Column(String(255), unique=True, index=True, nullable=True)
    role = Column(String(20), default="teacher")  # teacher | admin | superadmin
    is_active = Column(Boolean, default=True)
    is_subscribed = Column(Boolean, default=False)
    subscription_plan = Column(String(50), nullable=True)
    institution_name = Column(String(255), nullable=True)
    address = Column(String(255), nullable=True)
    tokens = Column(Integer, default=500)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    exams = relationship("Exam", back_populates="teacher", foreign_keys="[Exam.teacher_id]")
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    wallet = relationship("TokenWallet", back_populates="user", uselist=False)
    subscriptions = relationship(
        "Subscription",
        back_populates="user",
        foreign_keys="[Subscription.user_id]",
    )
    ledger_entries = relationship("TokenLedger", back_populates="user")
    scan_batches = relationship("ScanBatch", back_populates="user")
    scan_jobs = relationship("ScanJob", back_populates="user")
    scanned_sheets = relationship("ScannedSheet", back_populates="user")
    pending_payments = relationship(
        "PendingPayment",
        back_populates="user",
        foreign_keys="[PendingPayment.user_id]",
    )


class UserProfile(Base):
    """Extended user onboarding profile."""

    __tablename__ = "user_profiles"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    institute_name = Column(String(255), nullable=True)
    institute_address = Column(String(500), nullable=True)
    phone = Column(String(50), nullable=True)
    website = Column(String(255), nullable=True)
    profile_completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")


class Plan(Base):
    """Subscription plan catalog."""

    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    price_amount = Column(Float, nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")
    billing_cycle = Column(String(20), nullable=False, default="monthly")
    tokens_included = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    subscriptions = relationship("Subscription", back_populates="plan")


class Subscription(Base):
    """User subscription lifecycle."""

    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    plan_id = Column(Integer, ForeignKey("plans.id"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    provider = Column(String(20), nullable=False, default="manual")
    provider_ref = Column(String(255), nullable=True)
    starts_at = Column(DateTime, nullable=True)
    ends_at = Column(DateTime, nullable=True)
    activated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="subscriptions", foreign_keys=[user_id])
    plan = relationship("Plan", back_populates="subscriptions")

    __table_args__ = (
        Index("idx_sub_user_status", "user_id", "status"),
    )


class TokenWallet(Base):
    """Token balance per user."""

    __tablename__ = "token_wallets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    balance = Column(Integer, nullable=False, default=0)
    version = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="wallet")
    ledger_entries = relationship("TokenLedger", back_populates="wallet")


class TokenLedger(Base):
    """Immutable wallet transaction ledger."""

    __tablename__ = "token_ledger"

    id = Column(Integer, primary_key=True, index=True)
    wallet_id = Column(Integer, ForeignKey("token_wallets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    direction = Column(String(10), nullable=False)  # credit | debit
    reason = Column(String(50), nullable=False)
    reference_type = Column(String(50), nullable=False)
    reference_id = Column(String(255), nullable=False)
    delta = Column(Integer, nullable=False)
    before_balance = Column(Integer, nullable=False)
    after_balance = Column(Integer, nullable=False)
    idempotency_key = Column(String(255), nullable=True, unique=True)
    metadata_json = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    wallet = relationship("TokenWallet", back_populates="ledger_entries")
    user = relationship("User", back_populates="ledger_entries")

    __table_args__ = (
        UniqueConstraint("user_id", "reason", "reference_type", "reference_id", name="uq_ledger_reason_ref"),
        Index("idx_ledger_user_created", "user_id", "created_at"),
    )


class Exam(Base):
    """Exam model - supports legacy and v1 fields."""

    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Legacy fields
    title = Column(String(255), nullable=True)
    subject_code = Column(String(50), nullable=True)

    # v1 fields
    exam_name = Column(String(255), nullable=True)
    subject_name = Column(String(255), nullable=True)
    has_set = Column(Boolean, default=False)
    set_count = Column(Integer, default=1)
    total_questions = Column(Integer, default=60)
    options_per_question = Column(Integer, default=4)
    negative_marking = Column(Boolean, default=False)
    negative_value = Column(Float, default=0.0)
    mark_per_question = Column(Float, default=1.0)
    status = Column(String(20), default="draft")

    date_created = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    teacher = relationship("User", back_populates="exams", foreign_keys=[teacher_id])
    exam_sets = relationship("ExamSet", back_populates="exam", cascade="all, delete-orphan")
    answer_keys = relationship("AnswerKey", back_populates="exam", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="exam", cascade="all, delete-orphan")
    scan_batches = relationship("ScanBatch", back_populates="exam", cascade="all, delete-orphan")
    scan_jobs = relationship("ScanJob", back_populates="exam", cascade="all, delete-orphan")
    scanned_sheets = relationship("ScannedSheet", back_populates="exam", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_exams_user_created", "teacher_id", "date_created"),
    )


class ExamSet(Base):
    """Per-exam set labels (A/B/C/D or 1/2/3/4)."""

    __tablename__ = "exam_sets"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    set_label = Column(String(20), nullable=False)
    set_order = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    exam = relationship("Exam", back_populates="exam_sets")
    answer_keys = relationship("AnswerKey", back_populates="exam_set")

    __table_args__ = (
        UniqueConstraint("exam_id", "set_label", name="uq_exam_set_label"),
    )


class AnswerKey(Base):
    """Answer keys for legacy and v1 workflows."""

    __tablename__ = "answer_keys"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    set_id = Column(Integer, ForeignKey("exam_sets.id"), nullable=True)

    # Legacy fields
    set_code = Column(String(10), nullable=False, default="A")
    answers = Column(JSONB, nullable=False, default=dict)

    # v1 fields
    mapping = Column(JSONB, nullable=True)
    version = Column(Integer, default=1)
    is_published = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    exam = relationship("Exam", back_populates="answer_keys")
    exam_set = relationship("ExamSet", back_populates="answer_keys")

    __table_args__ = (
        UniqueConstraint("exam_id", "set_id", "version", name="uq_exam_set_version"),
    )


class Result(Base):
    """Legacy grading result table."""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    roll_number = Column(String(50), nullable=False)
    set_code = Column(String(10), nullable=False)
    marks_obtained = Column(Integer, default=0)
    wrong_answers = Column(JSONB, default=lambda: [])
    percentage = Column(Float, default=0.0)
    image_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    exam = relationship("Exam", back_populates="results")


class PendingPayment(Base):
    """Legacy manual payment submissions awaiting admin approval."""

    __tablename__ = "pending_payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    plan_id = Column(String(50), nullable=False)
    amount = Column(Float, nullable=False)
    payment_method = Column(String(30), nullable=False)
    transaction_id = Column(String(100), nullable=False)
    sender_name = Column(String(255), nullable=False)
    sender_phone = Column(String(50), nullable=True)
    sender_email = Column(String(255), nullable=True)
    status = Column(String(20), default="pending")
    admin_notes = Column(String(500), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="pending_payments", foreign_keys=[user_id])


class ScanBatch(Base):
    """Bulk upload batch container for scan jobs."""

    __tablename__ = "scan_batches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    idempotency_key = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="created")
    total_files = Column(Integer, nullable=False, default=0)
    processed_files = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="scan_batches")
    exam = relationship("Exam", back_populates="scan_batches")
    jobs = relationship("ScanJob", back_populates="batch", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("user_id", "exam_id", "idempotency_key", name="uq_scan_batch_idempotency"),
    )


class ScanJob(Base):
    """Per-file scanning job."""

    __tablename__ = "scan_jobs"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("scan_batches.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    source_file_key = Column(String(500), nullable=False)
    source_file_hash = Column(String(128), nullable=True)
    status = Column(String(20), nullable=False, default="queued")
    task_id = Column(String(255), nullable=True)
    attempts = Column(Integer, default=0)
    error_code = Column(String(100), nullable=True)
    error_message = Column(String(1000), nullable=True)
    token_charged = Column(Boolean, default=False)
    token_ledger_id = Column(Integer, ForeignKey("token_ledger.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    batch = relationship("ScanBatch", back_populates="jobs")
    user = relationship("User", back_populates="scan_jobs")
    exam = relationship("Exam", back_populates="scan_jobs")
    token_ledger = relationship("TokenLedger")
    scanned_sheet = relationship("ScannedSheet", back_populates="scan_job", uselist=False)

    __table_args__ = (
        Index("idx_jobs_batch_status", "batch_id", "status"),
        Index("idx_jobs_user_created", "user_id", "created_at"),
    )


class ScannedSheet(Base):
    """Final scored sheet record."""

    __tablename__ = "scanned_sheets"

    id = Column(Integer, primary_key=True, index=True)
    scan_job_id = Column(Integer, ForeignKey("scan_jobs.id"), nullable=False, unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    student_identifier = Column(String(100), nullable=True)
    detected_identifier = Column(String(100), nullable=True)
    set_label_detected = Column(String(20), nullable=True)
    set_label_final = Column(String(20), nullable=True)
    manual_override = Column(Boolean, default=False)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    unanswered_count = Column(Integer, default=0)
    invalid_count = Column(Integer, default=0)
    raw_score = Column(Float, default=0.0)
    final_score = Column(Float, default=0.0)
    percentage = Column(Float, default=0.0)
    evaluated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    scan_job = relationship("ScanJob", back_populates="scanned_sheet")
    user = relationship("User", back_populates="scanned_sheets")
    exam = relationship("Exam", back_populates="scanned_sheets")
    answers = relationship("SheetAnswer", back_populates="sheet", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sheet_exam_eval", "exam_id", "evaluated_at"),
        Index("idx_sheet_user_eval", "user_id", "evaluated_at"),
    )


class SheetAnswer(Base):
    """Per-question answer details."""

    __tablename__ = "sheet_answers"

    id = Column(Integer, primary_key=True, index=True)
    sheet_id = Column(Integer, ForeignKey("scanned_sheets.id"), nullable=False)
    question_no = Column(Integer, nullable=False)
    selected_option = Column(String(10), nullable=True)
    correct_option = Column(String(10), nullable=True)
    status = Column(String(20), nullable=False)
    mark_awarded = Column(Float, nullable=False, default=0.0)
    is_overridden = Column(Boolean, default=False)

    sheet = relationship("ScannedSheet", back_populates="answers")

    __table_args__ = (
        UniqueConstraint("sheet_id", "question_no", name="uq_sheet_question"),
    )


class AuditLog(Base):
    """Lightweight audit log table."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(255), nullable=True)
    ip = Column(String(64), nullable=True)
    user_agent = Column(String(500), nullable=True)
    metadata_json = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_audit_actor_created", "actor_user_id", "created_at"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
    )
