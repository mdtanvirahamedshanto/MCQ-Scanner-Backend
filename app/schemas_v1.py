"""Pydantic schemas for v1 SaaS APIs."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, model_validator


class MessageResponse(BaseModel):
    message: str


class GoogleExchangeRequest(BaseModel):
    id_token: str = Field(..., min_length=10)


class AuthUserResponse(BaseModel):
    id: int
    email: EmailStr
    role: str
    profile_completed: bool


class AuthExchangeResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserResponse


class AuthSessionResponse(BaseModel):
    authenticated: bool
    user: AuthUserResponse


class ProfileResponse(BaseModel):
    institute_name: Optional[str] = None
    institute_address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    profile_completed: bool
    profile_completed_at: Optional[datetime] = None


class ProfileUpdateRequest(BaseModel):
    institute_name: str = Field(..., min_length=1, max_length=255)
    institute_address: str = Field(..., min_length=1, max_length=500)
    phone: Optional[str] = Field(default=None, max_length=50)
    website: Optional[str] = Field(default=None, max_length=255)


class ProfileStatusResponse(BaseModel):
    profile_completed: bool


class PlanResponse(BaseModel):
    id: int
    code: str
    name: str
    price_amount: float
    currency: str
    billing_cycle: str
    tokens_included: int
    is_active: bool


class PlanCreateRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    price_amount: float = Field(..., ge=0)
    currency: str = Field(default="BDT", min_length=3, max_length=3)
    billing_cycle: str = Field(default="monthly", max_length=20)
    tokens_included: int = Field(..., ge=0)


class PlanUpdateRequest(BaseModel):
    code: Optional[str] = Field(default=None, min_length=1, max_length=100)
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    price_amount: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)
    billing_cycle: Optional[str] = Field(default=None, max_length=20)
    tokens_included: Optional[int] = Field(default=None, ge=0)
    is_active: Optional[bool] = None


class SubscriptionCreateRequest(BaseModel):
    plan_code: str
    payment_mode: str = Field(default="manual")
    transaction_ref: Optional[str] = None


class SubscriptionVerifyRequest(BaseModel):
    status: str = Field(pattern="^(active|rejected)$")
    admin_note: Optional[str] = None


class SubscriptionResponse(BaseModel):
    id: int
    user_id: int
    plan_id: int
    status: str
    provider: str
    provider_ref: Optional[str] = None
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    created_at: datetime


class WalletResponse(BaseModel):
    balance: int


class WalletLedgerEntryResponse(BaseModel):
    id: int
    direction: str
    reason: str
    reference_type: str
    reference_id: str
    delta: int
    before_balance: int
    after_balance: int
    created_at: datetime


class WalletLedgerListResponse(BaseModel):
    balance: int
    entries: List[WalletLedgerEntryResponse]


class ExamCreateRequest(BaseModel):
    exam_name: str = Field(..., min_length=1, max_length=255)
    subject_name: str = Field(..., min_length=1, max_length=255)
    subject_code: Optional[str] = Field(default=None, max_length=100)
    has_set: bool = False
    set_count: int = Field(default=1, ge=1, le=4)
    set_labels: List[str] = Field(default_factory=lambda: ["A"])
    total_questions: int = Field(..., ge=1, le=200)
    negative_marking: bool = False
    negative_value: float = Field(default=0.0, ge=0, le=10)
    options_per_question: int = Field(default=4, ge=4, le=5)

    @model_validator(mode="after")
    def validate_set_configuration(self):
        if self.has_set:
            if len(self.set_labels) != self.set_count:
                raise ValueError("set_labels count must match set_count")
        else:
            self.set_count = 1
            if not self.set_labels:
                self.set_labels = ["A"]
            elif len(self.set_labels) > 1:
                self.set_labels = [self.set_labels[0]]
        return self


class ExamUpdateRequest(BaseModel):
    exam_name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    subject_name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    subject_code: Optional[str] = Field(default=None, max_length=100)
    total_questions: Optional[int] = Field(default=None, ge=1, le=200)
    negative_marking: Optional[bool] = None
    negative_value: Optional[float] = Field(default=None, ge=0, le=10)
    options_per_question: Optional[int] = Field(default=None, ge=4, le=5)
    status: Optional[str] = Field(default=None, pattern="^(draft|active|archived)$")


class ExamSetResponse(BaseModel):
    id: int
    set_label: str
    set_order: int


class ExamResponse(BaseModel):
    id: int
    exam_name: str
    subject_name: str
    subject_code: Optional[str] = None
    has_set: bool
    set_count: int
    total_questions: int
    options_per_question: int
    negative_marking: bool
    negative_value: float
    mark_per_question: float
    status: str
    created_at: datetime
    set_labels: List[ExamSetResponse]


class AnswerKeyUpsertRequest(BaseModel):
    mapping: Dict[str, str]


class AnswerKeyResponse(BaseModel):
    id: int
    exam_id: int
    set_label: Optional[str] = None
    mapping: Dict[str, str]
    version: int
    is_published: bool
    updated_at: Optional[datetime] = None


class ScanJobResponse(BaseModel):
    id: int
    batch_id: int
    exam_id: int
    source_file_key: str
    status: str
    attempts: int
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    token_charged: bool
    created_at: datetime
    updated_at: Optional[datetime] = None


class ScanBatchResponse(BaseModel):
    id: int
    exam_id: int
    status: str
    total_files: int
    processed_files: int
    created_at: datetime
    jobs: List[ScanJobResponse]


class ScanJobsListResponse(BaseModel):
    items: List[ScanJobResponse]


class ScanManualOverrideRequest(BaseModel):
    student_identifier: Optional[str] = None
    set_label: Optional[str] = None


class ResultListItem(BaseModel):
    sheet_id: int
    exam_id: int
    exam_name: Optional[str] = None
    student_identifier: Optional[str] = None
    set_label: Optional[str] = None
    final_score: float
    percentage: float
    evaluated_at: Optional[datetime] = None


class ResultListResponse(BaseModel):
    items: List[ResultListItem]


class ResultQuestionItem(BaseModel):
    question_no: int
    selected_option: Optional[str] = None
    correct_option: Optional[str] = None
    status: str
    mark_awarded: float


class ResultSummary(BaseModel):
    correct: int
    wrong: int
    unanswered: int
    invalid: int
    raw_score: float
    final_score: float
    percentage: float


class ResultDetailResponse(BaseModel):
    sheet_id: int
    exam_id: int
    student_identifier: Optional[str] = None
    set_label: Optional[str] = None
    summary: ResultSummary
    questions: List[ResultQuestionItem]


class FileSignRequest(BaseModel):
    file_key: str
    expires_in: Optional[int] = Field(default=600, ge=60, le=3600)


class FileSignResponse(BaseModel):
    url: str
    expires_at: int


class WebhookStubRequest(BaseModel):
    provider: str
    payload: Dict[str, Any]
