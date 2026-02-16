"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Optional, Dict, List

from pydantic import BaseModel, EmailStr, Field


# --- Auth Schemas ---
class UserCreate(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    """Schema for user in responses."""

    id: int
    email: str
    is_subscribed: bool
    subscription_plan: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """Schema for login request."""

    email: EmailStr
    password: str


# --- Exam Schemas ---
# Answer key as JSON: {"1": 0, "2": 2, "3": 1, ...} question_no -> correct_option (0-3)
AnswerKeyDict = Dict[str, int]


class AnswerKeySet(BaseModel):
    """Answer key for one set (e.g., ক, খ). answers: question_no (str) -> correct_option (0-3)."""

    set_code: str = Field(..., min_length=1, max_length=10)
    answers: Dict[str, int] = Field(
        ...,
        description='{"1": 0, "2": 2, "3": 1, ...}',
        examples=[{"1": 2, "2": 0, "3": 1}],
    )


class ExamCreate(BaseModel):
    """Schema for creating an exam with answer keys."""

    title: str = Field(..., min_length=1, max_length=255)
    subject_code: str = Field(..., min_length=1, max_length=50)
    total_questions: int = Field(default=60, ge=1, le=100)
    answer_keys: list[AnswerKeySet] = Field(..., min_length=1)


class ExamResponse(BaseModel):
    """Schema for exam in responses."""

    id: int
    teacher_id: int
    title: str
    subject_code: str
    total_questions: int
    date_created: datetime

    class Config:
        from_attributes = True


# --- Result Schemas ---
class ScanResultResponse(BaseModel):
    """Response from OMR scan endpoint."""

    roll_number: str
    set_code: str
    marks_obtained: int
    wrong_answers: List[int]
    percentage: float
    answers: List[int]
    success: bool
    message: str = ""


class ResultResponse(BaseModel):
    """Schema for result in responses."""

    id: int
    exam_id: int
    roll_number: str
    set_code: str
    marks_obtained: int
    wrong_answers: List[int]
    percentage: float
    image_url: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
