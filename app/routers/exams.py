"""Exam management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, Exam, AnswerKey
from app.schemas import ExamCreate, ExamResponse
from app.dependencies import get_current_user

router = APIRouter(prefix="/exams", tags=["exams"])


@router.post("/create", response_model=ExamResponse)
async def create_exam(
    exam_data: ExamCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create an exam and upload answer keys for different sets."""
    exam = Exam(
        teacher_id=current_user.id,
        title=exam_data.title,
        subject_code=exam_data.subject_code,
        total_questions=exam_data.total_questions,
    )
    db.add(exam)
    await db.flush()

    for ak_set in exam_data.answer_keys:
        # Convert answers dict keys to strings for JSON storage
        answers_json = {str(k): v for k, v in ak_set.answers.items()}
        answer_key = AnswerKey(
            exam_id=exam.id,
            set_code=ak_set.set_code,
            answers=answers_json,
        )
        db.add(answer_key)

    await db.refresh(exam)
    return exam
