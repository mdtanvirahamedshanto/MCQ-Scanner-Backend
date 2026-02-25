"""v1 scan job status and control endpoints."""

from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.v1_dependencies import get_current_v1_user, require_profile_complete
from app.database import get_db
from app.models import AnswerKey, Exam, ScanJob, ScannedSheet, SheetAnswer, User
from app.schemas_v1 import ScanJobResponse, ScanJobsListResponse, ScanManualOverrideRequest
from app.services.scoring_service import score_sheet
from app.workers.tasks_scan import process_scan_job

router = APIRouter()


def _job_response(job: ScanJob) -> ScanJobResponse:
    return ScanJobResponse(
        id=job.id,
        batch_id=job.batch_id,
        exam_id=job.exam_id,
        source_file_key=job.source_file_key,
        status=job.status,
        attempts=job.attempts,
        error_code=job.error_code,
        error_message=job.error_message,
        token_charged=bool(job.token_charged),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _answer_key_to_mapping(answer_key: AnswerKey) -> Dict[int, str]:
    raw = answer_key.mapping or {}
    if not raw and answer_key.answers:
        inv = ["A", "B", "C", "D", "E"]
        for q, v in (answer_key.answers or {}).items():
            try:
                idx = int(v)
                raw[str(q)] = inv[idx] if 0 <= idx < len(inv) else "A"
            except (TypeError, ValueError):
                raw[str(q)] = "A"

    normalized: Dict[int, str] = {}
    for q, opt in raw.items():
        try:
            q_no = int(q)
        except (TypeError, ValueError):
            continue
        if isinstance(opt, str):
            normalized[q_no] = opt.upper()
    return normalized


@router.get("/scan-jobs", response_model=ScanJobsListResponse)
async def list_scan_jobs(
    batch_id: Optional[int] = Query(default=None),
    exam_id: Optional[int] = Query(default=None),
    status: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ScanJob).where(ScanJob.user_id == current_user.id)
    if batch_id is not None:
        stmt = stmt.where(ScanJob.batch_id == batch_id)
    if exam_id is not None:
        stmt = stmt.where(ScanJob.exam_id == exam_id)
    if status:
        stmt = stmt.where(ScanJob.status == status)

    stmt = stmt.order_by(ScanJob.created_at.desc()).limit(500)
    result = await db.execute(stmt)
    jobs = list(result.scalars().all())
    return ScanJobsListResponse(items=[_job_response(j) for j in jobs])


@router.get("/scan-jobs/{job_id}", response_model=ScanJobResponse)
async def get_scan_job(
    job_id: int,
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ScanJob).where(ScanJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return _job_response(job)


@router.patch("/scan-jobs/{job_id}/manual", response_model=ScanJobResponse)
async def manual_override_scan_job(
    job_id: int,
    payload: ScanManualOverrideRequest,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ScanJob).where(ScanJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    sheet_result = await db.execute(select(ScannedSheet).where(ScannedSheet.scan_job_id == job.id))
    sheet = sheet_result.scalar_one_or_none()
    if not sheet:
        raise HTTPException(status_code=400, detail="No scored sheet available for override")

    exam = await db.get(Exam, job.exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found for this job")
    await db.refresh(exam, attribute_names=["answer_keys"])
    for ak in exam.answer_keys:
        if ak.set_id:
            await db.refresh(ak, attribute_names=["exam_set"])

    answer_key_map: Dict[str, Dict[int, str]] = {}
    for ak in exam.answer_keys:
        label = ak.set_code
        if ak.exam_set and ak.exam_set.set_label:
            label = ak.exam_set.set_label
        answer_key_map[label] = _answer_key_to_mapping(ak)

    chosen_set = payload.set_label or sheet.set_label_final or next(iter(answer_key_map.keys()), None)
    if not chosen_set or chosen_set not in answer_key_map:
        raise HTTPException(status_code=400, detail="Invalid set_label for override")

    sheet_answers_result = await db.execute(select(SheetAnswer).where(SheetAnswer.sheet_id == sheet.id))
    current_answers = list(sheet_answers_result.scalars().all())

    extracted: Dict[int, Optional[str]] = {a.question_no: a.selected_option for a in current_answers}
    rows, summary = score_sheet(
        extracted_answers=extracted,
        answer_key=answer_key_map[chosen_set],
        total_questions=exam.total_questions,
        mark_per_question=exam.mark_per_question or 1.0,
        negative_marking=bool(exam.negative_marking),
        negative_value=float(exam.negative_value or 0.0),
    )

    rows_by_q = {r["question_no"]: r for r in rows}
    for ans in current_answers:
        row = rows_by_q.get(ans.question_no)
        if not row:
            continue
        ans.correct_option = row["correct_option"]
        ans.status = row["status"]
        ans.mark_awarded = row["mark_awarded"]
        ans.is_overridden = True

    sheet.student_identifier = payload.student_identifier or sheet.student_identifier
    sheet.set_label_final = chosen_set
    sheet.manual_override = True
    sheet.correct_count = summary["correct"]
    sheet.wrong_count = summary["wrong"]
    sheet.unanswered_count = summary["unanswered"]
    sheet.invalid_count = summary["invalid"]
    sheet.raw_score = summary["raw_score"]
    sheet.final_score = summary["final_score"]
    sheet.percentage = summary["percentage"]
    sheet.evaluated_at = datetime.utcnow()

    await db.flush()
    return _job_response(job)


@router.post("/scan-jobs/{job_id}/retry", response_model=ScanJobResponse)
async def retry_scan_job(
    job_id: int,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ScanJob).where(ScanJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if job.status == "processing":
        raise HTTPException(status_code=409, detail="Job is already processing")

    job.status = "queued"
    job.error_code = None
    job.error_message = None
    await db.flush()

    try:
        task = process_scan_job.delay(job.id)
        job.task_id = task.id
    except Exception:
        job.task_id = None

    await db.flush()
    return _job_response(job)
