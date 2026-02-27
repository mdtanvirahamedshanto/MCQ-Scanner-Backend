"""v1 exam, template, answer key, and scan-batch endpoints."""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.v1_dependencies import (
    get_current_v1_user,
    get_idempotency_key,
    require_profile_complete,
)
from app.database import get_db
from app.models import AnswerKey, Exam, ExamSet, ScanBatch, ScanJob, User
from app.schemas_v1 import (
    AnswerKeyResponse,
    AnswerKeyUpsertRequest,
    ExamCreateRequest,
    ExamResponse,
    ExamSetResponse,
    ExamUpdateRequest,
    ScanBatchResponse,
    ScanJobResponse,
)
from app.services.idempotency_service import find_existing_scan_batch
from app.services.storage_service import save_upload_file
from app.services.template_service import generate_omr_template_pdf
from app.services.token_service import get_wallet
from app.workers.tasks_scan import process_scan_job

router = APIRouter()
settings = get_settings()
OPTION_ALIASES = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "ক": "A",
    "খ": "B",
    "গ": "C",
    "ঘ": "D",
    "ঙ": "E",
}


def _exam_set_response(exam: Exam) -> List[ExamSetResponse]:
    sets = sorted(exam.exam_sets, key=lambda s: s.set_order)
    return [ExamSetResponse(id=s.id, set_label=s.set_label, set_order=s.set_order) for s in sets]


def _exam_response(exam: Exam) -> ExamResponse:
    return ExamResponse(
        id=exam.id,
        exam_name=exam.exam_name or exam.title or "Untitled Exam",
        subject_name=exam.subject_name or exam.subject_code or "N/A",
        subject_code=exam.subject_code,
        has_set=bool(exam.has_set),
        set_count=exam.set_count or 1,
        total_questions=exam.total_questions,
        options_per_question=exam.options_per_question or 4,
        negative_marking=bool(exam.negative_marking),
        negative_value=float(exam.negative_value or 0.0),
        mark_per_question=float(exam.mark_per_question or 1.0),
        status=exam.status or "draft",
        created_at=exam.date_created,
        set_labels=_exam_set_response(exam),
    )


def _mapping_to_legacy_answers(mapping: Dict[str, str]) -> Dict[str, int]:
    inv = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return {str(k): inv.get(str(v).upper(), 0) for k, v in mapping.items()}


def _normalize_and_validate_mapping(mapping: Dict[str, str], total_questions: int, options_per_question: int) -> Dict[str, str]:
    allowed = ["A", "B", "C", "D", "E"][: max(4, min(5, options_per_question))]
    normalized: Dict[str, str] = {}

    for q_str, option in mapping.items():
        try:
            q_no = int(q_str)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid question number: {q_str}")

        if not 1 <= q_no <= total_questions:
            raise HTTPException(status_code=400, detail=f"Question number out of range: {q_no}")

        raw_option = str(option).strip()
        option_norm = OPTION_ALIASES.get(raw_option.upper()) or OPTION_ALIASES.get(raw_option)
        if option_norm not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid option '{raw_option}' for question {q_no}")

        normalized[str(q_no)] = option_norm

    return normalized


async def _get_exam_for_owner(db: AsyncSession, exam_id: int, owner_id: int) -> Exam:
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.teacher_id != owner_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    await db.refresh(exam, attribute_names=["exam_sets", "answer_keys"])
    return exam


@router.post("/exams", response_model=ExamResponse)
async def create_exam(
    payload: ExamCreateRequest,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    exam = Exam(
        teacher_id=current_user.id,
        title=payload.exam_name,
        exam_name=payload.exam_name,
        subject_name=payload.subject_name,
        subject_code=payload.subject_code,
        has_set=payload.has_set,
        set_count=payload.set_count,
        total_questions=payload.total_questions,
        options_per_question=payload.options_per_question,
        negative_marking=payload.negative_marking,
        negative_value=payload.negative_value,
        mark_per_question=1.0,
        status="draft",
    )
    db.add(exam)
    await db.flush()

    labels = payload.set_labels[: payload.set_count]
    for idx, label in enumerate(labels, start=1):
        db.add(ExamSet(exam_id=exam.id, set_label=str(label), set_order=idx))

    await db.flush()
    await db.refresh(exam, attribute_names=["exam_sets"])
    return _exam_response(exam)


@router.get("/exams", response_model=list[ExamResponse])
async def list_exams(
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Exam).where(Exam.teacher_id == current_user.id).order_by(Exam.date_created.desc())
    )
    exams = list(result.scalars().all())
    for exam in exams:
        await db.refresh(exam, attribute_names=["exam_sets"])
    return [_exam_response(exam) for exam in exams]


@router.get("/exams/{exam_id}", response_model=ExamResponse)
async def get_exam(
    exam_id: int,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    exam = await _get_exam_for_owner(db, exam_id, current_user.id)
    return _exam_response(exam)


@router.patch("/exams/{exam_id}", response_model=ExamResponse)
async def update_exam(
    exam_id: int,
    payload: ExamUpdateRequest,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    exam = await _get_exam_for_owner(db, exam_id, current_user.id)

    updates = payload.model_dump(exclude_none=True)
    for field, value in updates.items():
        setattr(exam, field, value)
        if field == "exam_name":
            exam.title = value

    await db.flush()
    await db.refresh(exam, attribute_names=["exam_sets"])
    return _exam_response(exam)


@router.get("/exams/{exam_id}/template.pdf")
async def download_template(
    exam_id: int,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    exam = await _get_exam_for_owner(db, exam_id, current_user.id)
    labels = [s.set_label for s in sorted(exam.exam_sets, key=lambda s: s.set_order)] or ["A"]

    pdf_buffer: BytesIO = generate_omr_template_pdf(exam, labels)
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="exam-{exam.id}-template.pdf"'},
    )


@router.put("/exams/{exam_id}/answer-keys/{set_label}", response_model=AnswerKeyResponse)
async def upsert_answer_key(
    exam_id: int,
    set_label: str,
    payload: AnswerKeyUpsertRequest,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    exam = await _get_exam_for_owner(db, exam_id, current_user.id)

    set_entity = None
    for s in exam.exam_sets:
        if s.set_label == set_label:
            set_entity = s
            break

    if exam.has_set and not set_entity:
        raise HTTPException(status_code=404, detail="Set label not configured for this exam")

    normalized = _normalize_and_validate_mapping(
        payload.mapping,
        total_questions=exam.total_questions,
        options_per_question=exam.options_per_question or 4,
    )

    query = await db.execute(
        select(AnswerKey).where(
            AnswerKey.exam_id == exam.id,
            AnswerKey.set_id == (set_entity.id if set_entity else None),
            AnswerKey.version == 1,
        )
    )
    answer_key = query.scalar_one_or_none()

    if not answer_key:
        answer_key = AnswerKey(
            exam_id=exam.id,
            set_id=set_entity.id if set_entity else None,
            set_code=set_label,
            answers=_mapping_to_legacy_answers(normalized),
            mapping=normalized,
            version=1,
            is_published=True,
        )
        db.add(answer_key)
    else:
        answer_key.set_code = set_label
        answer_key.answers = _mapping_to_legacy_answers(normalized)
        answer_key.mapping = normalized
        answer_key.is_published = True

    await db.flush()
    await db.refresh(answer_key)

    return AnswerKeyResponse(
        id=answer_key.id,
        exam_id=answer_key.exam_id,
        set_label=set_label,
        mapping=answer_key.mapping or {},
        version=answer_key.version or 1,
        is_published=bool(answer_key.is_published),
        updated_at=answer_key.updated_at,
    )


@router.get("/exams/{exam_id}/answer-keys", response_model=list[AnswerKeyResponse])
async def list_answer_keys(
    exam_id: int,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    exam = await _get_exam_for_owner(db, exam_id, current_user.id)

    items: list[AnswerKeyResponse] = []
    for ak in exam.answer_keys:
        set_label = ak.set_code
        if ak.exam_set and ak.exam_set.set_label:
            set_label = ak.exam_set.set_label
        items.append(
            AnswerKeyResponse(
                id=ak.id,
                exam_id=ak.exam_id,
                set_label=set_label,
                mapping=(ak.mapping or {}),
                version=ak.version or 1,
                is_published=bool(ak.is_published),
                updated_at=ak.updated_at,
            )
        )
    return items


@router.post("/exams/{exam_id}/scan-batches", response_model=ScanBatchResponse)
async def create_scan_batch(
    exam_id: int,
    files: list[UploadFile] = File(...),
    idempotency_key: str = Depends(get_idempotency_key),
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    exam = await _get_exam_for_owner(db, exam_id, current_user.id)

    # Ensure at least one answer key exists
    if len(exam.answer_keys) == 0:
        raise HTTPException(status_code=400, detail="Configure answer keys before scanning")

    wallet = await get_wallet(db, current_user.id)
    if wallet.balance < max(1, settings.TOKEN_COST_PER_SUCCESS_SHEET):
        raise HTTPException(status_code=402, detail="Insufficient tokens")

    existing_batch = await find_existing_scan_batch(db, current_user.id, exam.id, idempotency_key)
    if existing_batch:
        await db.refresh(existing_batch, attribute_names=["jobs"])
        return ScanBatchResponse(
            id=existing_batch.id,
            exam_id=existing_batch.exam_id,
            status=existing_batch.status,
            total_files=existing_batch.total_files,
            processed_files=existing_batch.processed_files,
            created_at=existing_batch.created_at,
            jobs=[
                ScanJobResponse(
                    id=j.id,
                    batch_id=j.batch_id,
                    exam_id=j.exam_id,
                    source_file_key=j.source_file_key,
                    status=j.status,
                    attempts=j.attempts,
                    error_code=j.error_code,
                    error_message=j.error_message,
                    token_charged=bool(j.token_charged),
                    created_at=j.created_at,
                    updated_at=j.updated_at,
                )
                for j in existing_batch.jobs
            ],
        )

    batch = ScanBatch(
        user_id=current_user.id,
        exam_id=exam.id,
        idempotency_key=idempotency_key,
        status="queued",
        total_files=len(files),
        processed_files=0,
    )
    db.add(batch)
    await db.flush()

    jobs: list[ScanJob] = []
    for item in files:
        content_type = (item.content_type or "").lower()
        if content_type not in {
            "image/jpeg",
            "image/png",
            "image/jpg",
            "application/pdf",
        }:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {item.filename}")

        ext = Path(item.filename or "file").suffix or ".jpg"
        rel_key = f"original/{current_user.id}/{batch.id}/{uuid.uuid4()}{ext}"
        file_key, file_hash = await save_upload_file(item, rel_key)

        job = ScanJob(
            batch_id=batch.id,
            user_id=current_user.id,
            exam_id=exam.id,
            source_file_key=file_key,
            source_file_hash=file_hash,
            status="queued",
        )
        db.add(job)
        jobs.append(job)

    await db.flush()

    for job in jobs:
        try:
            async_result = process_scan_job.delay(job.id)
            job.task_id = async_result.id
        except Exception:
            # Worker/broker might be down in dev; job remains queued for manual retry.
            job.task_id = None

    await db.flush()

    return ScanBatchResponse(
        id=batch.id,
        exam_id=batch.exam_id,
        status=batch.status,
        total_files=batch.total_files,
        processed_files=batch.processed_files,
        created_at=batch.created_at,
        jobs=[
            ScanJobResponse(
                id=j.id,
                batch_id=j.batch_id,
                exam_id=j.exam_id,
                source_file_key=j.source_file_key,
                status=j.status,
                attempts=j.attempts,
                error_code=j.error_code,
                error_message=j.error_message,
                token_charged=bool(j.token_charged),
                created_at=j.created_at,
                updated_at=j.updated_at,
            )
            for j in jobs
        ],
    )


@router.get("/scan-batches/{batch_id}", response_model=ScanBatchResponse)
async def get_scan_batch(
    batch_id: int,
    current_user: User = Depends(get_current_v1_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ScanBatch).where(ScanBatch.id == batch_id))
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    await db.refresh(batch, attribute_names=["jobs"])
    return ScanBatchResponse(
        id=batch.id,
        exam_id=batch.exam_id,
        status=batch.status,
        total_files=batch.total_files,
        processed_files=batch.processed_files,
        created_at=batch.created_at,
        jobs=[
            ScanJobResponse(
                id=j.id,
                batch_id=j.batch_id,
                exam_id=j.exam_id,
                source_file_key=j.source_file_key,
                status=j.status,
                attempts=j.attempts,
                error_code=j.error_code,
                error_message=j.error_message,
                token_charged=bool(j.token_charged),
                created_at=j.created_at,
                updated_at=j.updated_at,
            )
            for j in sorted(batch.jobs, key=lambda x: x.id)
        ],
    )


@router.post("/exams/{exam_id}/evaluate")
async def evaluate_omr_sheet(
    exam_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    """Synchronous OMR evaluation: upload image, process, grade, return result."""
    from app.utils.omr_engine import process_omr_image, grade_omr_result

    valid_types = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
    if not file.content_type or file.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG) or PDF")

    exam = await _get_exam_for_owner(db, exam_id, current_user.id)

    # Ensure answer keys exist
    if not exam.answer_keys:
        raise HTTPException(status_code=400, detail="Configure answer keys before evaluating")

    # Build answer key lookup: set_code -> answers dict
    BENGALI_CODES = {"ক", "খ", "গ", "ঘ", "ঙ", "চ"}
    answer_key_by_set: Dict[str, dict] = {}
    use_bengali = True
    for ak in exam.answer_keys:
        key = ak.set_code or "A"
        answer_key_by_set[key] = ak.answers or {}
        if key not in BENGALI_CODES:
            use_bengali = False

    # Save uploaded file
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "image").suffix or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    filepath = upload_dir / filename

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB}MB")

    with open(filepath, "wb") as f:
        f.write(contents)

    # Convert PDF to image if needed
    image_path = filepath
    if file.content_type == "application/pdf":
        try:
            import fitz
            doc = fitz.open(filepath)
            if len(doc) == 0:
                raise HTTPException(status_code=400, detail="Empty PDF file")
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            image_path = filepath.with_suffix(".jpg")
            pix.save(str(image_path))
            doc.close()
        except ImportError:
            raise HTTPException(status_code=500, detail="PDF processing not available")

    # Run OMR engine synchronously
    omr_result = process_omr_image(
        str(image_path),
        num_questions=exam.total_questions,
        use_bengali_set_codes=use_bengali,
    )

    if not omr_result.success:
        return {
            "success": False,
            "message": omr_result.error_message,
            "roll_number": omr_result.roll_number or "",
            "set_code": omr_result.set_code or "?",
            "marks_obtained": 0,
            "wrong_answers": [],
            "percentage": 0.0,
            "answers": omr_result.answers,
            "image_url": f"/{settings.UPLOAD_DIR}/{filename}",
        }

    # Pick correct answer key for the detected set
    if len(answer_key_by_set) == 1:
        set_code = list(answer_key_by_set.keys())[0]
    else:
        # If it's a NormalOMRLayout, it might return N/A or ?.
        # In that case, we can't reliably pick a set if there are multiple sets,
        # but we can default to the first set to avoid a hard error if possible.
        if omr_result.set_code in ("N/A", "?"):
            set_code = list(answer_key_by_set.keys())[0]
        else:
            set_code = omr_result.set_code

    if set_code not in answer_key_by_set:
        return {
            "success": False,
            "message": f"No answer key found for set '{set_code}'",
            "roll_number": omr_result.roll_number or "",
            "set_code": set_code,
            "marks_obtained": 0,
            "wrong_answers": [],
            "percentage": 0.0,
            "answers": omr_result.answers,
            "image_url": f"/{settings.UPLOAD_DIR}/{filename}",
        }

    answer_key = answer_key_by_set[set_code]
    marks_obtained, wrong_answers_list, percentage = grade_omr_result(
        omr_result, answer_key
    )

    return {
        "success": True,
        "message": "OMR processed successfully",
        "roll_number": omr_result.roll_number or "",
        "set_code": set_code,
        "marks_obtained": marks_obtained,
        "wrong_answers": wrong_answers_list,
        "percentage": round(percentage, 2),
        "answers": omr_result.answers,
        "image_url": f"/{settings.UPLOAD_DIR}/{filename}",
    }
