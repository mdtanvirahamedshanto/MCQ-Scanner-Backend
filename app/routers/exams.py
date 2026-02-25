"""Exam management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
import io
import uuid
from pathlib import Path

from app.database import get_db
from app.models import User, Exam, AnswerKey, Result
from app.config import get_settings
from app.schemas import ExamCreate, ExamResponse, ResultResponse, ResultsWithAnalytics
from app.dependencies import get_current_user
from app.utils.omr_template import generate_omr_template_pdf
from app.utils.export_results import export_results_excel, export_results_pdf
from app.utils.omr_engine import process_omr_image, grade_omr_result
from app.utils.answer_key_ocr import parse_answer_key_image
from app.schemas import ScanResultResponse

router = APIRouter(prefix="/exams", tags=["exams"])
settings = get_settings()
BENGALI_CODES = {"ক", "খ", "গ", "ঘ", "ঙ", "চ"}


@router.post("/parse-answer-key")
async def parse_answer_key_from_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload answer key image (handwritten or printed).
    Supports: ১. ক, 1. A, 1. 1 format. Returns extracted answers for teacher to confirm.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "img").suffix or ".jpg"
    tmp_path = upload_dir / f"answer_key_{uuid.uuid4()}{ext}"

    try:
        contents = await file.read()
        if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        answers, err = parse_answer_key_image(str(tmp_path))
        if err:
            raise HTTPException(status_code=400, detail=err)

        # Convert to frontend format: {1: "A", 2: "B", ...}
        return {"answers": answers, "count": len(answers)}
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.post("/create", response_model=ExamResponse)
@router.post("", response_model=ExamResponse)
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

    for ak_set in exam_data.get_answer_keys_list():
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


@router.post("/{exam_id}/scan", response_model=dict)
async def bulk_scan_omr(
    exam_id: int,
    images: List[UploadFile] = File(..., alias="images"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bulk upload and scan OMR images. Returns list of results."""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    ak_result = await db.execute(select(AnswerKey).where(AnswerKey.exam_id == exam_id))
    answer_keys = ak_result.scalars().all()
    answer_key_by_set = {ak.set_code: ak.answers for ak in answer_keys}
    use_bengali_set_codes = True
    if answer_keys:
        use_bengali_set_codes = (answer_keys[0].set_code in BENGALI_CODES)
    if not answer_key_by_set:
        raise HTTPException(status_code=400, detail="Exam has no answer key configured")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    scan_results = []

    for file in images:
        if not file.content_type or not file.content_type.startswith("image/"):
            scan_results.append(
                ScanResultResponse(
                    roll_number="",
                    set_code="?",
                    marks_obtained=0,
                    wrong_answers=[],
                    percentage=0.0,
                    answers=[],
                    success=False,
                    message=f"Skip: {file.filename or 'file'} is not an image",
                )
            )
            continue

        ext = Path(file.filename or "image").suffix or ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        filepath = upload_dir / filename
        try:
            contents = await file.read()
            if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                scan_results.append(
                    ScanResultResponse(
                        roll_number="",
                        set_code="?",
                        marks_obtained=0,
                        wrong_answers=[],
                        percentage=0.0,
                        answers=[],
                        success=False,
                        message=f"File too large: {file.filename}",
                    )
                )
                continue
            with open(filepath, "wb") as f:
                f.write(contents)
        except Exception as e:
            scan_results.append(
                ScanResultResponse(
                    roll_number="",
                    set_code="?",
                    marks_obtained=0,
                    wrong_answers=[],
                    percentage=0.0,
                    answers=[],
                    success=False,
                    message=str(e),
                )
            )
            continue

        omr_result = process_omr_image(
            filepath,
            num_questions=exam.total_questions,
            use_bengali_set_codes=use_bengali_set_codes,
        )

        if not omr_result.success:
            scan_results.append(
                ScanResultResponse(
                    roll_number=omr_result.roll_number,
                    set_code=omr_result.set_code or "?",
                    marks_obtained=0,
                    wrong_answers=[],
                    percentage=0.0,
                    answers=omr_result.answers,
                    success=False,
                    message=omr_result.error_message,
                )
            )
            continue

        if len(answer_key_by_set) == 1:
            set_code = list(answer_key_by_set.keys())[0]
        else:
            set_code = omr_result.set_code
        if set_code not in answer_key_by_set:
            scan_results.append(
                ScanResultResponse(
                    roll_number=omr_result.roll_number,
                    set_code=set_code,
                    marks_obtained=0,
                    wrong_answers=[],
                    percentage=0.0,
                    answers=omr_result.answers,
                    success=False,
                    message=f"No answer key for set '{set_code}'",
                )
            )
            continue

        answer_key = answer_key_by_set[set_code]
        marks_obtained, wrong_answers_list, percentage = grade_omr_result(
            omr_result, answer_key
        )
        image_url = f"/{settings.UPLOAD_DIR}/{filename}"
        result_record = Result(
            exam_id=exam_id,
            roll_number=omr_result.roll_number or "unknown",
            set_code=set_code,
            marks_obtained=marks_obtained,
            wrong_answers=wrong_answers_list,
            percentage=percentage,
            image_url=image_url,
        )
        db.add(result_record)
        scan_results.append(
            ScanResultResponse(
                roll_number=omr_result.roll_number,
                set_code=set_code,
                marks_obtained=marks_obtained,
                wrong_answers=wrong_answers_list,
                percentage=percentage,
                answers=omr_result.answers,
                success=True,
                message="Processed successfully",
            )
        )

    return {"results": [r.model_dump() for r in scan_results]}


@router.get("", response_model=List[ExamResponse])
async def list_exams(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all exams for the current teacher."""
    result = await db.execute(
        select(Exam).where(Exam.teacher_id == current_user.id).order_by(Exam.date_created.desc())
    )
    exams = result.scalars().all()
    return exams


@router.get("/{exam_id}", response_model=ExamResponse)
async def get_exam(
    exam_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single exam by ID."""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return exam


@router.get("/{exam_id}/results", response_model=ResultsWithAnalytics)
async def get_exam_results(
    exam_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all results for an exam with class-wide analytics."""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    res_result = await db.execute(
        select(Result).where(Result.exam_id == exam_id).order_by(Result.created_at.desc())
    )
    results = res_result.scalars().all()

    # Compute analytics
    total_marks = exam.total_questions
    if results:
        marks_list = [r.marks_obtained for r in results]
        pct_list = [r.percentage for r in results]
        return ResultsWithAnalytics(
            results=[ResultResponse.model_validate(r) for r in results],
            total_count=len(results),
            average_percentage=round(sum(pct_list) / len(pct_list), 2),
            highest_marks=max(marks_list),
            lowest_marks=min(marks_list),
            total_marks=total_marks,
        )
    return ResultsWithAnalytics(
        results=[],
        total_count=0,
        average_percentage=0.0,
        highest_marks=0,
        lowest_marks=0,
        total_marks=total_marks,
    )


@router.get("/{exam_id}/results/export")
async def export_exam_results(
    exam_id: int,
    format: str = "xlsx",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export results as Excel (xlsx) or PDF."""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    res_result = await db.execute(select(Result).where(Result.exam_id == exam_id))
    results = res_result.scalars().all()
    results_data = [
        {
            "roll_number": r.roll_number,
            "set_code": r.set_code,
            "marks_obtained": r.marks_obtained,
            "wrong_answers": r.wrong_answers or [],
            "percentage": r.percentage,
            "created_at": r.created_at,
        }
        for r in results
    ]

    if format.lower() == "pdf":
        stats = {}
        if results:
            stats = {
                "average": sum(r.percentage for r in results) / len(results),
                "highest": max(r.marks_obtained for r in results),
                "lowest": min(r.marks_obtained for r in results),
            }
        buffer = export_results_pdf(
            results_data, exam.title, exam.total_questions, stats
        )
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="exam-{exam_id}-results.pdf"'},
        )
    else:
        buffer = export_results_excel(
            results_data, exam.title, exam.total_questions
        )
        # Check if we got xlsx (openpyxl) or CSV fallback
        buffer.seek(0)
        first_bytes = buffer.read(4)
        buffer.seek(0)
        is_xlsx = first_bytes[:2] == b"PK"  # xlsx is a zip file
        media_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if is_xlsx
            else "text/csv"
        )
        ext = "xlsx" if is_xlsx else "csv"
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="exam-{exam_id}-results.{ext}"'},
        )


@router.get("/{exam_id}/omr-template")
async def download_omr_template(
    exam_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download standardized OMR sheet PDF template."""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    ak_result = await db.execute(select(AnswerKey).where(AnswerKey.exam_id == exam_id))
    answer_keys = ak_result.scalars().all()
    use_bengali = True
    num_sets = 4
    if answer_keys:
        first_set = answer_keys[0].set_code
        use_bengali = first_set in ("ক", "খ", "গ", "ঘ")
        num_sets = len(answer_keys)

    buffer = generate_omr_template_pdf(
        exam_title=exam.title,
        subject_code=exam.subject_code,
        total_questions=exam.total_questions,
        use_bengali_set_codes=use_bengali,
        num_sets=num_sets,
    )
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="OMR-Sheet-{exam.subject_code}.pdf"'},
    )
