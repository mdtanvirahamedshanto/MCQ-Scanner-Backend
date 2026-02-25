"""OMR scan endpoint."""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, Exam, AnswerKey, Result
from app.schemas import ScanResultResponse
from app.dependencies import get_current_user
from app.utils.omr_engine import process_omr_image, grade_omr_result
from app.config import get_settings
import fitz # PyMuPDF

router = APIRouter(prefix="/scan-omr", tags=["scan"])
settings = get_settings()
BENGALI_CODES = {"ক", "খ", "গ", "ঘ", "ঙ", "চ"}


@router.post("/{exam_id}", response_model=ScanResultResponse)
async def scan_omr(
    exam_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload OMR image, process via engine, compare with answer key,
    save to Result table, and return detailed JSON response.
    """
    valid_content_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
    if not file.content_type or file.content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG) or PDF",
        )

    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exam not found",
        )
    if exam.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to scan this exam",
        )

    # Token check
    if current_user.tokens <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="You have no tokens left. Please purchase a subscription to scan more sheets.",
        )

    ak_result = await db.execute(
        select(AnswerKey).where(AnswerKey.exam_id == exam_id)
    )
    answer_keys = ak_result.scalars().all()
    # Build dict: set_code -> answers (for lookup by set)
    answer_key_by_set = {ak.set_code: ak.answers for ak in answer_keys}
    use_bengali_set_codes = True
    if answer_keys:
        use_bengali_set_codes = (answer_keys[0].set_code in BENGALI_CODES)

    if not answer_key_by_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exam has no answer key configured",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "image").suffix or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    filepath = upload_dir / filename

    try:
        contents = await file.read()
        if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB",
            )
        with open(filepath, "wb") as f:
            f.write(contents)
            
        # Convert PDF to Image if necessary
        image_path = filepath
        if file.content_type == "application/pdf":
            doc = fitz.open(filepath)
            if len(doc) == 0:
                raise HTTPException(status_code=400, detail="Empty PDF file.")
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # render at 300 DPI
            image_path = filepath.with_suffix(".jpg")
            pix.save(str(image_path))
            doc.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    omr_result = process_omr_image(
        str(image_path),
        num_questions=exam.total_questions,
        use_bengali_set_codes=use_bengali_set_codes,
    )

    if not omr_result.success:
        return ScanResultResponse(
            roll_number=omr_result.roll_number,
            set_code=omr_result.set_code or "?",
            marks_obtained=0,
            wrong_answers=[],
            percentage=0.0,
            answers=omr_result.answers,
            success=False,
            message=omr_result.error_message,
        )

    # Get answer key for detected set
    if len(answer_key_by_set) == 1:
        set_code = list(answer_key_by_set.keys())[0]
    else:
        set_code = omr_result.set_code

    if set_code not in answer_key_by_set:
        return ScanResultResponse(
            roll_number=omr_result.roll_number,
            set_code=set_code,
            marks_obtained=0,
            wrong_answers=[],
            percentage=0.0,
            answers=omr_result.answers,
            success=False,
            message=f"No answer key found for set '{set_code}'",
        )

    answer_key = answer_key_by_set[set_code]
    marks_obtained, wrong_answers_list, percentage = grade_omr_result(
        omr_result,
        answer_key,
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

    # Deduct token
    current_user.tokens -= 1
    db.add(current_user)

    return ScanResultResponse(
        roll_number=omr_result.roll_number,
        set_code=set_code,
        marks_obtained=marks_obtained,
        wrong_answers=wrong_answers_list,
        percentage=percentage,
        answers=omr_result.answers,
        success=True,
        message="OMR processed successfully",
    )
