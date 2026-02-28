"""v1 result list/detail/export endpoints."""

from io import BytesIO
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from app.core.v1_dependencies import require_profile_complete
from app.database import get_db
from app.models import Exam, ScannedSheet, SheetAnswer, User
from app.schemas_v1 import (
    ResultDetailResponse,
    ResultListItem,
    ResultListResponse,
    ResultQuestionItem,
    ResultSummary,
    ResultCreateRequest,
)

router = APIRouter()


@router.get("/results", response_model=ResultListResponse)
async def list_results(
    exam_id: Optional[int] = Query(default=None),
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ScannedSheet).where(ScannedSheet.user_id == current_user.id)
    if exam_id is not None:
        stmt = stmt.where(ScannedSheet.exam_id == exam_id)

    stmt = stmt.order_by(ScannedSheet.evaluated_at.desc().nullslast(), ScannedSheet.id.desc()).limit(1000)
    rows = (await db.execute(stmt)).scalars().all()

    items: list[ResultListItem] = []
    for row in rows:
        exam = await db.get(Exam, row.exam_id)
        items.append(
            ResultListItem(
                sheet_id=row.id,
                exam_id=row.exam_id,
                exam_name=(exam.exam_name if exam else None),
                student_identifier=row.student_identifier,
                set_label=row.set_label_final,
                final_score=row.final_score or 0.0,
                percentage=row.percentage or 0.0,
                evaluated_at=row.evaluated_at,
            )
        )

    return ResultListResponse(items=items)


@router.post("/exams/{exam_id}/results", response_model=ResultListItem)
async def save_result(
    exam_id: int,
    payload: ResultCreateRequest,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    """Saves a transient evaluated OMR result explicitly into the database."""
    exam = await db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
        
    set_label = payload.set_code if payload.set_code not in ("?", "N/A") else "A"
    
    wrong_count = len(payload.wrong_answers)
    total_answered = len(payload.answers) # Assumes all parsed are answered, though actual missing isn't purely in payload
    # Approximations since frontend evaluation doesn't pass detailed answer breakdown
    correct_count = total_answered - wrong_count if total_answered > wrong_count else 0
    
    sheet = ScannedSheet(
        exam_id=exam_id,
        user_id=current_user.id,
        student_identifier=payload.roll_number or "Unknown",
        set_label_final=set_label,
        raw_score=payload.marks_obtained,
        final_score=payload.marks_obtained,
        percentage=payload.percentage,
        correct_count=correct_count,
        wrong_count=wrong_count,
        unanswered_count=exam.total_questions - total_answered if exam.total_questions > total_answered else 0,
        invalid_count=0,
        evaluated_at=datetime.utcnow()
    )
    db.add(sheet)
    await db.flush()
    await db.refresh(sheet)
    
    for i, ans in enumerate(payload.answers):
        q_no = i + 1
        is_wrong = q_no in payload.wrong_answers
        status = "wrong" if is_wrong else "correct"
        # Option index to string: 0->A, 1->B, 2->C, etc. -1 means unattempted/invalid
        option_str = chr(65 + ans) if ans >= 0 else ("unanswered" if ans == -1 else "invalid")
        sheet_answer = SheetAnswer(
            sheet_id=sheet.id,
            question_no=q_no,
            selected_option=option_str,
            status=status,
            mark_awarded=0 if is_wrong else float(exam.mark_per_question or 1.0)
        )
        db.add(sheet_answer)
        
    await db.commit()
    
    return ResultListItem(
        sheet_id=sheet.id,
        exam_id=sheet.exam_id,
        exam_name=(exam.exam_name if exam else None),
        student_identifier=sheet.student_identifier,
        set_label=sheet.set_label_final,
        final_score=sheet.final_score or 0.0,
        percentage=sheet.percentage or 0.0,
        evaluated_at=sheet.evaluated_at,
    )


@router.get("/results/{sheet_id}", response_model=ResultDetailResponse)
async def result_detail(
    sheet_id: int,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    sheet = await db.get(ScannedSheet, sheet_id)
    if not sheet:
        raise HTTPException(status_code=404, detail="Sheet not found")
    if sheet.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    answers = (
        await db.execute(
            select(SheetAnswer)
            .where(SheetAnswer.sheet_id == sheet.id)
            .order_by(SheetAnswer.question_no.asc())
        )
    ).scalars().all()

    return ResultDetailResponse(
        sheet_id=sheet.id,
        exam_id=sheet.exam_id,
        student_identifier=sheet.student_identifier,
        set_label=sheet.set_label_final,
        summary=ResultSummary(
            correct=sheet.correct_count or 0,
            wrong=sheet.wrong_count or 0,
            unanswered=sheet.unanswered_count or 0,
            invalid=sheet.invalid_count or 0,
            raw_score=sheet.raw_score or 0.0,
            final_score=sheet.final_score or 0.0,
            percentage=sheet.percentage or 0.0,
        ),
        questions=[
            ResultQuestionItem(
                question_no=a.question_no,
                selected_option=a.selected_option,
                correct_option=a.correct_option,
                status=a.status,
                mark_awarded=a.mark_awarded,
            )
            for a in answers
        ],
    )


@router.delete("/results/{sheet_id}")
async def delete_result(
    sheet_id: int,
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    sheet = await db.get(ScannedSheet, sheet_id)
    if not sheet:
        raise HTTPException(status_code=404, detail="Sheet not found")
    if sheet.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Delete associated sheet answers
    await db.execute(SheetAnswer.__table__.delete().where(SheetAnswer.sheet_id == sheet.id))
    # Delete sheet
    await db.delete(sheet)
    await db.commit()
    
    return {"success": True, "message": "Result deleted successfully"}

@router.get("/results/export.csv")
async def export_results_csv(
    exam_id: Optional[int] = Query(default=None),
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ScannedSheet).where(ScannedSheet.user_id == current_user.id)
    if exam_id is not None:
        stmt = stmt.where(ScannedSheet.exam_id == exam_id)
    rows = (await db.execute(stmt.order_by(ScannedSheet.id.desc()))).scalars().all()

    lines = ["sheet_id,exam_id,student_identifier,set_label,correct,wrong,unanswered,invalid,final_score,percentage,evaluated_at"]
    for r in rows:
        lines.append(
            ",".join(
                [
                    str(r.id),
                    str(r.exam_id),
                    str(r.student_identifier or ""),
                    str(r.set_label_final or ""),
                    str(r.correct_count or 0),
                    str(r.wrong_count or 0),
                    str(r.unanswered_count or 0),
                    str(r.invalid_count or 0),
                    f"{(r.final_score or 0.0):.2f}",
                    f"{(r.percentage or 0.0):.2f}",
                    str(r.evaluated_at.isoformat() if r.evaluated_at else ""),
                ]
            )
        )

    content = "\n".join(lines).encode("utf-8")
    return StreamingResponse(
        BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


@router.get("/results/export.pdf")
async def export_results_pdf(
    exam_id: Optional[int] = Query(default=None),
    current_user: User = Depends(require_profile_complete),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ScannedSheet).where(ScannedSheet.user_id == current_user.id)
    if exam_id is not None:
        stmt = stmt.where(ScannedSheet.exam_id == exam_id)
    rows = (await db.execute(stmt.order_by(ScannedSheet.id.desc()).limit(300))).scalars().all()

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 12)
    c.drawString(15 * mm, height - 15 * mm, "OMR Results Export")
    c.setFont("Helvetica", 9)
    c.drawString(15 * mm, height - 21 * mm, f"Total rows: {len(rows)}")

    y = height - 30 * mm
    c.setFont("Helvetica-Bold", 8)
    c.drawString(15 * mm, y, "Sheet")
    c.drawString(30 * mm, y, "Student")
    c.drawString(70 * mm, y, "Set")
    c.drawString(85 * mm, y, "Correct")
    c.drawString(105 * mm, y, "Wrong")
    c.drawString(125 * mm, y, "Score")
    c.drawString(145 * mm, y, "%")

    c.setFont("Helvetica", 8)
    y -= 5 * mm
    for r in rows:
        if y < 15 * mm:
            c.showPage()
            y = height - 15 * mm
            c.setFont("Helvetica", 8)

        c.drawString(15 * mm, y, str(r.id))
        c.drawString(30 * mm, y, str(r.student_identifier or "-"))
        c.drawString(70 * mm, y, str(r.set_label_final or "-"))
        c.drawString(85 * mm, y, str(r.correct_count or 0))
        c.drawString(105 * mm, y, str(r.wrong_count or 0))
        c.drawString(125 * mm, y, f"{(r.final_score or 0.0):.2f}")
        c.drawString(145 * mm, y, f"{(r.percentage or 0.0):.2f}")
        y -= 4 * mm

    c.save()
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=results.pdf"},
    )
