"""Celery tasks for scan-job processing."""

import asyncio
from datetime import datetime
from typing import Dict

from celery import Task
from sqlalchemy import func, select

from app.config import get_settings
from app.database import AsyncSessionLocal
from app.models import AnswerKey, Exam, ScanBatch, ScanJob, ScannedSheet, SheetAnswer
from app.services.scoring_service import normalize_omr_answers_to_options, score_sheet
from app.services.storage_service import resolve_storage_path
from app.services.token_service import debit_tokens_once
from app.utils.omr_engine import process_omr_image
from app.workers.celery_app import celery_app

settings = get_settings()
BENGALI_CODES = {"ক", "খ", "গ", "ঘ", "ঙ", "চ"}


def _answer_key_to_mapping(answer_key: AnswerKey) -> Dict[int, str]:
    raw = answer_key.mapping or {}
    if not raw and answer_key.answers:
        # legacy numeric answer key {"1": 0, "2": 2}
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


async def _resolve_answer_keys(exam: Exam) -> Dict[str, Dict[int, str]]:
    labels: Dict[str, Dict[int, str]] = {}
    for ak in exam.answer_keys:
        label = ak.set_code
        if ak.exam_set and ak.exam_set.set_label:
            label = ak.exam_set.set_label
        labels[label] = _answer_key_to_mapping(ak)
    return labels


async def _refresh_batch_status(db, batch_id: int) -> None:
    batch = await db.get(ScanBatch, batch_id)
    if not batch:
        return

    total = await db.scalar(select(func.count()).select_from(ScanJob).where(ScanJob.batch_id == batch_id))
    done = await db.scalar(
        select(func.count()).select_from(ScanJob).where(ScanJob.batch_id == batch_id, ScanJob.status == "done")
    )
    failed = await db.scalar(
        select(func.count()).select_from(ScanJob).where(ScanJob.batch_id == batch_id, ScanJob.status == "failed")
    )

    batch.total_files = int(total or 0)
    batch.processed_files = int((done or 0) + (failed or 0))

    if batch.processed_files == 0:
        batch.status = "queued"
    elif done and failed:
        batch.status = "partial_failed"
    elif failed and batch.processed_files == batch.total_files:
        batch.status = "failed"
    elif done and batch.processed_files == batch.total_files:
        batch.status = "completed"
    else:
        batch.status = "processing"


async def _process_scan_job(job_id: int) -> None:
    async with AsyncSessionLocal() as db:
        job = await db.get(ScanJob, job_id)
        if not job:
            return
        if job.status in {"done", "canceled"}:
            return

        job.status = "processing"
        job.attempts = (job.attempts or 0) + 1
        job.error_code = None
        job.error_message = None
        await db.commit()

    try:
        async with AsyncSessionLocal() as db:
            query = (
                select(ScanJob)
                .where(ScanJob.id == job_id)
                .options(
                    # lazy loading is sufficient under async relationship access in this context
                )
            )
            result = await db.execute(query)
            job = result.scalar_one()

            exam = await db.get(Exam, job.exam_id)
            if not exam:
                job.status = "failed"
                job.error_code = "exam_not_found"
                job.error_message = "Exam configuration not found"
                await _refresh_batch_status(db, job.batch_id)
                await db.commit()
                return

            await db.refresh(exam, attribute_names=["answer_keys", "exam_sets"])
            for ak in exam.answer_keys:
                if ak.set_id:
                    await db.refresh(ak, attribute_names=["exam_set"])

            file_path = resolve_storage_path(job.source_file_key)
            answer_keys_by_set = await _resolve_answer_keys(exam)
            use_bengali = any(label in BENGALI_CODES for label in answer_keys_by_set.keys())

            omr = process_omr_image(
                str(file_path),
                num_questions=exam.total_questions,
                use_bengali_set_codes=use_bengali,
            )

            if not omr.success:
                job.status = "failed"
                job.error_code = "omr_parse_failed"
                job.error_message = omr.error_message or "Failed to parse OMR sheet"
                await _refresh_batch_status(db, job.batch_id)
                await db.commit()
                return

            extracted = normalize_omr_answers_to_options(omr.answers)
            detected_set = omr.set_code if omr.set_code and omr.set_code != "?" else None

            if exam.has_set:
                final_set = detected_set
                if not final_set or final_set not in answer_keys_by_set:
                    final_set = next(iter(answer_keys_by_set.keys()), None)
            else:
                final_set = next(iter(answer_keys_by_set.keys()), None)

            if not final_set or final_set not in answer_keys_by_set:
                job.status = "failed"
                job.error_code = "answer_key_missing"
                job.error_message = "No answer key found for this sheet"
                await _refresh_batch_status(db, job.batch_id)
                await db.commit()
                return

            rows, summary = score_sheet(
                extracted_answers=extracted,
                answer_key=answer_keys_by_set[final_set],
                total_questions=exam.total_questions,
                mark_per_question=exam.mark_per_question or 1.0,
                negative_marking=bool(exam.negative_marking),
                negative_value=float(exam.negative_value or 0.0),
            )

            ledger = await debit_tokens_once(
                db=db,
                user_id=job.user_id,
                amount=max(1, settings.TOKEN_COST_PER_SUCCESS_SHEET),
                reason="scan_sheet_success",
                reference_type="scan_job",
                reference_id=str(job.id),
                metadata={"batch_id": job.batch_id},
            )
            if ledger is None:
                job.status = "failed"
                job.error_code = "insufficient_tokens"
                job.error_message = "Insufficient token balance"
                await _refresh_batch_status(db, job.batch_id)
                await db.commit()
                return

            sheet_result = await db.execute(select(ScannedSheet).where(ScannedSheet.scan_job_id == job.id))
            sheet = sheet_result.scalar_one_or_none()
            if not sheet:
                sheet = ScannedSheet(scan_job_id=job.id, user_id=job.user_id, exam_id=job.exam_id)
                db.add(sheet)
                await db.flush()

            # Replace old answers on re-run
            old_answers = await db.execute(select(SheetAnswer).where(SheetAnswer.sheet_id == sheet.id))
            for ans in old_answers.scalars().all():
                await db.delete(ans)

            sheet.detected_identifier = omr.roll_number or None
            sheet.student_identifier = omr.roll_number or None
            sheet.set_label_detected = detected_set
            sheet.set_label_final = final_set
            sheet.manual_override = False
            sheet.correct_count = summary["correct"]
            sheet.wrong_count = summary["wrong"]
            sheet.unanswered_count = summary["unanswered"]
            sheet.invalid_count = summary["invalid"]
            sheet.raw_score = summary["raw_score"]
            sheet.final_score = summary["final_score"]
            sheet.percentage = summary["percentage"]
            sheet.evaluated_at = datetime.utcnow()

            for row in rows:
                db.add(
                    SheetAnswer(
                        sheet_id=sheet.id,
                        question_no=row["question_no"],
                        selected_option=row["selected_option"],
                        correct_option=row["correct_option"],
                        status=row["status"],
                        mark_awarded=row["mark_awarded"],
                        is_overridden=False,
                    )
                )

            job.status = "done"
            job.token_charged = True
            job.token_ledger_id = ledger.id
            job.error_code = None
            job.error_message = None

            await _refresh_batch_status(db, job.batch_id)
            await db.commit()

    except Exception as exc:
        async with AsyncSessionLocal() as db:
            job = await db.get(ScanJob, job_id)
            if job and job.status != "done":
                job.status = "failed"
                job.error_code = "unexpected_error"
                job.error_message = str(exc)
                await _refresh_batch_status(db, job.batch_id)
                await db.commit()
        raise


class BaseScanTask(Task):
    autoretry_for = (RuntimeError,)
    retry_backoff = True
    retry_backoff_max = 60
    retry_jitter = True
    max_retries = 3


@celery_app.task(bind=True, base=BaseScanTask, name="app.workers.tasks_scan.process_scan_job")
def process_scan_job(self, job_id: int) -> None:
    asyncio.run(_process_scan_job(job_id))
