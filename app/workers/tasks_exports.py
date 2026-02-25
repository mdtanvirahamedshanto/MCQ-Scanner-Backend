"""Placeholder for async export tasks."""

from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks_exports.generate_export")
def generate_export(payload: dict) -> dict:
    # Reserved for heavier async export workflows.
    return {"status": "queued", "payload": payload}
