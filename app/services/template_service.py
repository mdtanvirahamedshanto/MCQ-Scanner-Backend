"""OMR template PDF generator for v1."""

from io import BytesIO
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from app.models import Exam


def generate_omr_template_pdf(exam: Exam, set_labels: List[str]) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, f"OMR Sheet: {exam.exam_name or exam.title or 'Exam'}")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, height - 27 * mm, f"Subject: {exam.subject_name or exam.subject_code or '-'}")
    c.drawString(20 * mm, height - 33 * mm, f"Total Questions: {exam.total_questions}")

    c.rect(20 * mm, height - 50 * mm, 60 * mm, 10 * mm)
    c.drawString(22 * mm, height - 45 * mm, "Student ID / Roll")

    c.drawString(90 * mm, height - 45 * mm, "Set")
    x = 105 * mm
    for label in set_labels:
        c.circle(x, height - 45 * mm, 3 * mm)
        c.drawString(x + 5 * mm, height - 46 * mm, str(label))
        x += 20 * mm

    start_y = height - 65 * mm
    q_per_col = 25
    cols = max(1, (exam.total_questions + q_per_col - 1) // q_per_col)
    options = ["A", "B", "C", "D", "E"][: max(4, min(5, exam.options_per_question or 4))]

    for q in range(1, exam.total_questions + 1):
        col = (q - 1) // q_per_col
        row = (q - 1) % q_per_col
        x0 = 20 * mm + col * 60 * mm
        y0 = start_y - row * 8 * mm

        c.setFont("Helvetica", 8)
        c.drawString(x0, y0, f"{q:02d}")

        ox = x0 + 12 * mm
        for option in options:
            c.circle(ox, y0 + 1.5 * mm, 1.8 * mm)
            c.drawString(ox + 3 * mm, y0, option)
            ox += 10 * mm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
