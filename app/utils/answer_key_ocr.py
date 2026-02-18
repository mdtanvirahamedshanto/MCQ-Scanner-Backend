"""
Parse answer key from image (handwritten or printed).
Supports: ১. ক, 1. A, 1. 1 (numeric) - Bengali + English.
"""

import re
from typing import Dict, Optional, Tuple
from pathlib import Path

# Bengali numerals ০-৯ -> 0-9
BN_NUM = {"০": 0, "১": 1, "২": 2, "৩": 3, "৪": 4, "৫": 5, "৬": 6, "৭": 7, "৮": 8, "৯": 9}
# Bengali option letters ক, খ, গ, ঘ -> A, B, C, D (0, 1, 2, 3)
BN_OPT = {"ক": "A", "খ": "B", "গ": "C", "ঘ": "D"}
# Numeric 1,2,3,4 -> A,B,C,D
NUM_OPT = {"1": "A", "2": "B", "3": "C", "4": "D"}


def _to_question_num(s: str) -> Optional[int]:
    """Convert '১', '১২', '1', '10' etc to int."""
    s = str(s).strip()
    if not s:
        return None
    if s in BN_NUM:
        return BN_NUM[s]
    if all(c in BN_NUM for c in s):
        num = 0
        for c in s:
            num = num * 10 + BN_NUM[c]
        return num
    try:
        return int(s)
    except ValueError:
        return None


def _to_option(s: str) -> Optional[str]:
    """Convert 'ক', 'A', '1' etc to A|B|C|D."""
    s = str(s).strip()
    if not s:
        return None
    if s in BN_OPT:
        return BN_OPT[s]
    s_upper = s.upper()
    if s_upper in ("A", "B", "C", "D"):
        return s_upper
    if s in NUM_OPT:
        return NUM_OPT[s]
    return None


def _parse_line(line: str) -> Optional[Tuple[int, str]]:
    """
    Parse lines like: "১. ক", "1. A", "1.1", "2. B", " 3 . C "
    Returns (question_num, option) or None.
    """
    line = line.strip()
    if not line or len(line) < 3:
        return None

    # Pattern: number + optional dot/period + optional space + option
    # Match: ১. ক, 1. A, 1.A, 2. B, 3.2, ১.ক (no space), etc.
    m = re.match(r"^([০-৯0-9]+)\s*[\.\:\-]\s*(.+)$", line)
    if not m:
        return None

    q_str, opt_str = m.group(1), m.group(2).strip()
    q_num = _to_question_num(q_str)
    opt = _to_option(opt_str)
    if q_num is not None and opt is not None and 1 <= q_num <= 100:
        return (q_num, opt)
    return None


def parse_answer_key_from_text(lines: list) -> Dict[int, str]:
    """Parse list of text lines into {q_num: "A"|"B"|"C"|"D"}."""
    result: Dict[int, str] = {}
    for line in lines:
        parsed = _parse_line(line)
        if parsed:
            q_num, opt = parsed
            result[q_num] = opt
    return result


def parse_answer_key_image(image_path: str) -> Tuple[Dict[int, str], str]:
    """
    OCR image and parse answer key.
    Returns (answers_dict, error_message).
    """
    try:
        import easyocr
    except ImportError:
        return {}, "easyocr not installed. Run: pip install easyocr"

    path = Path(image_path)
    if not path.exists():
        return {}, f"File not found: {image_path}"

    try:
        reader = easyocr.Reader(["en", "bn"], gpu=False, verbose=False)
        detections = reader.readtext(str(path))
    except Exception as e:
        return {}, f"OCR failed: {str(e)}"

    # Sort by Y position (top to bottom)
    def center_y(bbox):
        return (bbox[0][1] + bbox[2][1]) / 2

    detections = sorted(detections, key=lambda x: center_y(x[0]))

    lines = []
    for (bbox, text, conf) in detections:
        if text and conf > 0.2:
            t = str(text).strip()
            if t:
                lines.append(t)

    # Build text - each detection may be "1. A" or "১. ক" or split
    full_text = "\n".join(lines)
    all_lines = re.split(r"[\n,;]+", full_text)

    answers = parse_answer_key_from_text(all_lines)
    return answers, ""
