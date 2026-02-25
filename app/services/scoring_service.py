"""Scoring helpers for sheet evaluation."""

from typing import Dict, List, Optional, Tuple, Union


ExtractedAnswer = Union[str, List[str], None]
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


def normalize_option(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    return OPTION_ALIASES.get(raw.upper()) or OPTION_ALIASES.get(raw)


def score_sheet(
    extracted_answers: Dict[int, ExtractedAnswer],
    answer_key: Dict[int, str],
    total_questions: int,
    mark_per_question: float = 1.0,
    negative_marking: bool = False,
    negative_value: float = 0.25,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    correct = wrong = unanswered = invalid = 0
    total = 0.0

    for q in range(1, total_questions + 1):
        selected = extracted_answers.get(q)
        expected = normalize_option(answer_key.get(q))
        status = "unanswered"
        mark = 0.0

        if selected is None:
            status = "unanswered"
            unanswered += 1
        elif isinstance(selected, list):
            status = "invalid"
            invalid += 1
            mark = -negative_value if negative_marking else 0.0
        elif normalize_option(selected) == expected:
            status = "correct"
            correct += 1
            mark = mark_per_question
        else:
            status = "wrong"
            wrong += 1
            mark = -negative_value if negative_marking else 0.0

        total += mark
        rows.append(
            {
                "question_no": q,
                "selected_option": normalize_option(selected) if isinstance(selected, str) else None,
                "correct_option": expected,
                "status": status,
                "mark_awarded": round(mark, 4),
            }
        )

    final_score = max(0.0, round(total, 4))
    denominator = max(total_questions * mark_per_question, 1)
    percentage = round((final_score / denominator) * 100.0, 2)

    summary = {
        "correct": correct,
        "wrong": wrong,
        "unanswered": unanswered,
        "invalid": invalid,
        "raw_score": round(total, 4),
        "final_score": final_score,
        "percentage": percentage,
    }
    return rows, summary


def normalize_omr_answers_to_options(answers: List[int]) -> Dict[int, Optional[str]]:
    """Convert numeric OMR outputs to A/B/C/D style options."""
    options = ["A", "B", "C", "D", "E"]
    converted: Dict[int, Optional[str]] = {}
    for idx, val in enumerate(answers, start=1):
        if isinstance(val, int) and 0 <= val < len(options):
            converted[idx] = options[val]
        else:
            converted[idx] = None
    return converted
