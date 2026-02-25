from app.services.scoring_service import score_sheet


def test_score_sheet_without_negative_marking():
    rows, summary = score_sheet(
        extracted_answers={1: "A", 2: "B", 3: None},
        answer_key={1: "A", 2: "C", 3: "D"},
        total_questions=3,
        mark_per_question=1.0,
        negative_marking=False,
        negative_value=0.25,
    )

    assert summary["correct"] == 1
    assert summary["wrong"] == 1
    assert summary["unanswered"] == 1
    assert summary["invalid"] == 0
    assert summary["final_score"] == 1.0
    assert len(rows) == 3


def test_score_sheet_with_negative_marking_and_invalid():
    rows, summary = score_sheet(
        extracted_answers={1: "A", 2: ["B", "C"], 3: "C"},
        answer_key={1: "A", 2: "B", 3: "D"},
        total_questions=3,
        mark_per_question=1.0,
        negative_marking=True,
        negative_value=0.25,
    )

    assert summary["correct"] == 1
    assert summary["invalid"] == 1
    assert summary["wrong"] == 1
    assert summary["final_score"] == 0.5
    assert any(r["status"] == "invalid" for r in rows)


def test_score_sheet_accepts_bengali_answer_options():
    _, summary = score_sheet(
        extracted_answers={1: "ক", 2: "খ", 3: "গ"},
        answer_key={1: "A", 2: "খ", 3: "D"},
        total_questions=3,
        mark_per_question=1.0,
        negative_marking=False,
        negative_value=0.25,
    )

    assert summary["correct"] == 2
    assert summary["wrong"] == 1
    assert summary["final_score"] == 2.0
