from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_old_m1_predicate_metrics_cannot_be_presented_as_current_results() -> None:
    resume = (REPO_ROOT / "docs" / "RESUME_DRAFT_TEMPLATE.md").read_text(
        encoding="utf-8"
    )
    storyboard = (REPO_ROOT / "docs" / "DEMO_STORYBOARD.md").read_text(encoding="utf-8")

    assert "M1-safe" not in resume
    assert "CURRENT-STRICT-EVIDENCE-PENDING" in resume
    assert "M1-HISTORICAL-SUCCESS" in resume
    assert "[VERIFIED-LOCAL] 面向机器人学习" not in resume

    assert "`Fixed seeds: 99/100 success`" not in storyboard
    assert "M1 scripted expert — local acceptance" not in storyboard
    assert "HISTORICAL-LOCAL / old center predicate" in storyboard
    assert "[PENDING] Current strict M1 result" in storyboard
