"""LLM judge tests. They need Apple Silicon, mlx-lm, and a model, so CI skips them.

    VENDORMATCH_LLM_MODEL=mlx-community/Qwen3-14B-4bit uv run pytest -m llm
"""

import os

import pytest

from vendormatch.llm import describe, user_message

MODEL = os.environ.get("VENDORMATCH_LLM_MODEL")


def test_prompt_lists_only_present_fields():
    text = describe({"name": "SQ *BLUE BOTTLE", "city": "OAKLAND", "street": "", "source": "card"}, "A")
    assert text == 'Record A (card feed): name "SQ *BLUE BOTTLE"; city OAKLAND'
    assert user_message({"name": "x"}, {"name": "y"}).count("\n") == 1


@pytest.mark.llm
@pytest.mark.skipif(not MODEL, reason="set VENDORMATCH_LLM_MODEL to run")
def test_judge_obvious_cases():
    from vendormatch.llm import LLMJudge

    judge = LLMJudge(MODEL)
    yes = judge.judge({"name": "IBM"}, {"name": "INTERNATIONAL BUSINESS MACHINES CORP", "city": "ARMONK"})
    no = judge.judge(
        {"name": "Delta Air Lines Inc", "city": "Atlanta"}, {"name": "Delta Apparel Inc", "city": "Duluth"}
    )
    assert yes.match and yes.p_match > 0.5 and yes.reason
    assert not no.match


def test_combined_score_is_monotone_in_model_score_and_verdict():
    from vendormatch.llm import CALIBRATION_PATH, combined_score

    if not CALIBRATION_PATH.exists():
        pytest.skip("no calibration file")
    lo_no, hi_no = combined_score(0.05, False), combined_score(0.8, False)
    lo_yes, hi_yes = combined_score(0.05, True), combined_score(0.8, True)
    assert 0 < lo_no < hi_no < 1 and 0 < lo_yes < hi_yes < 1
    assert lo_yes > lo_no and hi_yes > hi_no
