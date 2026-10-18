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
