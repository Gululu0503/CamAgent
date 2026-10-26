from __future__ import annotations

import pytest

from camagent.agent_stress import (
    _score_planning_case,
    _score_privacy_case,
    build_agent_contract_stress_cases,
)
from camagent.llm_agent import WildlifeAgent


class _NoNetworkLLM:
    def invoke(self, _messages):
        raise AssertionError("Deterministic contract tests must not call an external model.")


AGENT = WildlifeAgent(_NoNetworkLLM())
CASES = build_agent_contract_stress_cases()


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_agent_contract_stress_case(case):
    scorer = _score_privacy_case if case["category"] == "public_trace_privacy" else _score_planning_case
    result = scorer(AGENT, case)
    assert result["passed"], result
