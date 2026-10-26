#!/usr/bin/env python3
"""Run the 60-case deterministic Agent stress test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camagent.agent_stress import run_agent_contract_stress_matrix
from camagent.llm_agent import WildlifeAgent


class _NoNetworkLLM:
    def invoke(self, _messages):
        raise AssertionError("The Agent stress test must not invoke an external language model.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/agent_stress/stress_60",
        help="Directory for Agent-stress JSON/CSV records and manifest.",
    )
    args = parser.parse_args()
    result = run_agent_contract_stress_matrix(
        WildlifeAgent(_NoNetworkLLM()),
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
    return 0 if result["manifest"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
