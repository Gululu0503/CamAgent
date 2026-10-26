from __future__ import annotations

import argparse
import json

from .research_pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic CamAgent research pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
