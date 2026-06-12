from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import ExperimentRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run clean GEPA core experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and strategy without launching an experiment.")
    parser.add_argument("--print-plan", action="store_true", help="Print resolved strategy plan as JSON.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    plan = runner.plan()
    if args.print_plan:
        print(json.dumps(plan.__dict__, indent=2, default=str))
        return 0

    result = runner.run(dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "status": result.status,
                "run_dir": result.run_dir,
                "final_score": result.final_score,
                "search_iterations": result.search_iterations,
                "metric_calls": result.metric_calls,
                "message": result.message,
                "accounting": result.accounting.as_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

