"""Select test questions whose initial route changes under a calibrated threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate import read_jsonl


DEFAULT_THRESHOLD = 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--ids-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.05 <= args.threshold <= DEFAULT_THRESHOLD:
        raise ValueError("Learned threshold must be in [0.05, 0.35]")
    rows = [
        row for row in read_jsonl(args.input)
        if row.get("split") == "test" and row.get("mode") == "adaptive" and row.get("status") == "ok"
    ]
    selected: list[dict[str, Any]] = []
    for row in rows:
        f = row.get("features", {})
        coverage = float(f.get("supplied_coverage", 0.0))
        eligible = (
            float(f.get("supplied_count", 0.0)) > 0
            and float(f.get("supplied_scope_fraction", 0.0)) == 1.0
            and float(f.get("bp_scope", 1.0)) == 1.0
            and float(f.get("personalized_request", 0.0)) == 0.0
        )
        if eligible and args.threshold <= coverage < DEFAULT_THRESHOLD:
            selected.append(row)
    selected.sort(key=lambda row: str(row["id"]))
    ids = [str(row["id"]) for row in selected]
    args.ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.ids_out.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    report = {
        "split": "test",
        "source_mode": "adaptive",
        "default_initial_threshold": DEFAULT_THRESHOLD,
        "calibrated_initial_threshold": args.threshold,
        "changed_route_condition": "eligible supplied evidence and calibrated_threshold <= supplied_coverage < 0.35",
        "test_adaptive_rows": len(rows),
        "n_route_changed_questions": len(ids),
        "question_ids": ids,
        "features": [{"id": str(r["id"]), **r["features"]} for r in selected],
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(ids)} IDs: {' '.join(ids)}")


if __name__ == "__main__":
    main()

