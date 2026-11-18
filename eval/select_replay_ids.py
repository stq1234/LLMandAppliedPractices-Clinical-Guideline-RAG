"""Select dev cases where a lower initial_threshold can change adaptive routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate import read_jsonl


DEFAULT_THRESHOLD = 0.35
MIN_THRESHOLD = 0.05


def eligible(features: dict[str, Any]) -> bool:
    coverage = float(features.get("supplied_coverage", 0.0))
    return (
        MIN_THRESHOLD <= coverage < DEFAULT_THRESHOLD
        and float(features.get("supplied_count", 0.0)) > 0
        and float(features.get("supplied_scope_fraction", 0.0)) == 1.0
        and float(features.get("bp_scope", 1.0)) == 1.0
        and float(features.get("personalized_request", 0.0)) == 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ids-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row for row in read_jsonl(args.input)
        if row.get("split") == "dev" and row.get("mode") == "adaptive" and row.get("status") == "ok"
    ]
    by_id = {str(row["id"]): row for row in rows}
    selected = sorted(qid for qid, row in by_id.items() if eligible(row.get("features", {})))
    details = [
        {"id": qid, **by_id[qid]["features"]}
        for qid in selected
    ]
    args.ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.ids_out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    args.report_out.write_text(json.dumps({
        "source_split": "dev",
        "source_mode": "adaptive",
        "threshold_interval": [MIN_THRESHOLD, DEFAULT_THRESHOLD],
        "route_change_condition": "supplied_count>0, supplied_scope_fraction==1, bp_scope==1, personalized_request==0, and supplied_coverage in [0.05,0.35)",
        "n_dev_adaptive_rows": len(rows),
        "n_replays": len(selected),
        "replay_ids": selected,
        "features": details,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} IDs: {' '.join(selected)}")


if __name__ == "__main__":
    main()

