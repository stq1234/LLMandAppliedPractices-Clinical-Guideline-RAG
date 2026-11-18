"""Compose a test-set learned-policy trajectory without duplicate model calls.

When a threshold changes only a subset of adaptive routes, rerun those IDs and
reuse the paired baseline adaptive outputs for unchanged routes. Reused rows are
explicitly marked; no response is presented as a second independent run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reruns", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path, required=True)
    args = parser.parse_args()
    params = json.loads(args.params.read_text(encoding="utf-8"))
    base_rows = read_jsonl(args.baseline)
    rerun_rows = read_jsonl(args.reruns) if args.reruns.exists() else []
    baseline = {
        str(row["id"]): row for row in base_rows
        if row.get("split") == "test" and row.get("mode") == "adaptive"
    }
    changed = {
        str(row["id"]): row for row in rerun_rows
        if row.get("split") == "test" and row.get("mode") == "adaptive_learned"
    }
    test_ids = sorted({
        str(row["id"]) for row in base_rows if row.get("split") == "test"
    })
    if len(test_ids) != 26:
        raise ValueError(f"Expected 26 frozen test questions, found {len(test_ids)}")
    if set(baseline) != set(test_ids):
        raise ValueError("Every test question must have one baseline adaptive trajectory")
    if not set(changed).issubset(set(test_ids)):
        raise ValueError("Learned-policy rerun contains unknown test IDs")

    out_rows = []
    reused = 0
    for qid in test_ids:
        if qid in changed:
            row = dict(changed[qid])
            row["policy_replay"] = "actual_changed_route_run"
        else:
            row = dict(baseline[qid])
            row["mode"] = "adaptive_learned"
            row["params"] = params
            row["policy_replay"] = "reused_baseline_identical_route"
            diag = dict(row.get("diagnostics") or {})
            diag["params"] = params
            diag["policy_replay"] = "route unchanged by calibrated initial_threshold"
            row["diagnostics"] = diag
            reused += 1
        out_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        # Keep all raw baseline rows and add one complete learned-policy test
        # trajectory, so the standard summarizer can pair every mode by ID.
        for row in base_rows + out_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta = {
        "split": "test",
        "policy": params,
        "test_questions": len(test_ids),
        "actually_rerun_changed_routes": len(changed),
        "reused_identical_routes": reused,
        "changed_question_ids": sorted(changed),
        "baseline_source": str(args.baseline),
        "rerun_source": str(args.reruns),
        "interpretation": "A reused row has the same route and expected deterministic output as baseline; it is included for complete paired aggregation and is not counted as a new model run.",
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

