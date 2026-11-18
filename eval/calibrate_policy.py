"""Calibrate the reuse threshold from dev trajectories; never reads test labels."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate import read_jsonl
from grading import risk_cost


DEFAULTS = {
    "initial_threshold": 0.35,
    "supplement_threshold": 0.25,
    "multi_threshold": 0.50,
    "initial_k": 5,
    "supplement_k": 5,
    "max_evidence": 8,
    "lexical_weight": 0.15,
    "force_action": None,
}


def candidate_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    values = sorted(
        float(r.get("features", {}).get("supplied_coverage", 0.0))
        for r in rows
        if float(r.get("features", {}).get("supplied_coverage", 0.0)) <= DEFAULTS["initial_threshold"]
    )
    candidates = {DEFAULTS["initial_threshold"]}
    candidates.update(round(x, 3) for x in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30))
    if not values:
        return sorted(candidates)
    for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        pos = round(q * (len(values) - 1))
        value = round(values[pos], 3)
        if 0.05 <= value <= DEFAULTS["initial_threshold"]:
            candidates.add(value)
    # Midpoints create the actual decision boundaries between observed values.
    unique = sorted(set(values))
    for left, right in zip(unique, unique[1:]):
        midpoint = round((left + right) / 2.0, 3)
        if 0.05 <= midpoint <= DEFAULTS["initial_threshold"]:
            candidates.add(midpoint)
    return sorted(candidates)


def family_mean(losses: list[tuple[str, float]]) -> tuple[float, float, dict[str, float]]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for family, loss in losses:
        if family in {"review_evidence", "evidence_review"}:
            family = "evidence_review"
        by_family[family].append(loss)
    means = {family: sum(vals) / len(vals) for family, vals in by_family.items()}
    values = list(means.values())
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        se = math.sqrt(variance / len(values))
    else:
        se = 0.0
    return mean, se, means


def row_loss(row: dict[str, Any]) -> float:
    if row.get("status") != "ok":
        return 250.0
    return risk_cost(row.get("grading") or {}, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="combined JSONL containing dev reuse/adaptive trajectories")
    parser.add_argument("--params-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    records = [r for r in read_jsonl(args.input) if r.get("split") == "dev"]
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    question_by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("mode") not in ("adaptive_force_reuse", "adaptive"):
            continue
        by_id[str(row["id"])][str(row["mode"])] = row
        question_by_id[str(row["id"])] = row
    paired = [question_by_id[qid] for qid, modes in by_id.items() if "adaptive" in modes]
    if len(paired) < 12:
        raise ValueError(f"Need at least 12 paired dev cases; found {len(paired)}")
    if any("supplied_coverage" not in r.get("features", {}) for r in paired):
        raise ValueError("Dev logs lack features.supplied_coverage")

    candidates = candidate_thresholds(paired)
    results = []
    for threshold in candidates:
        losses: list[tuple[str, float]] = []
        actions: dict[str, str] = {}
        for row in paired:
            qid = str(row["id"])
            modes = by_id[qid]
            coverage = float(row["features"].get("supplied_coverage", 0.0))
            features = row.get("features", {})
            can_reuse = (
                float(features.get("supplied_count", 0.0)) > 0
                and float(features.get("supplied_scope_fraction", 0.0)) == 1.0
                and float(features.get("bp_scope", 1.0)) == 1.0
                and float(features.get("personalized_request", 0.0)) == 0.0
                and coverage >= threshold
            )
            action = (
                "adaptive_force_reuse"
                if can_reuse and coverage < DEFAULTS["initial_threshold"]
                else "adaptive"
            )
            if action not in modes:
                raise ValueError(
                    f"Missing dev action trajectory {action!r} for {qid}; "
                    "replay only the queries whose initial_threshold route changes"
                )
            actions[qid] = action
            losses.append((str(row.get("family") or "unknown"), row_loss(modes[action])))
        mean, se, family_losses = family_mean(losses)
        results.append({
            "initial_threshold": threshold,
            "family_macro_risk_cost": mean,
            "family_se": se,
            "family_costs": family_losses,
            "action_counts": {a: list(actions.values()).count(a) for a in ("adaptive_force_reuse", "adaptive")},
        })

    best = min(results, key=lambda r: r["family_macro_risk_cost"])
    eligible = [
        r for r in results
        if r["family_macro_risk_cost"] <= best["family_macro_risk_cost"] + best["family_se"]
    ]
    chosen = min(
        eligible,
        key=lambda r: (
            abs(r["initial_threshold"] - DEFAULTS["initial_threshold"]),
            r["family_macro_risk_cost"],
        ),
    )

    params = dict(DEFAULTS)
    params["initial_threshold"] = chosen["initial_threshold"]
    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.params_out.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "training_split": "dev",
        "paired_questions": len(paired),
        "families": sorted({str(r.get("family") or "unknown") for r in paired}),
        "features_used": ["supplied_coverage"],
        "candidate_policy": (
            "adaptive reuse route when supplied_count>0 and supplied_scope_fraction==1 "
            "and supplied_coverage>=initial_threshold; otherwise default adaptive retrieval route"
        ),
        "supplement_threshold": DEFAULTS["supplement_threshold"],
        "multi_threshold": DEFAULTS["multi_threshold"],
        "objective": (
            "100*answer_error + 100*unsupported_answer + 30*wrong_refusal; "
            "plus 0.002*tokens + 0.5*retrieval_rounds; latency is reported separately"
        ),
        "aggregation": "family-macro mean; one-standard-error rule; threshold search restricted to [0.05, default 0.35]",
        "minimum_candidate": best,
        "selected_candidate": chosen,
        "all_candidates": results,
        "params_file": str(args.params_out),
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "paired_questions", "features_used", "candidate_policy", "objective",
        "aggregation", "minimum_candidate", "selected_candidate", "params_file"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

