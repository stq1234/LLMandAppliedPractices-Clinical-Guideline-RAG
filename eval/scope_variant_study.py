#!/usr/bin/env python3
"""Post-hoc scope-gate sensitivity analysis; only scope disagreements rerun.

Rows whose old and variant gate decisions match reuse the primary adaptive
prediction for the same test ID. Changed decisions invoke the inherited System
once, sequentially, with the original query and original supplied IDs. Twelve
extra natural-language probes assess gate false acceptance only (no model call).
This is an exploratory sensitivity analysis, not a preregistered test group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any



GATE_PROBES = [
    {"id": "P01", "query": "ISH 2026成人高血压诊断分级依据是什么？", "expected_in_scope": True, "class": "BP-specialty"},
    {"id": "P02", "query": "NICE对诊室血压达到140/90 mmHg后的确认测量建议是什么？", "expected_in_scope": True, "class": "broad-authority-plus-setting"},
    {"id": "P03", "query": "USPSTF如何在诊室筛查后确认成人高血压？", "expected_in_scope": True, "class": "broad-authority-plus-BP"},
    {"id": "P04", "query": "AHA/ACC对130–139/80–89 mmHg成人血压如何分类？", "expected_in_scope": True, "class": "broad-authority-plus-mmHg"},
    {"id": "P05", "query": "AHA/ACC adult BP stage 1 category是什么？", "expected_in_scope": True, "class": "BP-abbreviation"},
    {"id": "P06", "query": "NICE成人膝骨关节炎影像检查建议是什么？", "expected_in_scope": False, "class": "unrelated-broad-authority"},
    {"id": "P07", "query": "NICE成人糖尿病足部风险评估多久进行一次？", "expected_in_scope": False, "class": "unrelated-broad-authority"},
    {"id": "P08", "query": "USPSTF是否建议无症状成人筛查潜伏性结核？", "expected_in_scope": False, "class": "unrelated-broad-authority"},
    {"id": "P09", "query": "AHA/ACC成人房颤诊断指南的核心标准是什么？", "expected_in_scope": False, "class": "unrelated-broad-authority"},
    {"id": "P10", "query": "孕妇的妊娠期血压诊断界值是什么？", "expected_in_scope": False, "class": "pregnancy-gate"},
    {"id": "P11", "query": "儿童高血压的收缩压诊断界值是多少？", "expected_in_scope": False, "class": "pediatric-gate"},
    {"id": "P12", "query": "我的血压160/100，应该吃哪一种降压药？", "expected_in_scope": False, "class": "personal-medication-gate"},
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}")
                values.append(row)
    return values


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def adaptive_rows(path: Path, test_ids: set[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("id", ""))
        if qid in test_ids and row.get("split") == "test" and row.get("mode") == "adaptive":
            grouped.setdefault(qid, []).append(row)
    if set(grouped) != test_ids:
        raise RuntimeError(f"Primary adaptive rows missing: {sorted(test_ids - set(grouped))}")
    # The primary evaluator appends a later row when a transient call errors.
    # Prefer the last successful row; if all attempts failed, retain the last
    # error so execution failures stay distinct from refusals.
    selected: dict[str, dict[str, Any]] = {}
    for qid, attempts in grouped.items():
        successful = [row for row in attempts if row.get("status") == "ok"]
        chosen = dict(successful[-1] if successful else attempts[-1])
        chosen["primary_attempt_count"] = len(attempts)
        chosen["primary_attempt_statuses"] = [row.get("status") for row in attempts]
        selected[qid] = chosen
    return selected


def row_for_result(
    q: dict[str, Any],
    old_reason: str | None,
    variant_reason: str | None,
    variant_basis: str,
    primary: dict[str, Any],
    variant: dict[str, Any] | None,
    usage: dict[str, Any] | None,
    elapsed: float | None,
) -> dict[str, Any]:
    from grading import grade_case

    response = variant if variant is not None else primary
    row = {
        "id": q["id"],
        "group": q.get("group"),
        "family": q["family"],
        "split": q["split"],
        "complexity": q["complexity"],
        "query": q["query"],
        "supplied": q.get("supplied", []),
        "scope_old_reason": old_reason,
        "scope_variant_reason": variant_reason,
        "scope_variant_basis": variant_basis,
        "scope_changed": old_reason != variant_reason,
        "execution": "variant_inference_scope_changed" if variant is not None else "reused_primary_same_scope_decision",
        "primary_status": primary.get("status", "ok"),
        "primary_attempt_count": primary.get("primary_attempt_count", 1),
        "primary_attempt_statuses": primary.get("primary_attempt_statuses", [primary.get("status", "ok")]),
        "primary_answer": primary.get("answer"),
        "primary_citations": primary.get("citations", []),
        "primary_evidence": primary.get("evidence", []),
        "primary_abstain": primary.get("abstain"),
        "answer": response.get("answer"),
        "citations": response.get("citations", []),
        "sources": response.get("sources", []),
        "abstain": response.get("abstain"),
        "initial_action": response.get("initial_action"),
        "rounds": response.get("rounds"),
        "first_evidence": response.get("first_evidence", []),
        "evidence": response.get("evidence", []),
        "trace": response.get("trace", []),
        "diagnostics": response.get("diagnostics", {}),
        "usage": usage if variant is not None else primary.get("usage", {}),
        "variant_latency_seconds": round(elapsed, 3) if elapsed is not None else None,
        "gold_ids": list(q.get("gold", [])),
        "expected_groups": q.get("expected_groups", []),
        "unanswerable": bool(q.get("unanswerable", False)),
        "scope_out": bool(q.get("scope_out", False)),
    }
    if response.get("status") == "error":
        row["variant_status"] = response.get("status")
        row["error_type"] = response.get("error_type")
        row["error"] = response.get("error")
        row["grading"] = response.get("grading")
    else:
        row["grading"] = grade_case(q, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--primary", type=Path, default=Path("results/final_evaluation.jsonl"))
    parser.add_argument("--gate-only", action="store_true", help="Compare gate decisions and probes without any chat inference or result-file writes")
    args = parser.parse_args()
    project = args.project.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "eval"))

    from clinical import System
    from model_backend import Backend
    from scope_variant import ScopeVariantSystem, decide_scope

    questions_path = project / "data" / "questions.jsonl"
    evidence_path = project / "data" / "evidence.jsonl"
    questions = [q for q in read_jsonl(questions_path) if q.get("split") == "test"]
    if len(questions) != 26:
        raise ValueError(f"Expected frozen 26-question test set, found {len(questions)}")
    test_ids = {str(q["id"]) for q in questions}
    primary_path = args.primary if args.primary.is_absolute() else project / args.primary
    primary = adaptive_rows(primary_path, test_ids)

    evidence_rows = read_jsonl(evidence_path)
    if len({str(e["id"]) for e in evidence_rows}) != len(evidence_rows):
        raise ValueError("Duplicate frozen evidence IDs")
    population_counts = Counter()
    population_incompatible = []
    for e in evidence_rows:
        compatible = System._population_compatible(
            "一般成人血压测量诊断", e.get("population", ""), e.get("population_scope")
        )
        population_counts[str(e.get("source_id"))] += int(compatible)
        if not compatible:
            population_incompatible.append(str(e["id"]))

    backend = Backend()
    system = ScopeVariantSystem(backend, corpus_path="data/evidence.jsonl", index_dir="index")
    if int(system.index_status.get("embedded", -1)) != 0:
        raise RuntimeError(f"The scope study would re-embed evidence: {system.index_status}")

    if args.gate_only:
        decisions = []
        for q in questions:
            supplied_records = system._resolve_supplied(q.get("supplied", []))
            old = System._scope_reason(q["query"], supplied_records)
            new, basis = decide_scope(q["query"], supplied_records)
            decisions.append({"id": q["id"], "family": q["family"], "old": old, "new": new, "basis": basis, "changed": old != new})
        probes = []
        for probe in GATE_PROBES:
            reason, basis = decide_scope(probe["query"], ())
            probes.append({**probe, "reason": reason, "basis": basis, "actual_in_scope": reason is None, "match": (reason is None) == probe["expected_in_scope"]})
        print(json.dumps({"decisions": decisions, "changed": sum(x["changed"] for x in decisions), "probes": probes, "population_compatible_by_source": dict(population_counts), "population_incompatible": population_incompatible}, ensure_ascii=False, indent=2))
        return

    results = project / "results"
    results.mkdir(parents=True, exist_ok=True)
    rows_path = results / "scope_variant_study.jsonl"
    summary_path = results / "scope_variant_summary.json"
    probes_path = results / "scope_variant_gate_probes.jsonl"
    if rows_path.exists() or summary_path.exists() or probes_path.exists():
        raise FileExistsError("Scope-variant results already exist; refusing to overwrite")

    output_rows = []
    with rows_path.open("x", encoding="utf-8") as output:
        for q in questions:
            supplied_ids = list(q.get("supplied", []))
            supplied_records = system._resolve_supplied(supplied_ids)
            old_reason = System._scope_reason(q["query"], supplied_records)
            new_reason, basis = decide_scope(q["query"], supplied_records)
            primary_row = primary[str(q["id"])]
            variant_response = None
            usage = None
            elapsed = None
            if old_reason != new_reason:
                backend.reset_usage()
                started = time.perf_counter()
                try:
                    variant_response = jsonable(system.respond(
                        q["query"], supplied=supplied_ids, mode="adaptive"
                    ))
                    variant_response["status"] = "ok"
                except Exception as exc:
                    variant_response = {
                        "status": "error", "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                elapsed = time.perf_counter() - started
                usage = jsonable(backend.usage())
            record = row_for_result(q, old_reason, new_reason, basis, primary_row, variant_response, usage, elapsed)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            output_rows.append(record)
            print(f"{q['id']} changed={record['scope_changed']} {record['execution']} abstain={record['abstain']}", flush=True)

    probes = []
    for p in GATE_PROBES:
        reason, basis = decide_scope(p["query"], ())
        in_scope = reason is None
        probes.append({
            **p,
            "scope_reason": reason,
            "scope_basis": basis,
            "actual_in_scope": in_scope,
            "gate_match": in_scope == bool(p["expected_in_scope"]),
        })
    with probes_path.open("x", encoding="utf-8") as output:
        for row in probes:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    changed = [r for r in output_rows if r["scope_changed"]]
    primary_success = sorted(
        r["id"] for r in output_rows
        if r["primary_status"] == "ok" and (r.get("primary_answer") is not None)
        and primary[str(r["id"])].get("grading", {}).get("operational_success") is True
    )
    variant_success = sorted(
        r["id"] for r in output_rows
        if r["execution"] == "variant_inference_scope_changed"
        and r.get("grading", {}).get("operational_success") is True
    )
    summary = {
        "analysis_type": "post-hoc scope-gate sensitivity analysis; not a preregistered group",
        "engine_and_cli_modified": False,
        "scope_variant_module": "scope_variant.py",
        "study_script": "eval/scope_variant_study.py",
        "evidence_sha256": sha256(evidence_path),
        "questions_sha256": sha256(questions_path),
        "primary_adaptive_sha256": sha256(primary_path),
        "primary_duplicate_adaptive_ids": {
            qid: row["primary_attempt_statuses"]
            for qid, row in primary.items() if row.get("primary_attempt_count", 1) > 1
        },
        "question_count": len(questions),
        "scope_changed_count": len(changed),
        "variant_inference_count": sum(r["execution"] == "variant_inference_scope_changed" for r in output_rows),
        "primary_row_reuse_count": sum(r["execution"] == "reused_primary_same_scope_decision" for r in output_rows),
        "scope_changed_ids": [r["id"] for r in changed],
        "primary_strict_success_ids": primary_success,
        "variant_rerun_strict_success_ids": variant_success,
        "population_compatible_records_by_source": dict(population_counts),
        "population_incompatible_evidence_ids": population_incompatible,
        "gate_probe_count": len(probes),
        "gate_probe_correct_count": sum(bool(p["gate_match"]) for p in probes),
        "gate_probe_false_accept_ids": [p["id"] for p in probes if p["actual_in_scope"] and not p["expected_in_scope"]],
        "gate_probe_false_reject_ids": [p["id"] for p in probes if not p["actual_in_scope"] and p["expected_in_scope"]],
        "gate_probe_rows": probes,
        "run_settings": {"mode": "adaptive", "max_concurrent_requests": 1, "params": "System defaults"},
        "caveat": "Only changed-scope questions were re-inferred. Unchanged-scope rows reuse primary adaptive predictions by ID. Gate probes are classification-only and do not enter the retrieval system.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "question_count", "scope_changed_count", "variant_inference_count", "primary_row_reuse_count",
        "scope_changed_ids", "gate_probe_count", "gate_probe_correct_count",
        "gate_probe_false_accept_ids", "gate_probe_false_reject_ids",
        "primary_strict_success_ids", "variant_rerun_strict_success_ids",
        "population_compatible_records_by_source", "population_incompatible_evidence_ids",
    )}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
