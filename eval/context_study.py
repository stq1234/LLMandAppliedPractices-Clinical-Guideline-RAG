#!/usr/bin/env python3
"""Three supplied-context interventions through the same adaptive system.

This script is a one-worker diagnostic study over eight preselected, answerable
test questions. It never sends labels or expected answers to System.respond.
The three conditions differ only in the initial verified evidence IDs supplied
to the normal adaptive API: none, all joint gold, or a count-matched set of
same-theme records from other sources. The default adaptive results from the
main experiment are compared post hoc by question ID and are not rerun here.
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


# Selection was fixed before inspecting any model outputs. Each neighbor-only
# set matches the number of gold records and comes from non-gold sources.
SELECTED = {
    "T01": {
        "distractors": ["NICE-CLASS-01", "USP-BOUNDARY-01"],
        "distractor_reason": "NICE第1期联合诊室/诊室外分级和USPSTF机构阈值范围都涉及130–140段，但不是AHA/ACC与ISH类别比较的共同金标。",
    },
    "T03": {
        "distractors": ["AHA-OUT-01", "ISH-DX-02"],
        "distractor_reason": "AHA/ACC与ISH也谈诊室外确认，但不是NICE规定的同次复测与ABPM区间金标。",
    },
    "T05": {
        "distractors": ["ISH-OUT-01"],
        "distractor_reason": "ISH的诊室外确认建议主题相近，但不是AHA/ACC表7的140/90测量映射值。",
    },
    "T06": {
        "distractors": ["AHA-CLASS-04"],
        "distractor_reason": "AHA/ACC也以140/90作为第2期阈值，但标签与ISH的Grade 2问题不能跨机构替代。",
    },
    "T10": {
        "distractors": ["NICE-MEAS-04", "USP-PREP-01"],
        "distractor_reason": "NICE与USPSTF都有安静坐位、上臂测量等相邻规则，但不是ISH的休息时长与手臂高度依据。",
    },
    "T12": {
        "distractors": ["NICE-POP-01", "AHA-OUT-01"],
        "distractor_reason": "NICE成人范围和AHA/ACC诊室外确认与筛查主题相近，但不能替代USPSTF对象及其确认前置条件。",
    },
    "T15": {
        "distractors": ["JAHA-AOBP-02", "AHA-OUT-01"],
        "distractor_reason": "JAHA有人/无人测量差值及AHA/ACC确诊建议涉及血压测量，但不提供USPSTF综述的初筛与诊室确诊准确性。",
    },
    "T19": {
        "distractors": ["ISH-DX-04"],
        "distractor_reason": "ISH的单次就诊诊断例外也涉及严重血压和靶器官损害，但不替代NICE的当日转诊红旗条件。",
    },
}

CONDITIONS = ("empty_supplied", "gold_supplied", "neighbor_only")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                rows.append(value)
    return rows


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def selected_rows(project: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = {str(q["id"]): q for q in read_jsonl(project / "data" / "questions.jsonl")}
    evidence = {str(e["id"]): e for e in read_jsonl(project / "data" / "evidence.jsonl")}
    if set(SELECTED) - set(questions):
        raise ValueError("A preselected test ID is absent from the frozen question file")
    if set(SELECTED) - {qid for qid, q in questions.items() if q.get("split") == "test" and not q.get("unanswerable") and not q.get("scope_out")}:
        raise ValueError("The study may include only answerable frozen test questions")
    rows = [questions[qid] for qid in SELECTED]
    for q in rows:
        gold = set(q.get("gold", []))
        distractors = set(SELECTED[str(q["id"])]["distractors"])
        if not gold or gold & distractors or len(gold) != len(distractors):
            raise ValueError(f"Invalid gold/distractor design for {q['id']}")
        if any(eid not in evidence for eid in gold | distractors):
            raise ValueError(f"Unknown evidence ID in design for {q['id']}")
        if any(evidence[eid]["source_id"] in {evidence[g]["source_id"] for g in gold} for eid in distractors):
            raise ValueError(f"Distractor source overlaps the gold sources for {q['id']}")
    return rows, evidence


def check_existing_index(project: Path, backend: Any, evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fail before System initialization if cache identity/content is not exact."""
    import numpy as np

    cache_path = project / "index" / "clinical-vectors.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Survey-built index is missing: {cache_path}")
    records = []
    for item in evidence.values():
        embed_text = "\n".join(
            part for part in (
                item.get("title", ""),
                " ".join(item.get("topics", [])),
                item.get("population", ""),
                item.get("text", ""),
            ) if part
        )
        records.append((str(item["id"]), hashlib.sha256(embed_text.encode("utf-8")).hexdigest()))
    records.sort()
    with np.load(cache_path, allow_pickle=False) as cache:
        signature = str(cache["signature"][0])
        ids = cache["ids"].astype(str).tolist()
        hashes = cache["hashes"].astype(str).tolist()
        vectors_shape = list(cache["vectors"].shape)
    if signature != backend.embedding_signature:
        raise RuntimeError("Embedding signature differs from survey-built index; refusing to rebuild it")
    if ids != [row[0] for row in records] or hashes != [row[1] for row in records]:
        raise RuntimeError("Index records differ from frozen corpus; refusing to rebuild it")
    if vectors_shape[0] != len(records):
        raise RuntimeError("Index row count differs from frozen corpus")
    return {"cache_path": str(cache_path), "signature": signature, "rows": vectors_shape[0], "dimensions": vectors_shape[1]}


def response_record(q: dict[str, Any], condition: str, context_ids: list[str], response: dict[str, Any], usage: dict[str, Any], elapsed: float, question: dict[str, Any]) -> dict[str, Any]:
    from grading import grade_case

    record = {
        "id": str(q["id"]),
        "group": q.get("group"),
        "family": q["family"],
        "split": q["split"],
        "complexity": q["complexity"],
        "condition": condition,
        "query": q["query"],
        "context_ids": context_ids,
        "gold_ids": list(q["gold"]),
        "distractor_ids": SELECTED[str(q["id"])]["distractors"] if condition == "neighbor_only" else [],
        "distractor_reason": SELECTED[str(q["id"])]["distractor_reason"] if condition == "neighbor_only" else None,
        "answer": response.get("answer"),
        "citations": response.get("citations", []),
        "sources": response.get("sources", []),
        "abstain": response.get("abstain"),
        "initial_action": response.get("initial_action"),
        "rounds": response.get("rounds"),
        "first_evidence": response.get("first_evidence", []),
        "evidence": response.get("evidence", []),
        "trace": response.get("trace", []),
        "claims": response.get("claims", []),
        "diagnostics": response.get("diagnostics", {}),
        "usage": usage,
        "latency_seconds": round(elapsed, 3),
    }
    record["grading"] = grade_case(question, record)
    return record


def summarize(records: list[dict[str, Any]], output_path: Path, design: dict[str, Any]) -> dict[str, Any]:
    conditions = sorted({str(row["condition"]) for row in records})
    successes: dict[str, set[str]] = {}
    summary: dict[str, Any] = {"design": design, "conditions": {}}
    for condition in conditions:
        rows = [r for r in records if r["condition"] == condition]
        success_ids = {
            str(r["id"]) for r in rows
            if r.get("grading", {}).get("operational_success") is True
        }
        successes[condition] = success_ids
        summary["conditions"][condition] = {
            "n": len(rows),
            "keypoint_pass": sum(bool(r.get("grading", {}).get("keypoint_match")) for r in rows),
            "citation_complete": sum(bool(r.get("grading", {}).get("citations_complete")) for r in rows),
            "strict_joint_pass": len(success_ids),
            "abstentions": sum(bool(r.get("abstain")) for r in rows),
            "strict_joint_pass_ids": sorted(success_ids),
            "gold_citation_coverage_mean": (
                sum(float(r["grading"]["citation_coverage"]) for r in rows) / len(rows)
                if rows and all(r["grading"].get("citation_coverage") is not None for r in rows)
                else None
            ),
        }
    intersections = {}
    for i, left in enumerate(conditions):
        for right in conditions[i + 1:]:
            intersections[f"{left}__AND__{right}"] = sorted(successes[left] & successes[right])
    summary["strict_pass_member_intersections"] = intersections
    summary["n_questions"] = len({str(r["id"]) for r in records})
    summary["n_condition_rows"] = len(records)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--tag", default="formal", choices=("formal",), help="Keep the formal rerun separate from retained pilot files.")
    args = parser.parse_args()
    project = args.project.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "eval"))

    from clinical import VERSION, System
    from model_backend import Backend

    questions, evidence = selected_rows(project)

    # Normal inference backend and System entry point; sequential calls keep this
    # add-on at one generation request stream. The exact cache is checked before
    # constructing System, then its status is checked for zero document re-embeds.
    backend = Backend()
    index_info = check_existing_index(project, backend, evidence)
    system = System(backend, corpus_path="data/evidence.jsonl", index_dir="index")
    if int(system.index_status.get("embedded", -1)) != 0 or int(system.index_status.get("reused", -1)) != len(evidence):
        raise RuntimeError(f"The intervention tried to change the survey-built index: {system.index_status}")

    design = {
        "title": "Frozen test subset: supplied-context intervention",
        "model_version": VERSION,
        "evidence_sha256": digest(project / "data" / "evidence.jsonl"),
        "question_sha256": digest(project / "data" / "questions.jsonl"),
        "index": index_info,
        "selection_rule": "Eight answerable test questions selected before reading their system outputs; spread across classification, source comparison, confirmation, measurement, screening, review evidence, and exceptions; includes single- and multi-evidence cases.",
        "conditions": {
            "empty_supplied": "No evidence IDs are preloaded; the same default adaptive router may retrieve its normal evidence.",
            "gold_supplied": "All joint gold IDs are preloaded; the same default adaptive router may reuse them or follow its ordinary evidence-gap behavior.",
            "neighbor_only": "A count-matched set of same-theme records from different non-gold sources is preloaded, with no gold IDs; the same default adaptive router may reuse, supplement or abstain.",
        },
        "one_worker": True,
        "selected_questions": [],
    }
    for q in questions:
        selection = SELECTED[str(q["id"])]
        design["selected_questions"].append({
            "id": q["id"], "group": q.get("group"), "family": q["family"],
            "complexity": q["complexity"], "query": q["query"],
            "gold_ids": q["gold"], "original_supplied": q.get("supplied", []),
            "gold_count": len(q["gold"]),
            "distractor_ids": selection["distractors"],
            "distractor_reason": selection["distractor_reason"],
        })

    results_dir = project / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"context_study_{args.tag}"
    design_path = results_dir / f"{prefix}_design.json"
    raw_path = results_dir / f"{prefix}.jsonl"
    summary_path = results_dir / f"{prefix}_summary.json"
    if raw_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen study results: {raw_path}")
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_rows: list[dict[str, Any]] = []
    with raw_path.open("x", encoding="utf-8") as out:
        for q in questions:
            qid = str(q["id"])
            gold_ids = list(q["gold"])
            distractor_ids = list(SELECTED[qid]["distractors"])
            variants = [
                ("empty_supplied", [], "adaptive"),
                ("gold_supplied", gold_ids, "adaptive"),
                ("neighbor_only", distractor_ids, "adaptive"),
            ]
            for condition, context_ids, mode in variants:
                backend.reset_usage()
                started = time.perf_counter()
                response = as_jsonable(system.respond(q["query"], supplied=context_ids, mode=mode))
                record = response_record(q, condition, context_ids, response, as_jsonable(backend.usage()), time.perf_counter() - started, q)
                record["mode"] = mode
                record["adaptive_baseline_source"] = "context_study_fixed_supply"
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                output_rows.append(record)
                print(f"{condition} {qid} abstain={record['abstain']} rounds={record['rounds']}", flush=True)

    summary = summarize(output_rows, summary_path, design)
    summary["runtime"] = {"chat_model": os.getenv("CHAT_MODEL", "qwen3.6-35b-a3b"), "workers": 1}
    summary["runtime"]["elapsed_note"] = "Sequential requests; all three conditions use the same default adaptive system. Main-experiment adaptive rows are a separate post-hoc comparator."
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
