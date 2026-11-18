"""Low-cost, no-generation candidate and facet retrieval sweep.

The sweep reuses the frozen corpus/index and the already-recorded multi-route
subqueries. Gold IDs are read only after ranking for offline scoring. The
named-source condition is query-visible (authority names in the question) and
does not use gold labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical import System, _tokens
from model_backend import Backend


DEPTHS = (3, 5, 8)
SOURCE_PATTERNS = (
    (r"NICE|NG136", {"NICE_NG136"}),
    (r"AHA|ACC", {"AHA_ACC_2025"}),
    (r"ISH", {"ISH_GLOBAL_2026"}),
    (r"USPSTF", {"USPSTF_SCREEN_2021", "USPSTF_REVIEW_2021"}),
    (r"JAHA|DERAGON", {"JAHA_AOBP_REVIEW_2025"}),
)


def named_source_ids(query: str) -> list[str]:
    found = set()
    for pattern, source_ids in SOURCE_PATTERNS:
        if re.search(pattern, query, flags=re.I):
            found.update(source_ids)
    return sorted(found)


def rank_candidates(
    system: System,
    query: str,
    vector: np.ndarray,
    source_filter: set[str] | None,
) -> tuple[list[str], int]:
    eligible = [
        i for i, record in enumerate(system.records)
        if system._population_compatible(
            query, record["population"], record.get("population_scope")
        ) and (source_filter is None or record["source_id"] in source_filter)
    ]
    if not eligible:
        return [], 0
    dense = np.asarray(system.vectors[eligible] @ vector, dtype=np.float32)
    lexical = system._lexical_bm25(_tokens(query), eligible)
    scored = []
    for position, idx in enumerate(eligible):
        record = system.records[idx]
        mix = (
            float(dense[position])
            + 0.15 * float(lexical.get(idx, 0.0))
            + (0.025 if record["kind"] == "guideline" else 0.0)
        )
        scored.append((mix, record["id"], float(dense[position]), float(lexical.get(idx, 0.0))))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in scored], len(eligible)


def facet_queries_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    for step in trace:
        if step.get("stage") == "retrieval" and step.get("action") == "supplement":
            queries = step.get("queries") or []
            return list(dict.fromkeys(str(q).strip() for q in queries if str(q).strip()))
    return []


def summarize_rankings(case_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    output = {}
    for depth in DEPTHS:
        recalls = []
        complete = 0
        for row in case_rows:
            gold = set(row["gold"])
            retrieved = set(row["rankings"][method][:depth])
            hit_count = len(gold & retrieved)
            recalls.append(hit_count / len(gold) if gold else 1.0)
            complete += int(bool(gold) and gold.issubset(retrieved))
        output[str(depth)] = {
            "n": len(case_rows),
            "complete_gold_n": complete,
            "complete_gold_rate": complete / len(case_rows) if case_rows else None,
            "macro_gold_recall": float(np.mean(recalls)) if recalls else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "extended_retrieval_sweep.json")
    parser.add_argument("--case-csv", type=Path, default=ROOT / "results" / "extended_retrieval_cases.csv")
    parser.add_argument("--facet-csv", type=Path, default=ROOT / "results" / "extended_facet_coverage.csv")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in (ROOT / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row for row in questions
        if row.get("split") == args.split
        and not row.get("scope_out", False)
        and not row.get("unanswerable", False)
        and row.get("gold")
    ]
    if args.split == "test" and len(selected) != 22:
        raise ValueError(f"Expected 22 fixed-scope answerable test questions, found {len(selected)}")

    # Final system logs contain the actual facet-level follow-up queries from
    # the frozen multi route. No model generation is made by this sweep.
    final_log = ROOT / "results" / "final_evaluation.jsonl"
    multi_rows = {}
    if final_log.exists():
        for line in final_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            if result.get("split") == args.split and result.get("mode") == "multi" and result.get("status") == "ok":
                multi_rows[str(result["id"])] = result

    backend = Backend()
    system = System(backend, ROOT / "data" / "evidence.jsonl", ROOT / "index")
    query_texts = []
    facets_by_id = {}
    named_by_id = {}
    for question in selected:
        qid = str(question["id"])
        facets = facet_queries_from_trace((multi_rows.get(qid) or {}).get("trace") or [])
        facets_by_id[qid] = facets
        named_by_id[qid] = named_source_ids(question["query"])
        query_texts.append(str(question["query"]))
        query_texts.extend(facets)
    unique_queries = list(dict.fromkeys(q for q in query_texts if q))
    vectors = backend.encode(unique_queries, query=True)
    vector_by_query = {query: vectors[i] for i, query in enumerate(unique_queries)}

    methods = ("lexical", "dense", "hybrid", "named_source_filter")
    case_rows = []
    facet_rows = []
    for question in selected:
        qid = str(question["id"])
        query = str(question["query"])
        gold = [str(g) for g in question.get("gold", [])]
        source_ids = named_by_id[qid]
        source_filter = set(source_ids) if source_ids else None
        all_rankings = {}
        candidate_counts = {}
        for method in methods:
            chosen_filter = source_filter if method == "named_source_filter" else None
            order, count = rank_candidates(system, query, vector_by_query[query], chosen_filter)
            candidate_counts[method] = count
            # rank_candidates returns the same hybrid score; preserve separate
            # lexical/dense baselines to attribute the available route.
            if method == "lexical":
                eligible = [i for i, rec in enumerate(system.records)
                            if system._population_compatible(query, rec["population"], rec.get("population_scope"))
                            and (chosen_filter is None or rec["source_id"] in chosen_filter)]
                lex = system._lexical_bm25(_tokens(query), eligible)
                order = [system.records[i]["id"] for i in sorted(eligible, key=lambda idx: (-lex.get(idx, 0.0), system.records[idx]["id"]))]
            elif method == "dense":
                eligible = [i for i, rec in enumerate(system.records)
                            if system._population_compatible(query, rec["population"], rec.get("population_scope"))
                            and (chosen_filter is None or rec["source_id"] in chosen_filter)]
                dense = np.asarray(system.vectors[eligible] @ vector_by_query[query], dtype=np.float32)
                order = [system.records[i]["id"] for _, i in sorted(((-float(score), idx) for score, idx in zip(dense, eligible)), key=lambda item: (item[0], system.records[item[1]]["id"]))]
            all_rankings[method] = order
        ranks = {
            method: {g: order.index(g) + 1 if g in order else None for g in gold}
            for method, order in all_rankings.items()
        }
        facet_details = []
        for facet_index, facet in enumerate(facets_by_id[qid], 1):
            facet_order, facet_count = rank_candidates(system, facet, vector_by_query[facet], None)
            filtered_order, filtered_count = rank_candidates(
                system, facet, vector_by_query[facet], source_filter
            )
            facet_details.append({
                "index": facet_index,
                "query": facet,
                "candidate_count": facet_count,
                "named_source_candidate_count": filtered_count,
                "gold_ranks_unfiltered": {g: facet_order.index(g) + 1 if g in facet_order else None for g in gold},
                "gold_ranks_named_source": {g: filtered_order.index(g) + 1 if g in filtered_order else None for g in gold},
                "top8_unfiltered": facet_order[:8],
                "top8_named_source": filtered_order[:8],
            })
        facet_details.sort(key=lambda f: f["index"])
        facet_summary = {}
        for depth in DEPTHS:
            union_unfiltered = set()
            union_named = set()
            cumulative_unfiltered = []
            cumulative_named = []
            for facet in facet_details:
                union_unfiltered.update(facet["top8_unfiltered"][:depth])
                union_named.update(facet["top8_named_source"][:depth])
                cumulative_unfiltered.append({
                    "facet": facet["index"],
                    "new_gold_ids": sorted(set(gold) & union_unfiltered),
                    "gold_recall": len(set(gold) & union_unfiltered) / len(gold),
                })
                cumulative_named.append({
                    "facet": facet["index"],
                    "new_gold_ids": sorted(set(gold) & union_named),
                    "gold_recall": len(set(gold) & union_named) / len(gold),
                })
            facet_summary[str(depth)] = {
                "unfiltered_gold_ids": sorted(set(gold) & union_unfiltered),
                "unfiltered_gold_recall": len(set(gold) & union_unfiltered) / len(gold),
                "unfiltered_complete": set(gold).issubset(union_unfiltered),
                "named_source_gold_ids": sorted(set(gold) & union_named),
                "named_source_gold_recall": len(set(gold) & union_named) / len(gold),
                "named_source_complete": set(gold).issubset(union_named),
                "cumulative_unfiltered": cumulative_unfiltered,
                "cumulative_named_source": cumulative_named,
            }
            facet_rows.append({
                "id": qid,
                "family": question.get("family"),
                "complexity": question.get("complexity"),
                "top_k": depth,
                "facet_count": len(facet_details),
                "unfiltered_gold_recall": facet_summary[str(depth)]["unfiltered_gold_recall"],
                "unfiltered_complete": facet_summary[str(depth)]["unfiltered_complete"],
                "named_source_gold_recall": facet_summary[str(depth)]["named_source_gold_recall"],
                "named_source_complete": facet_summary[str(depth)]["named_source_complete"],
            })
        case_rows.append({
            "id": qid,
            "family": question.get("family"),
            "complexity": question.get("complexity"),
            "query": query,
            "gold": gold,
            "named_source_ids_from_query": source_ids,
            "source_filter_applied": bool(source_ids),
            "candidate_counts": candidate_counts,
            "gold_ranks": ranks,
            "rankings": all_rankings,
            "facet_queries_from_multi_trace": facets_by_id[qid],
            "facet_by_facet": facet_details,
            "facet_union_summary": facet_summary,
        })

    summary = {
        "study": "extended candidate/source-filter/facet sweep; no chat generation",
        "split": args.split,
        "question_count": len(selected),
        "question_sha256": __import__("hashlib").sha256((ROOT / "data" / "questions.jsonl").read_bytes()).hexdigest(),
        "evidence_sha256": __import__("hashlib").sha256((ROOT / "data" / "evidence.jsonl").read_bytes()).hexdigest(),
        "backend_chat_calls": backend.usage().get("chat_calls", 0),
        "query_vectors_encoded": backend.usage().get("query_vectors", 0),
        "depths": list(DEPTHS),
        "methods": list(methods),
        "source_filter_rule": "only query-mentioned guideline authority IDs; otherwise no restriction beyond adult/population compatibility",
        "facet_rule": "actual follow-up retrieval query strings from the frozen forced-multi trace; per-facet retrieval is diagnostic and does not expose gold to System",
        "rank_summary": {
            method: summarize_rankings(case_rows, method)
            for method in methods
        },
        "cases": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.case_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.case_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "family", "complexity", "source_filter_applied", "named_source_ids_from_query", "candidate_counts", "gold_ranks"])
        writer.writeheader()
        for row in case_rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list)) else row[key] for key in writer.fieldnames})
    args.facet_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.facet_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(facet_rows[0]) if facet_rows else ["id", "top_k"])
        writer.writeheader()
        writer.writerows(facet_rows)
    print(json.dumps(summary["rank_summary"], ensure_ascii=False, indent=2))
    print(f"chat_calls={summary['backend_chat_calls']} query_vectors={summary['query_vectors_encoded']}")


if __name__ == "__main__":
    main()

