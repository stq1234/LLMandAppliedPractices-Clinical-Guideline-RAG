"""No-chat retrieval smoke for adult guideline population eligibility.

Reads only the frozen D01/D05 query text and gold IDs for a post-retrieval
assertion. The System receives only query strings; gold IDs are used afterward
to report their ranks in top-20 candidates.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical import System
from model_backend import Backend


def main() -> None:
    questions = [
        json.loads(line)
        for line in Path("data/questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {row["id"]: row for row in questions if row.get("id") in {"D01", "D05"}}
    if set(selected) != {"D01", "D05"}:
        raise RuntimeError("Frozen D01 and D05 queries are required for this smoke")
    system = System(Backend(), corpus_path="data/evidence.jsonl", index_dir="index")
    source_counts = Counter(row.get("authority", "") for row in system.records)
    scope = Counter(
        (str(row.get("authority")), tuple(row.get("population_scope", {}).get("included", [])))
        for row in system.records
    )
    observations = []
    for qid in ("D01", "D05"):
        q = selected[qid]
        hits = system._retrieve([q["query"]], 20, system._params(None))
        ranked = [row["id"] for row in hits]
        ranks = {gold: ranked.index(gold) + 1 if gold in ranked else None for gold in q.get("gold", [])}
        observations.append({
            "id": qid,
            "query": q["query"],
            "eligible_guideline_records": sum(
                row.get("kind") == "guideline"
                and system._population_compatible(q["query"], row["population"], row.get("population_scope"))
                for row in system.records
            ),
            "gold_candidate_ranks_top20": ranks,
            "top20_ids": ranked,
        })
    out = {
        "records": len(system.records),
        "guideline_records_by_authority": dict(source_counts),
        "population_scope_counts": {str(k): v for k, v in scope.items()},
        "queries": observations,
        "chat_calls": system.backend.usage().get("chat_calls", 0),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out["chat_calls"] != 0:
        raise RuntimeError("Population smoke unexpectedly called the chat model")


if __name__ == "__main__":
    main()

