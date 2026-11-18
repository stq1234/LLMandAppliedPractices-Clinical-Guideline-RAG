"""Controlled comparison of candidate generators for the clinical evidence corpus."""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical import System, _tokens
from model_backend import Backend


SOURCES = {
    r"NICE|NG136": {"NICE_NG136"},
    r"AHA|ACC": {"AHA_ACC_2025"},
    r"ISH": {"ISH_GLOBAL_2026"},
    r"USPSTF": {"USPSTF_SCREEN_2021","USPSTF_REVIEW_2021"},
    r"JAHA|DERAGON": {"JAHA_AOBP_REVIEW_2025"},
}
METHODS = ("lexical","dense","hybrid","named_sources")
DEPTHS = (1,3,5,8,10)


def rank_sources(question, system, query_vector):
    """Using query text only, score records under four first-pass conditions."""
    eligible = [i for i, doc in enumerate(system.records)
                if system._population_compatible(question, doc["population"], doc.get("population_scope"))]
    lexical = system._lexical_bm25(_tokens(question), eligible)
    dense = system.vectors @ query_vector
    named = set().union(*(sources for pattern, sources in SOURCES.items()
                          if re.search(pattern, question, flags=re.I)))
    scores = {m: [] for m in METHODS}
    for i in eligible:
        doc = system.records[i]
        item = doc["id"]
        mix = float(dense[i]) + 0.15 * lexical[i] + 0.025 * (doc["kind"] == "guideline")
        scores["dense"].append((float(dense[i]), item))
        scores["lexical"].append((lexical[i], item))
        scores["hybrid"].append((mix, item))
        if not named or doc["source_id"] in named:
            scores["named_sources"].append((mix, item))
    return {key: [i for _, i in sorted(pairs, key=lambda pair: (-pair[0],pair[1]))]
            for key,pairs in scores.items()}


def main():
    backend = Backend()
    system = System(backend, ROOT/"data/evidence.jsonl", ROOT/"index")
    questions = [json.loads(line) for line in (ROOT/"data/questions.jsonl").read_text().splitlines()]
    selected = [q for q in questions if q["split"]=="test" and not (q["scope_out"] or q["unanswerable"])]
    vectors = backend.encode([q["query"] for q in selected], query=True)
    cases = []
    for q, vector in zip(selected, vectors):
        ranked = rank_sources(q["query"], system, vector)
        gold = set(q["gold"])
        cases.append({"id":q["id"],"complexity":q["complexity"],"query":q["query"],
                      "gold":q["gold"],"ranks":{m:{g:order.index(g)+1 if g in order else None for g in sorted(gold)}
                      for m,order in ranked.items()}})
    groups = {"all":cases, "single":[x for x in cases if x["complexity"]=="single"],
              "multi":[x for x in cases if x["complexity"]=="multi"]}
    summary = {}
    for group,rows in groups.items():
        summary[group] = {}
        for method in METHODS:
            counts = {}
            for depth in DEPTHS:
                ranks = [[r or float("inf") for r in x["ranks"][method].values()] for x in rows]
                counts[depth] = {"complete":sum(max(r)<=depth for r in ranks),
                                 "macro_recall":float(np.mean([np.mean([v<=depth for v in r]) for r in ranks])),
                                 "n":len(rows)}
            summary[group][method] = counts
    dest=ROOT/"results"/"retrieval_study.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"summary":summary,"cases":cases},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v[5] for k,v in summary["all"].items()},ensure_ascii=False))


if __name__=="__main__":
    main()
