#!/usr/bin/env python3
"""Post-hoc scope-gate sensitivity variant over the unchanged clinical engine.

Only System._scope_reason is replaced. Retrieval, routing after the gate,
evidence checking, writing, citation validation and semantic validation are
inherited unchanged from clinical.System. The query passed to System.respond is
always the original query string.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from clinical import (
    System,
    _CHILD_TERMS,
    _DIAGNOSIS_TERMS,
    _PERSONAL_TERMS,
    _PREGNANCY_TERMS,
    _TREATMENT_TERMS,
    _has_bp_evidence_context,
    _norm,
)


_DIRECT_BP_TERMS = (
    "blood pressure", "blood-pressure", "hypertension", "hypertensive",
    "systolic", "diastolic", "bp", "abpm", "hbpm", "aobp", "mmhg",
    "sphygmomanometer", "orthostatic hypotension", "white-coat hypertension",
    "masked hypertension", "血压", "高血压", "低血压", "收缩压", "舒张压",
    "动态血压", "家庭血压", "诊室血压", "体位性低血压", "毫米汞柱",
)

# Broad organizations require an actual BP measurement setting/device, not a
# generic mention of screening, diagnosis, or measurement in another specialty.
_SETTING_OR_TECHNOLOGY = (
    "clinic", "office", "home bp", "home blood pressure", "ambulatory",
    "abpm", "hbpm", "aobp", "cuff", "cuff size", "sphygmomanometer",
    "mmhg", "blood pressure monitor", "血压测量", "血压监测", "诊室血压",
    "家庭血压", "动态血压", "诊室", "门诊血压", "家庭血压计",
    "诊室外", "袖带", "血压计", "上臂血压", "血压读数",
)
_MEASUREMENT_TASK_CONCEPTS = _SETTING_OR_TECHNOLOGY + (
    "measurement", "measuring", "measure", "monitoring", "monitor",
    "screening", "筛查", "初筛", "测量", "测压", "监测", "读数", "复测",
)
_REVIEW_ANALYSIS_CUES = (
    "sensitivity", "specificity", "accuracy", "systematic review", "meta-analysis",
    "敏感度", "特异度", "准确性", "证据综述", "系统综述", "初筛",
)
_ISH_ORG = ("ISH", "International Society of Hypertension", "国际高血压学会")
_BROAD_ORGS = (
    "NICE", "USPSTF", "AHA", "ACC",
    "National Institute for Health and Care Excellence",
    "U.S. Preventive Services Task Force",
    "American Heart Association", "American College of Cardiology",
)
_DIAGNOSTIC_CONCEPTS = tuple(_DIAGNOSIS_TERMS) + (
    "classification", "classify", "category", "categories", "grade", "graded",
    "分级", "分类", "类别", "判定",
)


def _has_ascii_term(text: str, term: str) -> bool:
    """Match ASCII tokens across punctuation/CJK boundaries (e.g. mmHg归类)."""
    folded = _norm(text)
    needle = _norm(term)
    if not needle:
        return False
    if not term.isascii():
        return needle in folded
    if " " in needle:
        return needle in folded
    if needle in {"diagnos", "measure", "classif", "confirm", "screen"}:
        return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"[A-Za-z]*(?![A-Za-z0-9])", folded, re.I))
    return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])", folded, re.I))


def _has_any(text: str, terms: Sequence[str]) -> bool:
    return any(_has_ascii_term(text, term) for term in terms)


def decide_scope(
    query: str,
    context_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[Optional[str], str]:
    """Return (refusal reason or None, an auditable positive/negative basis).

    No question IDs, gold evidence, expected strings or evaluation family are
    used. Valid supplied records can contextualize a short BP follow-up.
    """
    query = str(query or "")
    # Preserve the medication/personalized and pediatric/pregnancy gates first.
    if _has_any(query, _PERSONAL_TERMS) or _has_any(query, _TREATMENT_TERMS):
        return "personalized", "personal_or_treatment_gate"
    if _has_any(query, _PREGNANCY_TERMS) or _has_any(query, _CHILD_TERMS):
        return "out_of_scope", "pregnancy_or_pediatric_gate"

    # A direct BP concept is sufficient; matching handles ASCII/CJK adjacency.
    if _has_any(query, _DIRECT_BP_TERMS):
        return None, "direct_blood_pressure_term"

    diagnostic_concept = _has_any(query, _DIAGNOSTIC_CONCEPTS)
    measurement_task = _has_any(query, _MEASUREMENT_TASK_CONCEPTS)
    setting_or_device = _has_any(query, _SETTING_OR_TECHNOLOGY)
    if _has_any(query, _ISH_ORG) and (diagnostic_concept or measurement_task):
        return None, "hypertension_specialist_plus_diagnostic_or_measurement_concept"
    if _has_any(query, _BROAD_ORGS) and setting_or_device:
        return None, "broad_organization_plus_measurement_setting_or_technology"
    if context_records and _has_bp_evidence_context(context_records):
        if diagnostic_concept or measurement_task or _has_any(query, _REVIEW_ANALYSIS_CUES):
            return None, "verified_bp_context_plus_task_concept"
    return "out_of_scope", "no_positive_blood_pressure_domain_cue"


class ScopeVariantSystem(System):
    """The normal System API with only its scope decision replaced."""

    @staticmethod
    def _scope_reason(
        query: str,
        context_records: Sequence[Mapping[str, Any]] = (),
    ) -> Optional[str]:
        reason, _basis = decide_scope(query, context_records)
        return reason

    def scope_decision(self, query: str, supplied: Optional[Sequence[str]] = None) -> dict[str, Any]:
        records = self._resolve_supplied(supplied)
        reason, basis = decide_scope(query, records)
        return {"reason": reason, "basis": basis, "supplied_ids": [r["id"] for r in records]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent BP scope-gate CLI; other inference stages are unchanged.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=("reuse", "single", "multi", "adaptive", "adaptive_nocrit"), default="adaptive")
    parser.add_argument("--supplied", default="", help="Comma-separated verified evidence IDs")
    parser.add_argument("--corpus", default="data/evidence.jsonl")
    parser.add_argument("--index", default="index")
    parser.add_argument("--params", type=Path)
    args = parser.parse_args()

    project = Path.cwd().resolve()
    sys.path.insert(0, str(project))
    from model_backend import Backend

    supplied = [part.strip() for part in args.supplied.split(",") if part.strip()]
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else None
    backend = Backend()
    system = ScopeVariantSystem(backend, corpus_path=args.corpus, index_dir=args.index)
    variant_gate = system.scope_decision(args.query, supplied)
    response = system.respond(args.query, supplied=supplied, mode=args.mode, params=params)
    response["scope_variant_decision"] = variant_gate
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
