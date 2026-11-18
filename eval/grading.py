"""Task-specific grading helpers for the clinical RAG evaluation.

The system never receives answer keys. This module is used only after a response
has been recorded, so gold IDs and expected strings remain outside inference.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


_CRITICAL_REPLACEMENTS = (
    (r"(?i)\bno\s+less\s+than\b|\bat\s+least\b", ">="),
    (r"(?i)\bnot\s+less\s+than\b", ">="),
    (r"(?i)\bgreater\s+than\b|\bmore\s+than\b", ">"),
    (r"(?i)\bno\s+more\s+than\b|\bat\s+most\b", "<="),
    (r"(?i)\bless\s+than\b", "<"),
    (r"不低于|不得低于|至少", ">="),
    (r"高于|超过|大于", ">"),
    (r"不超过|至多|最多", "<="),
    (r"低于|少于|小于", "<"),
)
_NUM = r"[+-]?\d+(?:\.\d+)?"
_NUM_EXPR = re.compile(
    rf"(?P<op>>=|<=|>|<|=)?(?P<a>{_NUM})"
    rf"(?:-(?P<b>{_NUM}))?"
    rf"(?:/(?P<c>{_NUM})(?:-(?P<d>{_NUM}))?)?"
    r"(?P<unit>mmol/l|mg/dl|kg/m2|kg/m²|mmhg|mmhg|mm|cm|kg|mg|mcg|μg|µg|g|bpm|"
    r"years?|yrs?|yr|days?|weeks?|months?|hours?|minutes?|seconds?|perday|peryear|h|min|s|%)?",
    re.IGNORECASE,
)
_UNITS = re.compile(r"^(?:mmol/l|mg/dl|kg/m2|kg/m²|mmhg|mm|cm|kg|mg|mcg|μg|µg|g|bpm|years?|yrs?|yr|days?|weeks?|months?|hours?|minutes?|seconds?|perday|peryear|h|min|s|%)$", re.I)


def _norm(text: Any) -> str:
    """Compact text while preserving clinical numeric signs/operators/ranges."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = value.translate(str.maketrans({"−": "-", "﹣": "-", "–": "-", "—": "-", "＋": "+", "≧": "≥", "≦": "≤"}))
    value = value.replace("≥", ">=").replace("≤", "<=")
    for pattern, replacement in _CRITICAL_REPLACEMENTS:
        value = re.sub(pattern, replacement, value)
    # Range connectors are normalized only when they occur between numerals.
    value = re.sub(r"(?<=\d)\s*(?:至|到|~|～)\s*(?=[+-]?\d)", "-", value)
    return "".join(
        ch for ch in value
        if ch.isalnum() or "\u3400" <= ch <= "\u9fff" or ch in "+-<>=./%"
    )


def _decimal(value: str) -> str:
    # Trailing zeroes and an explicit positive sign do not change the value.
    value = value.lstrip("+")
    negative = value.startswith("-")
    raw = value[1:] if negative else value
    whole, dot, frac = raw.partition(".")
    whole = whole.lstrip("0") or "0"
    frac = frac.rstrip("0") if dot else ""
    canonical = whole + ("." + frac if frac else "")
    if canonical == "0":
        negative = False
    return ("-" if negative else "") + canonical


def _numeric_atoms(value: str) -> list[tuple[str, ...]]:
    atoms = []
    for match in _NUM_EXPR.finditer(value):
        parts = match.groupdict()
        numbers = tuple(_decimal(parts[k]) for k in ("a", "b", "c", "d") if parts[k])
        atoms.append((parts["op"] or "", *numbers, (parts["unit"] or "").casefold()))
    return atoms


def _without_numeric_atoms(value: str) -> str:
    return re.sub(_NUM_EXPR, "", value)


def _strip_provenance(answer: str, response: Mapping[str, Any]) -> str:
    """Remove citation IDs and displayed source labels before keypoint grading."""
    text = answer
    ids = _evidence_ids(response.get("citations"))
    sources = response.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
        for source in sources:
            if isinstance(source, Mapping):
                ids.extend(_evidence_ids(source))
                for key in ("title", "name", "display", "label", "source"):
                    value = source.get(key)
                    if isinstance(value, str) and len(value.strip()) >= 4:
                        text = re.sub(re.escape(value.strip()), " ", text, flags=re.I)
    for evidence_id in sorted(set(ids), key=len, reverse=True):
        escaped = re.escape(evidence_id)
        # Remove citation-like wrappers or an exact ID token, without deleting
        # the same digits embedded in a clinical value such as 140 mmHg.
        text = re.sub(rf"\[\s*{escaped}\s*\]|【\s*{escaped}\s*】", " ", text, flags=re.I)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:/-]*", evidence_id):
            text = re.sub(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", " ", text, flags=re.I)
    text = re.sub(r"(?im)^\s*(?:source|sources|来源|出处|引用依据)\s*[:：].*$", " ", text)
    return text


def _candidate_match(candidate: Any, answer: str) -> bool:
    needle = _norm(candidate)
    haystack = _norm(answer)
    if not needle:
        return False
    expected_atoms = _numeric_atoms(needle)
    answer_atoms = _numeric_atoms(haystack)
    if not expected_atoms and needle in haystack:
        return True
    if expected_atoms:
        remaining = list(answer_atoms)
        for expected in expected_atoms:
            op, *tail = expected
            expected_unit = tail[-1]
            expected_numbers = tail[:-1]
            found = None
            for i, observed in enumerate(remaining):
                obs_op, *obs_tail = observed
                observed_unit = obs_tail[-1]
                observed_numbers = obs_tail[:-1]
                if op != obs_op or expected_numbers != observed_numbers:
                    continue
                if expected_unit and expected_unit != observed_unit:
                    continue
                found = i
                break
            if found is None:
                return False
            remaining.pop(found)

    text_needle = _without_numeric_atoms(needle)
    text_haystack = _without_numeric_atoms(haystack)
    if text_needle:
        return text_needle in text_haystack
    return bool(expected_atoms)


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(v) for v in value.values() if isinstance(v, (str, int, float))]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping):
                out.extend(
                    str(item[k])
                    for k in ("id", "evidence_id", "source_id", "document_id")
                    if item.get(k) is not None
                )
        return out
    return []


def _evidence_ids(value: Any) -> list[str]:
    """Extract evidence IDs from the list/dict forms used by System outputs."""
    if isinstance(value, Mapping):
        for key in ("ids", "evidence_ids", "citations", "items"):
            if key in value:
                return _evidence_ids(value[key])
        for key in ("id", "evidence_id", "source_id", "document_id"):
            if value.get(key) is not None:
                return [str(value[key])]
        return []
    return [s.strip() for s in _strings(value) if s.strip()]


def _expected_match(question: Mapping[str, Any], response: Mapping[str, Any]) -> bool:
    answer = _strip_provenance(str(response.get("answer") or ""), response)
    groups = question.get("expected_groups")
    if groups:
        # Each group is one required point; entries within a group are synonyms.
        if isinstance(groups, Sequence) and not isinstance(groups, str):
            for group in groups:
                choices = group if isinstance(group, Sequence) and not isinstance(group, str) else [group]
                if not choices or not any(_candidate_match(choice, answer) for choice in choices):
                    return False
            return True
    expected = question.get("expected")
    if isinstance(expected, Sequence) and not isinstance(expected, str):
        expected = " ".join(str(x) for x in expected)
    return _candidate_match(expected, answer)


def grade_case(question: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    """Return answer, citation, and first/supplement evidence diagnostics.

    Answerable cases are scored by literal critical-keypoint matching and gold
    citation coverage, not semantic clinical adjudication. Unanswerable and
    scope-out cases are evaluated via abstention and reported separately.
    """
    answer = str(response.get("answer") or "")
    abstain = bool(response.get("abstain", False))
    scope_out = bool(question.get("scope_out", False))
    unanswerable = bool(question.get("unanswerable", False))
    initial_need_raw = question.get("initial_need")
    initial_need = bool(initial_need_raw) if initial_need_raw is not None else None
    answerable = not (scope_out or unanswerable)
    diagnostics = response.get("diagnostics") or {}
    model_scope_reason = diagnostics.get("scope_reason")
    model_out_action = str(response.get("initial_action") or "") == "out"
    model_scope_out = model_scope_reason == "out_of_scope"
    model_personalized_refusal = model_scope_reason == "personalized"

    gold_ids = sorted(set(_evidence_ids(question.get("gold"))))
    citation_ids = sorted(set(_evidence_ids(response.get("citations"))))
    first_ids = sorted(set(_evidence_ids(response.get("first_evidence"))))
    final_ids = sorted(set(_evidence_ids(response.get("evidence"))))
    citation_set, gold_set = set(citation_ids), set(gold_ids)
    first_set, final_set = set(first_ids), set(final_ids)
    citation_coverage = len(citation_set & gold_set) / len(gold_set) if gold_set else None
    first_coverage = len(first_set & gold_set) / len(gold_set) if gold_set else None
    final_coverage = len(final_set & gold_set) / len(gold_set) if gold_set else None
    expected_match = _expected_match(question, response) if answerable else None
    citations_complete = bool(gold_set) and gold_set.issubset(citation_set)
    evidence_complete = bool(gold_set) and gold_set.issubset(final_set)
    new_ids = sorted(final_set - first_set)
    new_gold = sorted((final_set - first_set) & gold_set)
    first_missing = sorted(gold_set - first_set)

    if answerable:
        keypoint_match = bool(expected_match and not abstain)
        supported_keypoint_match = bool(keypoint_match and citations_complete)
        answer_error = not keypoint_match
        unsupported_answer = bool(not abstain and not citations_complete)
        wrong_refusal = abstain
        correct_refusal = None
    else:
        keypoint_match = None
        supported_keypoint_match = None
        answer_error = not abstain
        unsupported_answer = False
        wrong_refusal = False
        correct_refusal = abstain

    return {
        "answerable": answerable,
        "unanswerable": unanswerable,
        "scope_out": scope_out,
        "initial_need": initial_need,
        "model_scope_reason": model_scope_reason,
        "model_out_action": model_out_action,
        "model_scope_out": model_scope_out,
        "model_personalized_refusal": model_personalized_refusal,
        "false_scope_out_prediction": bool(model_scope_out and not scope_out),
        "abstain": abstain,
        "expected_match": expected_match,
        "keypoint_match": keypoint_match,
        "supported_keypoint_match": supported_keypoint_match,
        "correct_refusal": correct_refusal,
        "operational_success": (
            supported_keypoint_match if answerable else bool(correct_refusal)
        ),
        "answer_error": answer_error,
        "unsupported_answer": unsupported_answer,
        "wrong_refusal": wrong_refusal,
        "gold_ids": gold_ids,
        "citation_ids": citation_ids,
        "first_evidence_ids": first_ids,
        "final_evidence_ids": final_ids,
        "citation_coverage": citation_coverage,
        "citations_complete": citations_complete,
        "first_gold_coverage": first_coverage,
        "final_gold_coverage": final_coverage,
        "final_evidence_complete": evidence_complete,
        "first_missing_gold_ids": first_missing,
        "new_evidence_ids": new_ids,
        "new_gold_ids": new_gold,
        "supplement_added_gold": bool(new_gold),
        "rescued_supported_keypoint_match": bool(
            answerable and first_missing and expected_match and not abstain and citations_complete
        ),
    }


def risk_cost(grade: Mapping[str, Any], response: Mapping[str, Any]) -> float:
    """Cost-sensitive dev objective; labels are literal keypoint proxies.

    A wrong non-refusal answer costs 100; an answer lacking all required gold
    citations adds 100; a refusal on an answerable case costs 30. This keeps a
    false refusal below a wrong answer while giving unsupported answers the
    largest penalty. Unanswerable/scope-out answers are wrong answers, not
    citation errors. Tokens and retrieval rounds are secondary costs; latency
    is reported separately and does not influence threshold selection. Runtime
    and API failures are handled separately.
    """
    answerable = bool(grade.get("answerable"))
    abstain = bool(grade.get("abstain"))
    wrong_answer = (answerable and not abstain and not bool(grade.get("expected_match"))) or (not answerable and not abstain)
    unsupported_answer = answerable and not abstain and not bool(grade.get("citations_complete"))
    false_refusal = answerable and abstain
    cost = 100.0 * float(wrong_answer)
    cost += 100.0 * float(unsupported_answer)
    cost += 30.0 * float(false_refusal)
    usage = response.get("usage") or {}
    tokens = float(usage.get("prompt_tokens", 0) or 0) + float(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )
    cost += 0.002 * tokens
    cost += 0.5 * float(response.get("rounds", 0) or 0)
    return cost

