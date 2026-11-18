"""Improved inference path for adult blood-pressure evidence QA.

This module subclasses the frozen baseline. It adds corpus-derived domain
recognition, faithful facet planning, guaranteed supplied-evidence inclusion,
facet-wise retrieval with source balancing, and an opt-in supported-partial
answer policy. It does not read evaluation questions or labels.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from clinical import (
    System as ClinicalSystem,
    _ADULT_TERMS,
    _CHILD_TERMS,
    _PREGNANCY_TERMS,
    _TREATMENT_TERMS,
    _PERSONAL_TERMS,
    _has_bp_evidence_context,
    _has_anchor,
    _norm,
    _parse_json,
    _tokens,
    _refusal,
)
from scope_variant import _BROAD_ORGS, _ISH_ORG, _has_any, decide_scope


VERSION = "0.2.0"

_EXPLICIT_SPLIT_RE = re.compile(
    r"(?:[?？;；。！!]|\b(?:and|also|whereas)\b|以及|并且|同时|另外|此外|分别)",
    re.IGNORECASE,
)
_DEICTIC_FOLLOWUP = (
    "this", "that", "these", "those", "it", "there", "上述", "前述", "这个",
    "这项", "这条", "该方法", "该条款", "它们", "它",
)


def _explicit_facets(query: str) -> List[str]:
    """Split only explicit clauses; never invent synthetic evidence needs."""
    raw = [part.strip(" \t\r\n,，、:：") for part in _EXPLICIT_SPLIT_RE.split(query)]
    parts = list(dict.fromkeys(part for part in raw if len(part) >= 2))
    if len(parts) <= 1:
        return [query.strip()] if query.strip() else []
    if len(parts) <= 3:
        return parts
    return [parts[0], parts[1], "；".join(parts[2:])]


class System(ClinicalSystem):
    """Evidence QA system with stricter query fidelity and retrieval coverage."""

    def __init__(self, backend: Any, corpus_path: str = "data/evidence.jsonl", index_dir: str = "index") -> None:
        super().__init__(backend, corpus_path=corpus_path, index_dir=index_dir)
        self._request_local = threading.local()
        self._authority_aliases: Dict[str, List[str]] = {}
        self._authority_terms: List[str] = []
        self._topic_terms: List[str] = []
        self._build_scope_vocabulary()

    def _build_scope_vocabulary(self) -> None:
        authority_aliases: Dict[str, set] = defaultdict(set)
        topic_authorities: Dict[str, set] = defaultdict(set)
        for record in self.records:
            authority = str(record.get("authority", "")).strip()
            if authority:
                aliases = [authority]
                aliases.extend(re.findall(r"\(([^()]+)\)", authority))
                for known in (*_BROAD_ORGS, *_ISH_ORG):
                    if _has_any(authority, (known,)):
                        aliases.append(known)
                for alias in aliases:
                    normalized = _norm(alias)
                    if normalized:
                        authority_aliases[normalized].add(authority)
            for topic in record.get("topics", []):
                normalized = _norm(topic)
                if normalized:
                    topic_authorities[normalized].add(authority or record["source_id"])

        # Short generic labels such as “筛查” are not sufficient to infer a
        # clinical domain. Direct BP terms are handled by scope_variant.py.
        for topic in topic_authorities:
            cjk_len = len(re.findall(r"[\u3400-\u9fff]", topic))
            latin_words = re.findall(r"[a-z]+", topic)
            numeric = bool(re.search(r"\d", topic))
            specific = cjk_len >= 3 or len(latin_words) >= 2 or numeric or len(topic) >= 8
            if specific:
                self._topic_terms.append(topic)

        self._authority_aliases = {
            alias: sorted(authorities)
            for alias, authorities in authority_aliases.items()
        }
        self._authority_terms = sorted(
            self._authority_aliases,
            key=lambda value: (-len(value), value),
        )
        self._topic_terms.sort(key=lambda value: (-len(value), value))

    def _state(self) -> Optional[Dict[str, Any]]:
        return getattr(self._request_local, "value", None)

    def _query_authorities(self, text: str) -> Tuple[List[str], List[str]]:
        """Return matched query aliases and their corpus authority values."""
        matched_aliases = [
            alias for alias in self._authority_terms
            if _has_any(text, (alias,))
        ]
        authority_values: List[str] = []
        for alias in matched_aliases:
            for authority in self._authority_aliases.get(alias, []):
                if authority not in authority_values:
                    authority_values.append(authority)
        return matched_aliases, authority_values

    def _query_population_anchors(self, query: str) -> List[str]:
        terms = (*_ADULT_TERMS, *_CHILD_TERMS, *_PREGNANCY_TERMS)
        return [term for term in terms if _has_any(query, (term,))]

    def _scope_anchors(self, query: str) -> List[str]:
        aliases, _authorities = self._query_authorities(query)
        anchors = list(aliases)
        for term in self._query_population_anchors(query):
            if term not in anchors:
                anchors.append(term)
        return anchors

    def _preserve_scope_anchors(self, facet: str, query: str) -> str:
        missing = [anchor for anchor in self._scope_anchors(query) if not _has_any(facet, (anchor,))]
        return (" ".join(missing) + " " + facet).strip() if missing else facet.strip()

    def _scope_reason(self, query: str, context_records: Sequence[Mapping[str, Any]] = ()) -> Optional[str]:
        reason, basis = decide_scope(query, context_records)
        if reason is None or reason == "personalized":
            state = self._state()
            if state is not None:
                state["scope_basis"] = basis
            return reason

        # Keep the adult-only service boundary ahead of corpus topic matches.
        if _has_any(query, (*_PREGNANCY_TERMS, *_CHILD_TERMS, *_PERSONAL_TERMS, *_TREATMENT_TERMS)):
            return reason

        query_topics = [topic for topic in self._topic_terms if _has_any(query, (topic,))]
        matched_aliases, _authorities = self._query_authorities(query)
        if query_topics and matched_aliases:
            state = self._state()
            if state is not None:
                state["scope_basis"] = "corpus_topic_plus_named_authority"
                state["scope_topics"] = query_topics
                state["scope_authorities"] = matched_aliases
            return None

        # A short deictic follow-up can inherit a verified BP context supplied
        # by the caller, while independent out-of-domain requests remain gated.
        if context_records and _has_bp_evidence_context(context_records) and _has_any(query, _DEICTIC_FOLLOWUP):
            state = self._state()
            if state is not None:
                state["scope_basis"] = "verified_bp_context_plus_deictic_followup"
            return None

        state = self._state()
        if state is not None:
            state["scope_basis"] = basis
        return reason

    def features(self, query: str, supplied: Optional[Iterable[str]] = None) -> Dict[str, float]:
        features = super().features(query, supplied)
        facets = _explicit_facets(str(query or ""))
        multi_score = min(1.0, max(0.0, (len(facets) - 1) / 2.0))
        features["facet_count"] = float(len(facets))
        features["multi_part"] = float(len(facets) > 1)
        features["multi_aspect_score"] = float(multi_score)
        return features

    def _params(self, params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        raw = dict(params or {})
        allow_partial = raw.pop("allow_partial", False)
        if not isinstance(allow_partial, bool):
            raise ValueError("allow_partial must be a boolean")
        evidence_budget = int(raw.pop("max_evidence", 8))
        if evidence_budget < 1 or evidence_budget > 256:
            raise ValueError("max_evidence must be between 1 and 256")
        checked = dict(raw)
        checked["max_evidence"] = min(evidence_budget, 12)
        result = ClinicalSystem._params(checked)
        result["max_evidence"] = evidence_budget
        result["allow_partial"] = allow_partial
        return result

    def _plan(self, query: str) -> Tuple[List[str], Optional[str]]:
        key = _norm(query)
        if key in self._plan_cache:
            return list(self._plan_cache[key]), None

        explicit = _explicit_facets(query)
        system = (
            "You are a faithful evidence-query planner. Do not answer the user. "
            "Split only the information needs the user explicitly asked for. "
            "Do not add treatment, monitoring, risk, management, or follow-up advice "
            "unless the user explicitly requested it. Preserve every named institution, "
            "population, measurement setting, number, and comparison. Every facet must "
            "be an exact contiguous span copied from the question; do not paraphrase. "
            "If the question has one information need, return the whole question as one "
            "facet. Return only JSON: {\"facets\":[\"exact question span\"]}."
        )
        fallback: Optional[str] = None
        try:
            parsed = _parse_json(self.backend.chat(
                system=system,
                user=json.dumps({"question": query}, ensure_ascii=False),
            ))
            raw = parsed.get("facets")
            proposed = [
                str(item).strip() for item in raw
                if isinstance(item, str) and item.strip()
            ] if isinstance(raw, list) else []
            exact = [
                facet for facet in proposed
                if _norm(facet) in _norm(query)
            ]
            if not exact:
                facets = explicit
                fallback = "planner_returned_no_exact_question_spans"
            else:
                # Explicitly punctuated clauses are mandatory; model spans can
                # refine them but cannot introduce a new question.
                combined = []
                if len(explicit) > 1:
                    combined.extend(explicit)
                combined.extend(exact)
                facets = list(dict.fromkeys(combined))[:3]
        except Exception as exc:
            facets = explicit
            fallback = type(exc).__name__

        facets = [self._preserve_scope_anchors(facet, query) for facet in facets]
        facets = list(dict.fromkeys(facets))[:3] or [self._preserve_scope_anchors(query, query)]
        self._plan_cache[key] = facets
        state = self._state()
        if state is not None:
            state["plan_facets"] = list(facets)
            state["plan_fallback"] = fallback
        return list(facets), fallback

    def _retrieve(self, queries: Sequence[str], k: int, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        state = self._state()
        original_query = state.get("query", "") if state else ""
        query_list = [self._preserve_scope_anchors(str(q).strip(), original_query)
                      for q in queries if str(q).strip()]
        if not query_list:
            query_list = [original_query]

        query_vectors = np.asarray(self.backend.encode(query_list, query=True), dtype=np.float32)
        if query_vectors.ndim != 2 or query_vectors.shape[0] != len(query_list):
            raise RuntimeError("Backend.encode returned an invalid query embedding shape")
        if query_vectors.shape[1] != self.vectors.shape[1]:
            raise RuntimeError("Query and evidence embedding dimensions do not match")

        named_authority_by_query: List[List[str]] = []
        ranked_by_query: List[List[Dict[str, Any]]] = []
        for q_index, query in enumerate(query_list):
            compatible = [
                i for i, record in enumerate(self.records)
                if self._population_compatible(
                    query, record["population"], record.get("population_scope")
                )
            ]
            _aliases, named_authorities = self._query_authorities(query)
            named_authority_by_query.append(named_authorities)
            if named_authorities:
                named_set = set(named_authorities)
                constrained = [i for i in compatible if self.records[i]["authority"] in named_set]
                if constrained:
                    compatible = constrained
            if not compatible:
                ranked_by_query.append([])
                continue

            dense = self.vectors[compatible] @ query_vectors[q_index]
            lexical = self._lexical_bm25(_tokens(query), compatible)
            ranked: List[Tuple[float, str, int, float, float]] = []
            for position, idx in enumerate(compatible):
                record = self.records[idx]
                dense_score = float(dense[position])
                lexical_score = float(lexical.get(idx, 0.0))
                authority_prior = 0.025 if record["kind"] == "guideline" else 0.0
                score = dense_score + float(params["lexical_weight"]) * lexical_score + authority_prior
                ranked.append((score, record["id"], idx, dense_score, lexical_score))
            ranked.sort(key=lambda row: (-row[0], row[1]))

            rows: List[Dict[str, Any]] = []
            explicit_authorities = named_authorities if len(named_authorities) > 1 else []
            if explicit_authorities:
                buckets: Dict[str, List[Tuple[float, str, int, float, float]]] = defaultdict(list)
                for row in ranked:
                    buckets[self.records[row[2]]["authority"]].append(row)
                for depth in range(max((len(buckets[a]) for a in explicit_authorities), default=0)):
                    for authority in explicit_authorities:
                        if depth < len(buckets[authority]):
                            row = buckets[authority][depth]
                            rows.append(self._ranked_record(row, q_index))
                used = {item["id"] for item in rows}
                rows.extend(self._ranked_record(row, q_index) for row in ranked if row[1] not in used)
            else:
                rows = [self._ranked_record(row, q_index) for row in ranked]
            ranked_by_query.append(rows[:max(k, 1)])

        # Round-robin over independently ranked missing facets so one generic
        # high-scoring record cannot consume the entire supplement budget.
        selected: List[Dict[str, Any]] = []
        seen = set()
        positions = [0] * len(ranked_by_query)
        while len(selected) < max(k, 1):
            progress = False
            for q_index, rows in enumerate(ranked_by_query):
                while positions[q_index] < len(rows) and rows[positions[q_index]]["id"] in seen:
                    positions[q_index] += 1
                if positions[q_index] >= len(rows):
                    continue
                item = rows[positions[q_index]]
                positions[q_index] += 1
                seen.add(item["id"])
                selected.append(item)
                progress = True
                if len(selected) >= max(k, 1):
                    break
            if not progress:
                break

        supplied_ids = state.get("supplied_ids", []) if state else []
        query_compatible = []
        for record_id in supplied_ids:
            record = self.records[self._id_to_index[record_id]]
            if all(self._population_compatible(q, record["population"], record.get("population_scope"))
                   for q in query_list):
                query_compatible.append(record_id)
        for record_id in query_compatible:
            if record_id not in seen:
                item = dict(self.records[self._id_to_index[record_id]])
                item.update({"_score": -1.0, "_dense_score": 0.0, "_lexical_score": 0.0,
                             "_supplied": True, "_retrieval_query_index": None})
                selected.append(item)
                seen.add(record_id)
            else:
                for item in selected:
                    if item["id"] == record_id:
                        item["_supplied"] = True
                        break

        if state is not None:
            state["retrieval_events"].append({
                "queries": list(query_list),
                "named_authorities": named_authority_by_query,
                "facet_candidates": [[item["id"] for item in rows] for rows in ranked_by_query],
                "returned": [item["id"] for item in selected],
            })
        return selected

    def _ranked_record(self, row: Tuple[float, str, int, float, float], query_index: int) -> Dict[str, Any]:
        score, _record_id, index, dense_score, lexical_score = row
        item = dict(self.records[index])
        item.update({
            "_score": float(score),
            "_dense_score": float(dense_score),
            "_lexical_score": float(lexical_score),
            "_retrieval_query_index": query_index,
        })
        return item

    def _merge_results(self, first: Sequence[Dict[str, Any]], second: Sequence[Dict[str, Any]],
                       max_evidence: int) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        first_order: List[str] = []
        second_order: List[str] = []
        for source, order in ((first, first_order), (second, second_order)):
            for record in source:
                record_id = record["id"]
                if record_id not in by_id:
                    by_id[record_id] = dict(record)
                    order.append(record_id)
                else:
                    current = by_id[record_id]
                    if float(record.get("_score", -1.0)) > float(current.get("_score", -1.0)):
                        for key in ("_score", "_dense_score", "_lexical_score"):
                            if key in record:
                                current[key] = record[key]
                    if record.get("_supplied"):
                        current["_supplied"] = True

        state = self._state()
        required: List[str] = []
        if state:
            required.extend(state.get("supplied_ids", []))
            required.extend(state.get("protected_ids", []))
        required = list(dict.fromkeys(record_id for record_id in required if record_id in by_id))
        ordered_ids = required + [record_id for record_id in second_order if record_id not in required]
        ordered_ids += [record_id for record_id in first_order if record_id not in ordered_ids]
        overflow = max(0, len(required) - max_evidence)
        if state is not None:
            state["evidence_budget_overflow"] = max(state.get("evidence_budget_overflow", 0), overflow)
        limit = max(max_evidence, len(required))
        return [by_id[record_id] for record_id in ordered_ids[:limit]]

    def _critique(self, query: str, facets: Sequence[str], records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        result = super()._critique(query, facets, records)
        state = self._state()
        if state is not None:
            state["last_critique"] = result
            state["last_records"] = [dict(record) for record in records]
            state["last_facets"] = list(facets)
            for item in result.get("facets", []):
                if item.get("supported"):
                    for citation in item.get("citations", []):
                        record_id = str(citation.get("id", ""))
                        if record_id and record_id not in state["protected_ids"]:
                            state["protected_ids"].append(record_id)
        return result

    @staticmethod
    def _followup_queries(critique: Mapping[str, Any], facets: Sequence[str]) -> List[str]:
        queries: List[str] = []
        # Per-request support IDs are recorded by _critique. Query only the
        # original missing facet, never a model-invented follow-up question.
        for item in critique.get("facets", []):
            if not item.get("supported"):
                facet = str(item.get("facet", "")).strip()
                if facet:
                    queries.append(facet)
        if not queries:
            queries = list(facets)
        return list(dict.fromkeys(queries))[:3]

    def _write_claims(self, query: str, facets: Sequence[str], records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        state = self._state()
        if not state or not state.get("allow_partial"):
            return super()._write_claims(query, facets, records)
        system = (
            "Answer only the listed evidence needs that are supported by the supplied adult blood-pressure sources. "
            "Do not answer any unlisted or unsupported part of the question, and do not add management, treatment, "
            "or follow-up recommendations. Use only supplied evidence and answer in the language of the question. "
            "Return only JSON: {\"claims\":[{\"text\":\"one supported factual statement\",\"citations\":[{\"id\":\"exact id\",\"quote\":\"exact contiguous source excerpt\"}]}]}. "
            "Every factual clause needs an exact quote. If none of the listed needs can be answered, return an empty claims array."
        )
        user = json.dumps({
            "question": query,
            "supported_evidence_needs_only": list(facets),
            "sources": self._evidence_payload(records, query),
        }, ensure_ascii=False)
        try:
            parsed = _parse_json(self.backend.chat(system=system, user=user))
        except Exception as exc:
            raise RuntimeError("Partial answer writer failed; no clinical answer was produced") from exc
        claims = parsed.get("claims")
        if not isinstance(claims, list):
            raise RuntimeError("Partial answer writer returned no claims array")
        return [item for item in claims if isinstance(item, dict)]

    def _abstention(self, query: str, mode: str, action: str, rounds: int,
                    first_ids: Sequence[str], evidence_ids: Sequence[str], trace: List[Dict[str, Any]],
                    features: Mapping[str, float], params: Mapping[str, Any], reason: str,
                    started: float, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        state = self._state()
        critique = state.get("last_critique", {}) if state else {}
        if (state and state.get("allow_partial") and mode != "adaptive_nocrit"
                and reason == "one_or_more_evidence_needs_unverified"):
            items = critique.get("facets", [])
            supported = [item for item in items if item.get("supported")]
            missing = [item for item in items if not item.get("supported")]
            support_ids = list(dict.fromkeys(
                str(citation.get("id", ""))
                for item in supported for citation in item.get("citations", [])
                if citation.get("id")
            ))
            available = {record["id"]: record for record in state.get("last_records", [])}
            support_records = [available[record_id] for record_id in support_ids if record_id in available]
            if supported and support_records:
                facets = [str(item.get("facet", "")).strip() for item in supported if item.get("facet")]
                claims = self._write_claims(query, facets, support_records)
                valid_claims, errors = self._validate_claims(claims, support_records)
                if not claims or errors or len(valid_claims) != len(claims):
                    raise RuntimeError("Partial answer citation validation failed")
                validations = self._semantic_validate(query, valid_claims, support_records)
                if len(validations) != len(valid_claims) or not all(validations):
                    raise RuntimeError("Partial answer semantic validation failed")
                answer, citations = self._format_answer(valid_claims)
                missing_facets = [str(item.get("facet", "")).strip() for item in missing if item.get("facet")]
                if re.search(r"[\u3400-\u9fff]", query):
                    answer += "\n\n尚缺依据、未作答：" + "；".join(missing_facets)
                else:
                    answer += "\n\nNot answered due to insufficient evidence: " + "; ".join(missing_facets)
                trace.append({
                    "stage": "answer",
                    "action": "partial",
                    "reason": "supported_facets_only; remaining_facets_explicitly_unanswered",
                    "supported_facets": facets,
                    "unanswered_facets": missing_facets,
                    "retrieved": citations,
                })
                return {
                    "answer": answer,
                    "citations": citations,
                    "abstain": False,
                    "partial": True,
                    "initial_action": action,
                    "rounds": rounds,
                    "first_evidence": list(first_ids),
                    "evidence": list(evidence_ids),
                    "trace": trace,
                    "sources": self._sources_for_ids(citations, support_records),
                    "claims": valid_claims,
                    "supported_facets": facets,
                    "unanswered_facets": missing_facets,
                    "diagnostics": {
                        "mode": mode,
                        "decision_reason": "partial_answer_with_unverified_facets",
                        "base_abstention_reason": reason,
                        "features": dict(features),
                        "params": dict(params),
                        "semantic_critic": True,
                        "index": self.index_status,
                        "supported_fraction": float(critique.get("supported_fraction", 0.0)),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                }
        result = super()._abstention(
            query, mode, action, rounds, first_ids, evidence_ids, trace,
            features, params, reason, started, extra,
        )
        if state:
            result["partial"] = False
        return result

    def respond(self, query: str, supplied: Optional[Iterable[str]] = None, mode: str = "adaptive",
                params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        query = str(query or "").strip()
        effective = self._params(params)
        resolved = self._resolve_supplied(supplied)
        compatible = [
            record["id"] for record in resolved
            if self._population_compatible(query, record["population"], record.get("population_scope"))
        ]
        # Explicitly supplied evidence is part of the request's evidence budget,
        # even when it exceeds the baseline's default eight-record window.
        effective["max_evidence"] = max(effective["max_evidence"], len(compatible))
        _reason, basis = decide_scope(query, resolved)
        self._request_local.value = {
            "query": query,
            "allow_partial": effective["allow_partial"],
            "supplied_ids": compatible,
            "protected_ids": [],
            "retrieval_events": [],
            "evidence_budget_overflow": 0,
            "scope_basis": basis,
            "scope_topics": [],
            "scope_authorities": [],
            "last_critique": {},
            "last_records": [],
            "last_facets": [],
            "plan_facets": [],
            "plan_fallback": None,
        }
        try:
            result = super().respond(query, supplied=supplied, mode=mode, params=effective)
            state = self._state() or {}
            result["engine"] = "improved"
            if not result.get("partial"):
                result.setdefault("partial", False)
            diagnostics = result.setdefault("diagnostics", {})
            diagnostics["engine_version"] = VERSION
            diagnostics["scope_basis"] = state.get("scope_basis")
            diagnostics["scope_topics"] = state.get("scope_topics", [])
            diagnostics["scope_authorities"] = state.get("scope_authorities", [])
            diagnostics["plan_facets"] = state.get("plan_facets", [])
            diagnostics["plan_fallback"] = state.get("plan_fallback")
            diagnostics["facet_retrieval"] = state.get("retrieval_events", [])
            diagnostics["preserved_supplied"] = state.get("supplied_ids", [])
            diagnostics["protected_first_pass"] = state.get("protected_ids", [])
            diagnostics["facet_support"] = [
                {"facet": item.get("facet"), "supported": bool(item.get("supported")),
                 "citations": [c.get("id") for c in item.get("citations", [])]}
                for item in state.get("last_critique", {}).get("facets", [])
            ]
            diagnostics["evidence_budget_overflow"] = state.get("evidence_budget_overflow", 0)
            diagnostics["allow_partial"] = effective["allow_partial"]
            return result
        finally:
            try:
                del self._request_local.value
            except AttributeError:
                pass

