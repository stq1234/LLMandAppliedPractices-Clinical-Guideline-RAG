"""Summarize per-question logs, paired outcomes, and supplement utility."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate import read_jsonl
from grading import grade_case


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / (1 + z2 / n)
    return max(0.0, center - half), min(1.0, center + half)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("status") == "ok"]
    answerable = [r for r in valid if (r.get("grading") or {}).get("answerable")]
    scope_out = [r for r in valid if (r.get("grading") or {}).get("scope_out")]
    unanswerable = [
        r for r in valid
        if (r.get("grading") or {}).get("unanswerable")
        and not (r.get("grading") or {}).get("scope_out")
    ]
    with_gold = [r for r in answerable if (r.get("grading") or {}).get("gold_ids")]
    supplemented = [r for r in valid if int(r.get("rounds", 0) or 0) >= 2]
    def usage_value(row: dict[str, Any], key: str, fallback: float = 0.0) -> float:
        value = (row.get("usage") or {}).get(key)
        return fallback if value is None else float(value or 0.0)

    def logical_calls(row: dict[str, Any]) -> float:
        return usage_value(row, "chat_calls")

    def physical_calls(row: dict[str, Any]) -> float:
        return usage_value(row, "real_chat_calls", logical_calls(row))

    def replayed_calls(row: dict[str, Any]) -> float:
        return usage_value(row, "replayed_chat_calls")

    cold_responses = [
        r for r in valid
        if logical_calls(r) > 0 and replayed_calls(r) == 0 and physical_calls(r) == logical_calls(r)
    ]
    replay_only_responses = [
        r for r in valid if logical_calls(r) > 0 and physical_calls(r) == 0 and replayed_calls(r) > 0
    ]
    mixed_responses = [
        r for r in valid if physical_calls(r) > 0 and replayed_calls(r) > 0
    ]
    supported_count = sum(bool(r["grading"]["supported_keypoint_match"]) for r in answerable)
    supported_ci = wilson(supported_count, len(answerable))
    unans_refused = sum(bool(r["grading"]["abstain"]) for r in unanswerable)
    unans_ci = wilson(unans_refused, len(unanswerable))
    scope_refused = sum(bool(r["grading"]["abstain"]) for r in scope_out)
    scope_ci = wilson(scope_refused, len(scope_out))
    first_incomplete = [
        r for r in with_gold
        if (r.get("grading") or {}).get("first_gold_coverage") is not None
        and (r["grading"]["first_gold_coverage"] < 1.0)
    ]
    answered = [r for r in answerable if not (r.get("grading") or {}).get("abstain")]
    keypoint_errors = [r for r in answered if not (r.get("grading") or {}).get("expected_match")]
    citation_incomplete_answers = [r for r in answered if not (r.get("grading") or {}).get("citations_complete")]
    wrong_with_complete_cites = [
        r for r in keypoint_errors if (r.get("grading") or {}).get("citations_complete")
    ]
    right_keypoint_missing_cites = [
        r for r in answered
        if (r.get("grading") or {}).get("expected_match")
        and not (r.get("grading") or {}).get("citations_complete")
    ]
    adequate_evidence_refusals = [
        r for r in with_gold
        if (r.get("grading") or {}).get("abstain")
        and (r.get("grading") or {}).get("final_evidence_complete")
    ]
    usable_supplements = [
        r for r in with_gold
        if r.get("supplement_attempted", int(r.get("rounds", 0) or 0) >= 2)
    ]
    initial_need_rows = [r for r in valid if (r.get("grading") or {}).get("initial_need") is not None]
    initial_need_required = [r for r in initial_need_rows if r["grading"]["initial_need"]]
    initial_need_not_required = [r for r in initial_need_rows if not r["grading"]["initial_need"]]
    initial_executed = lambda r: bool(r.get("initial_retrieval_executed"))
    initial_decision_matches = sum(
        initial_executed(r) == bool(r["grading"]["initial_need"]) for r in initial_need_rows
    )
    model_scope_out_rows = [r for r in valid if (r.get("grading") or {}).get("model_scope_out")]
    false_scope_out_rows = [
        r for r in valid
        if (r.get("grading") or {}).get("model_scope_out")
        and not (r.get("grading") or {}).get("scope_out")
    ]
    false_scope_out_answerable = [r for r in answerable if (r.get("grading") or {}).get("model_scope_out")]
    false_scope_out_unanswerable = [r for r in unanswerable if (r.get("grading") or {}).get("model_scope_out")]
    model_personalized_rows = [r for r in valid if (r.get("grading") or {}).get("model_personalized_refusal")]
    zero_round_answerable_refusals = [
        r for r in answerable
        if (r.get("grading") or {}).get("abstain") and int(r.get("rounds", 0) or 0) == 0
    ]
    return {
        "mode": rows[0].get("mode") if rows else None,
        "n_attempted": len(rows),
        "total_model_attempts": sum(int(r.get("_attempt_count", 1) or 1) for r in rows),
        "n_valid": len(valid),
        "n_errors": sum(r.get("status") != "ok" for r in rows),
        "n_error_attempts": sum(int(r.get("_error_attempt_count", int(r.get("status") != "ok")) or 0) for r in rows),
        "n_recovered_after_error": sum(bool(r.get("_recovered_after_error")) for r in rows),
        "answerable_n": len(answerable),
        "unanswerable_n": len(unanswerable),
        "scope_out_n": len(scope_out),
        "keypoint_match_rate": rate(sum(bool(r["grading"]["keypoint_match"]) for r in answerable), len(answerable)),
        "supported_keypoint_match_rate": rate(sum(bool(r["grading"]["supported_keypoint_match"]) for r in answerable), len(answerable)),
        "supported_keypoint_match_ci95_low": supported_ci[0],
        "supported_keypoint_match_ci95_high": supported_ci[1],
        "gold_citation_coverage_mean": mean([
            float(r["grading"]["citation_coverage"])
            for r in with_gold if r["grading"]["citation_coverage"] is not None
        ]),
        "complete_gold_citation_rate": rate(
            sum(bool(r["grading"]["citations_complete"]) for r in with_gold), len(with_gold)
        ),
        "answered_n_answerable": len(answered),
        "answerable_keypoint_error_n": len(keypoint_errors),
        "answerable_keypoint_error_rate_among_answers": rate(len(keypoint_errors), len(answered)),
        "answerable_citation_incomplete_answer_n": len(citation_incomplete_answers),
        "answerable_citation_incomplete_rate_among_answers": rate(len(citation_incomplete_answers), len(answered)),
        "wrong_keypoint_with_complete_gold_cites_n": len(wrong_with_complete_cites),
        "keypoint_match_missing_gold_cites_n": len(right_keypoint_missing_cites),
        "adequate_gold_evidence_but_refused_n": len(adequate_evidence_refusals),
        "adequate_gold_evidence_but_refused_rate_among_adequate_evidence": rate(
            len(adequate_evidence_refusals),
            sum(bool((r.get("grading") or {}).get("final_evidence_complete")) for r in with_gold),
        ),
        "unsupported_answer_rate": rate(
            sum(bool((r.get("grading") or {}).get("unsupported_answer")) for r in valid if not (r.get("grading") or {}).get("abstain")),
            sum(not bool((r.get("grading") or {}).get("abstain")) for r in valid),
        ),
        "unanswerable_correct_refusal_rate": rate(
            sum(bool(r["grading"]["abstain"]) for r in unanswerable), len(unanswerable)
        ),
        "unanswerable_refusal_ci95_low": unans_ci[0],
        "unanswerable_refusal_ci95_high": unans_ci[1],
        "scope_out_correct_refusal_rate": rate(
            sum(bool(r["grading"]["abstain"]) for r in scope_out), len(scope_out)
        ),
        "labelled_scope_out_n": len(scope_out),
        "scope_out_refusal_ci95_low": scope_ci[0],
        "scope_out_refusal_ci95_high": scope_ci[1],
        "model_scope_out_action_n": sum(bool((r.get("grading") or {}).get("model_out_action")) for r in valid),
        "model_scope_out_reason_n": len(model_scope_out_rows),
        "model_scope_out_false_positive_n": len(false_scope_out_rows),
        "model_scope_out_false_positive_rate_among_non_scope_out": rate(
            len(false_scope_out_rows), len(answerable) + len(unanswerable)
        ),
        "model_scope_out_on_answerable_n": len(false_scope_out_answerable),
        "model_scope_out_on_in_scope_unanswerable_n": len(false_scope_out_unanswerable),
        "model_personalized_refusal_n": len(model_personalized_rows),
        "answerable_zero_round_refusal_n": len(zero_round_answerable_refusals),
        "in_scope_unanswerable_zero_round_refusal_n": sum(
            bool(r["grading"]["abstain"]) and int(r.get("rounds", 0) or 0) == 0
            for r in unanswerable
        ),
        "rounds_mean": mean([float(r.get("rounds", 0) or 0) for r in valid]),
        "two_round_rate": rate(len(supplemented), len(valid)),
        "initial_need_labelled_n": len(initial_need_rows),
        "initial_need_retrieval_required_n": len(initial_need_required),
        "initial_need_retrieval_not_required_n": len(initial_need_not_required),
        "initial_need_false_scope_out_n": sum(bool(r["grading"]["scope_out"]) for r in initial_need_not_required),
        "initial_need_false_context_sufficient_n": sum(not bool((r.get("grading") or {}).get("scope_out")) for r in initial_need_not_required),
        "initial_retrieval_executed_when_required_n": sum(initial_executed(r) for r in initial_need_required),
        "missed_initial_retrieval_n": sum(not initial_executed(r) for r in initial_need_required),
        "initial_retrieval_without_need_n": sum(initial_executed(r) for r in initial_need_not_required),
        "initial_decision_match_rate": rate(initial_decision_matches, len(initial_need_rows)),
        "scope_out_with_retrieval_n": sum(initial_executed(r) for r in valid if (r.get("grading") or {}).get("scope_out")),
        "in_scope_unanswerable_with_retrieval_n": sum(initial_executed(r) for r in unanswerable),
        "in_scope_unanswerable_n": len(unanswerable),
        "prompt_tokens_mean": mean([float((r.get("usage") or {}).get("prompt_tokens", 0) or 0) for r in valid]),
        "completion_tokens_mean": mean([float((r.get("usage") or {}).get("completion_tokens", (r.get("usage") or {}).get("output_tokens", 0)) or 0) for r in valid]),
        "chat_calls_mean": mean([float((r.get("usage") or {}).get("chat_calls", 0) or 0) for r in valid]),
        "logical_chat_calls_mean": mean([logical_calls(r) for r in valid]),
        "physical_chat_calls_mean": mean([physical_calls(r) for r in valid]),
        "replayed_chat_calls_mean": mean([replayed_calls(r) for r in valid]),
        "real_model_wait_seconds_mean": mean([
            usage_value(r, "real_model_wait_seconds", usage_value(r, "chat_seconds")) for r in valid
        ]),
        "nominal_chat_seconds_mean": mean([
            usage_value(r, "nominal_chat_seconds", usage_value(r, "chat_seconds")) for r in valid
        ]),
        "singleflight_wait_seconds_mean": mean([usage_value(r, "singleflight_wait_seconds") for r in valid]),
        "queue_wait_seconds_mean": mean([usage_value(r, "queue_wait_seconds") for r in valid]),
        "fully_cold_model_response_n": len(cold_responses),
        "fully_cold_wall_latency_seconds_mean": mean([float(r.get("latency_seconds", 0) or 0) for r in cold_responses]),
        "cache_only_response_n": len(replay_only_responses),
        "mixed_real_and_replayed_response_n": len(mixed_responses),
        "latency_seconds_mean": mean([float(r.get("latency_seconds", 0) or 0) for r in valid]),
        "first_round_complete_gold_rate": rate(
            sum(float(r["grading"]["first_gold_coverage"]) >= 1.0 for r in with_gold if r["grading"]["first_gold_coverage"] is not None),
            sum(r["grading"]["first_gold_coverage"] is not None for r in with_gold),
        ),
        "first_incomplete_n": len(first_incomplete),
        "gold_recovery_added_any_required_evidence_n": sum(
            bool((r.get("grading") or {}).get("supplement_added_gold")) for r in usable_supplements
        ),
        "gold_recovery_rate_among_answerable_supplements": rate(
            sum(bool((r.get("grading") or {}).get("supplement_added_gold")) for r in usable_supplements), len(usable_supplements)
        ),
        "supplement_rescued_keypoint_rate_given_first_incomplete": rate(
            sum(bool((r.get("grading") or {}).get("rescued_supported_keypoint_match")) for r in usable_supplements if r in first_incomplete),
            sum(r in first_incomplete for r in usable_supplements),
        ),
        "nonproductive_gold_recovery_supplement_n": sum(
            not bool((r.get("grading") or {}).get("supplement_added_gold")) for r in usable_supplements
        ),
        "nonproductive_gold_recovery_supplement_rate": rate(
            sum(not bool((r.get("grading") or {}).get("supplement_added_gold")) for r in usable_supplements), len(usable_supplements)
        ),
    }


def paired_comparisons(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    by_id_mode = {(str(r.get("id")), str(r.get("mode"))): r for r in rows if r.get("split") == split}
    modes = sorted({str(r.get("mode")) for r in rows if r.get("split") == split})
    comparisons = []
    question_ids = sorted({qid for qid, _ in by_id_mode})
    for candidate in modes:
        for reference in modes:
            if candidate == reference:
                continue
            pairs = [
                (by_id_mode[(qid, reference)], by_id_mode[(qid, candidate)])
                for qid in question_ids
                if (qid, reference) in by_id_mode and (qid, candidate) in by_id_mode
                and by_id_mode[(qid, reference)].get("status") == "ok"
                and by_id_mode[(qid, candidate)].get("status") == "ok"
            ]
            gains = losses = ties = 0
            answerable_n = answerable_gains = answerable_losses = 0
            refusal_n = refusal_gains = refusal_losses = 0
            for ref_row, candidate_row in pairs:
                ref_grade = ref_row.get("grading") or {}
                candidate_grade = candidate_row.get("grading") or {}
                ref_ok = bool(ref_grade.get("operational_success"))
                candidate_ok = bool(candidate_grade.get("operational_success"))
                if candidate_ok and not ref_ok:
                    gains += 1
                elif ref_ok and not candidate_ok:
                    losses += 1
                else:
                    ties += 1
                if ref_grade.get("answerable") and candidate_grade.get("answerable"):
                    answerable_n += 1
                    ref_answer = bool(ref_grade.get("supported_keypoint_match"))
                    candidate_answer = bool(candidate_grade.get("supported_keypoint_match"))
                    if candidate_answer and not ref_answer:
                        answerable_gains += 1
                    elif ref_answer and not candidate_answer:
                        answerable_losses += 1
                else:
                    refusal_n += 1
                    ref_refusal = bool(ref_grade.get("correct_refusal"))
                    candidate_refusal = bool(candidate_grade.get("correct_refusal"))
                    if candidate_refusal and not ref_refusal:
                        refusal_gains += 1
                    elif ref_refusal and not candidate_refusal:
                        refusal_losses += 1
            comparisons.append({
                "split": split,
                "candidate": candidate,
                "reference": reference,
                "n_paired": len(pairs),
                "candidate_gains": gains,
                "candidate_losses": losses,
                "ties": ties,
                "answerable_n": answerable_n,
                "candidate_supported_keypoint_gains": answerable_gains,
                "candidate_supported_keypoint_losses": answerable_losses,
                "nonanswerable_refusal_n": refusal_n,
                "candidate_correct_refusal_gains": refusal_gains,
                "candidate_correct_refusal_losses": refusal_losses,
            })
    return comparisons


def supplement_diagnostics(rows: list[dict[str, Any]], split: str, mode: str) -> dict[str, Any]:
    by_id_mode = {(str(r.get("id")), str(r.get("mode"))): r for r in rows if r.get("split") == split}
    adaptive = [r for r in rows if r.get("split") == split and r.get("mode") == mode and r.get("status") == "ok"]
    eligible = [r for r in adaptive if (r.get("grading") or {}).get("answerable") and (r.get("grading") or {}).get("gold_ids")]
    incomplete = [r for r in eligible if (r.get("grading") or {}).get("first_gold_coverage") is not None and r["grading"]["first_gold_coverage"] < 1.0]
    corpus_feasible = [r for r in eligible if r.get("gold_in_corpus", True)]
    corpus_mismatch = [r for r in eligible if not r.get("gold_in_corpus", True)]
    attempted = [r for r in corpus_feasible if r.get("supplement_attempted", int(r.get("rounds", 0) or 0) >= 2)]
    new_gold = [r for r in attempted if (r.get("grading") or {}).get("new_gold_ids")]
    rescued = [r for r in attempted if (r.get("grading") or {}).get("rescued_supported_keypoint_match")]
    nonproductive = [r for r in attempted if not (r.get("grading") or {}).get("new_gold_ids")]
    single_route_supplements = [
        r for r in attempted
        if int(r.get("rounds", 0) or 0) >= 2 and r.get("effective_route") == "single"
    ]
    single_mode_rescues = []
    single_mode_errors = []
    single_mode_ties = []
    for r in single_route_supplements:
        single = by_id_mode.get((str(r.get("id")), "single"))
        if not single or single.get("status") != "ok":
            continue
        baseline_ok = bool((single.get("grading") or {}).get("supported_keypoint_match"))
        supplement_ok = bool((r.get("grading") or {}).get("supported_keypoint_match"))
        if supplement_ok and not baseline_ok:
            single_mode_rescues.append(r)
        elif baseline_ok and not supplement_ok:
            single_mode_errors.append(r)
        else:
            single_mode_ties.append(r)

    missed_candidates = [
        r for r in corpus_feasible
        if (r.get("grading") or {}).get("first_gold_coverage") is not None
        and r["grading"]["first_gold_coverage"] < 1.0
        and not r.get("supplement_attempted", int(r.get("rounds", 0) or 0) >= 2)
    ]
    missed_evaluable = []
    missed = []
    forced_multi_recovered = []
    for r in incomplete:
        multi = by_id_mode.get((str(r.get("id")), "multi"))
        if multi and multi.get("status") == "ok":
            if r in missed_candidates:
                missed_evaluable.append(r)
            g = multi.get("grading") or {}
            full_gold = g.get("final_gold_coverage") == 1.0 and bool(g.get("supported_keypoint_match"))
            if full_gold:
                forced_multi_recovered.append(r)
                if r in missed_candidates:
                    missed.append(r)

    adequate_evidence_refusals = [
        r for r in corpus_feasible
        if (r.get("grading") or {}).get("abstain")
        and (r.get("grading") or {}).get("final_evidence_complete")
    ]
    attempted_count = len(attempted)
    missed_rate = rate(len(missed), len(missed_evaluable))
    return {
        "split": split,
        "mode": mode,
        "answerable_with_gold_n": len(eligible),
        "gold_in_corpus_feasible_n": len(corpus_feasible),
        "gold_missing_from_corpus_n": len(corpus_mismatch),
        "first_round_missing_gold_n": len([r for r in corpus_feasible if r in incomplete]),
        "two_round_responses_n": sum(int(r.get("rounds", 0) or 0) >= 2 for r in corpus_feasible),
        "second_round_trace_n": attempted_count,
        "second_round_added_required_gold_n": len(new_gold),
        "second_round_added_required_gold_rate": rate(len(new_gold), attempted_count),
        "second_round_rescued_supported_keypoint_n": len(rescued),
        "second_round_rescued_rate_given_first_incomplete_and_attempted": rate(
            sum(r in incomplete for r in rescued),
            sum(r in incomplete for r in attempted),
        ),
        "second_round_no_new_required_gold_n": len(nonproductive),
        "second_round_no_new_required_gold_rate": rate(len(nonproductive), attempted_count),
        "adaptive_single_route_second_round_n": len(single_route_supplements),
        "single_route_supplement_paired_with_single_n": len(single_mode_rescues) + len(single_mode_errors) + len(single_mode_ties),
        "single_route_supplement_rescues_vs_single_n": len(single_mode_rescues),
        "single_route_supplement_new_errors_vs_single_n": len(single_mode_errors),
        "single_route_supplement_ties_vs_single_n": len(single_mode_ties),
        "adequate_gold_evidence_but_abstained_n": len(adequate_evidence_refusals),
        "missed_supplement_risk_n": len(missed_candidates),
        "missed_supplement_evaluable_n": len(missed_evaluable),
        "missed_supplement_recovered_by_paired_forced_multi_n": len(missed),
        "missed_supplement_rate_among_evaluable_at_risk": missed_rate,
        "forced_multi_recovered_all_gold_after_first_gap_n": len(forced_multi_recovered),
        "forced_multi_upper_bound_definition": (
            "Among answerable items whose first evidence missed required IDs, forced multi retrieved all gold and matched expected keypoints with complete gold citations; this is an observed trajectory upper bound, not proof that every second retrieval can recover evidence."
        ),
        "in_scope_unanswerable_excluded_from_gold_recovery_oracle_n": sum(
            bool((r.get("grading") or {}).get("unanswerable"))
            and not bool((r.get("grading") or {}).get("scope_out"))
            for r in adaptive
        ),
        "scope_out_excluded_from_retrieval_oracle_n": sum(
            bool((r.get("grading") or {}).get("scope_out")) for r in adaptive
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    raw_rows = read_jsonl(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    attempts: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (str(row.get("split")), str(row.get("id")), str(row.get("mode")))
        attempts[key].append(row)
    rows = []
    for history in attempts.values():
        valid_history = [row for row in history if row.get("status") == "ok"]
        selected = dict(valid_history[-1] if valid_history else history[-1])
        selected["_attempt_count"] = len(history)
        selected["_error_attempt_count"] = sum(row.get("status") != "ok" for row in history)
        selected["_recovered_after_error"] = bool(valid_history and any(row.get("status") != "ok" for row in history))
        rows.append(selected)
    for row in rows:
        if row.get("status") == "ok":
            # Recompute against immutable question fields and raw response;
            # this also keeps reports independent of earlier grading versions.
            row["grading"] = grade_case(row, row)
    # Development logs also contain adaptive force-action trajectories used
    # only for calibration; partial fixed-mode dev baselines are not reported.
    report_rows = [
        row for row in rows
        if row.get("split") == "test"
        or (row.get("split") == "dev" and row.get("mode") in ("adaptive", "adaptive_learned"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report_rows:
        grouped[(str(row.get("split")), str(row.get("mode")))].append(row)
    summaries = []
    for (split, _), group in sorted(grouped.items()):
        result = summarize_group(group)
        result["split"] = split
        summaries.append(result)
    paired = paired_comparisons(report_rows, "test")
    adaptive_modes = sorted({
        str(r.get("mode")) for r in report_rows
        if r.get("mode") in ("adaptive", "adaptive_learned")
    })
    supplements = [
        supplement_diagnostics(report_rows, split, mode)
        for split in ("dev", "test") for mode in adaptive_modes
    ]
    calibration_replays = [
        row for row in raw_rows
        if row.get("split") == "dev" and row.get("mode") == "adaptive_force_reuse"
    ]
    def usage_stat(row: dict[str, Any], key: str, fallback: float = 0.0) -> float:
        value = (row.get("usage") or {}).get(key)
        return fallback if value is None else float(value or 0.0)

    calibration_replay_stats = {
        "n_route_replays": sum(row.get("status") == "ok" for row in calibration_replays),
        "n_errors": sum(row.get("status") != "ok" for row in calibration_replays),
        "logical_chat_calls": sum(usage_stat(row, "chat_calls") for row in calibration_replays),
        "physical_chat_calls": sum(usage_stat(row, "real_chat_calls", usage_stat(row, "chat_calls")) for row in calibration_replays),
        "replayed_chat_calls": sum(usage_stat(row, "replayed_chat_calls") for row in calibration_replays),
        "prompt_tokens": sum(usage_stat(row, "prompt_tokens") for row in calibration_replays),
        "completion_tokens": sum(usage_stat(row, "completion_tokens", usage_stat(row, "output_tokens")) for row in calibration_replays),
        "mean_wall_latency_seconds": mean([float(row.get("latency_seconds", 0) or 0) for row in calibration_replays if row.get("status") == "ok"]),
    }
    for name, records in (("summary.csv", summaries), ("paired_gains_losses.csv", paired)):
        path = args.outdir / name
        fields = list(dict.fromkeys(k for record in records for k in record))
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    detail = {
        "modes": summaries,
        "paired_comparisons": paired,
        "supplement_diagnostics": supplements,
        "dev_calibration_route_replays": calibration_replay_stats,
    }
    (args.outdir / "summary.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(detail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

