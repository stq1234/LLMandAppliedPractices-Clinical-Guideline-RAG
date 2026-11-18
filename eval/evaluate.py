"""Run fixed-route and adaptive clinical RAG evaluations on dev/test JSONL."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from grading import grade_case


DEFAULT_MODES = ("reuse", "single", "multi", "adaptive", "adaptive_nocrit")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_no}")
                rows.append(row)
    return rows


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature_row(system: Any, question: dict[str, Any]) -> dict[str, Any]:
    # Only query and user-supplied evidence reach the model-side feature API.
    result = system.features(question["query"], supplied=question.get("supplied"))
    result = jsonable(result)
    if not isinstance(result, dict):
        raise TypeError(f"System.features must return a mapping, got {type(result).__name__}")
    values = {}
    for key, value in result.items():
        if isinstance(value, bool):
            values[str(key)] = int(value)
        elif isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
            values[str(key)] = float(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--ids", nargs="*", help="optional exact question IDs for focused route replays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--params", type=Path, help="JSON params passed to adaptive mode")
    parser.add_argument("--disable-replay-cache", action="store_true", help="use the physical Backend directly for retry accounting")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("--workers must be between 1 and 3")

    project = args.project.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    from clinical import System, VERSION as clinical_version  # imported after setting the project path

    questions_path = project / "data" / "questions.jsonl"
    corpus_path = project / "data" / "evidence.jsonl"
    if not questions_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"Expected {questions_path} and {corpus_path}")
    questions = read_jsonl(questions_path)
    corpus_rows = read_jsonl(corpus_path)
    corpus_ids = {str(row.get("id")) for row in corpus_rows}
    if args.split != "all":
        questions = [q for q in questions if q.get("split") == args.split]
    if args.ids is not None:
        selected_ids = set(args.ids)
        questions = [q for q in questions if str(q.get("id")) in selected_ids]
        missing_ids = selected_ids - {str(q.get("id")) for q in questions}
        if missing_ids:
            raise ValueError(f"Unknown question IDs for split={args.split}: {sorted(missing_ids)}")
    if not questions:
        raise ValueError(f"No questions selected for split={args.split}")
    if len({str(q.get("id")) for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique in the selected split")

    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else None
    cache_path = project / "results" / "chat_cache.sqlite3"
    if args.disable_replay_cache:
        from model_backend import Backend
        backend = Backend()
    else:
        from eval.replay_backend import ReplayBackend
        backend = ReplayBackend(cache_path=str(cache_path), engine_version=clinical_version)

    def route_config(mode: str) -> tuple[str, dict[str, Any] | None]:
        mode_params = dict(params or {}) if mode in ("adaptive", "adaptive_learned") or mode.startswith("adaptive_force_") else None
        engine_mode = "adaptive" if mode == "adaptive_learned" or mode.startswith("adaptive_force_") else mode
        if mode.startswith("adaptive_force_"):
            forced = mode.removeprefix("adaptive_force_")
            if forced not in ("reuse", "single", "multi"):
                raise ValueError(f"Unsupported forced adaptive route: {mode}")
            mode_params["force_action"] = forced
        return engine_mode, mode_params

    # Separate System instances isolate each mode's per-query plan cache, so a
    # cached planner response from one mode cannot lower another mode's costs.
    systems = {
        mode: System(backend, corpus_path="data/evidence.jsonl", index_dir="index")
        for mode in args.modes
    }
    feature_system = systems[args.modes[0]]
    features_by_id = {str(q["id"]): feature_row(feature_system, q) for q in questions}

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str]] = set()
    if output.exists():
        for row in read_jsonl(output):
            if row.get("status") == "ok":
                completed.add((str(row.get("id")), str(row.get("mode"))))

    tasks = [
        (q, mode)
        for q in questions
        for mode in args.modes
        if (str(q["id"]), mode) not in completed
    ]
    start_all = time.perf_counter()

    def run_one(question: dict[str, Any], mode: str) -> dict[str, Any]:
        query = str(question["query"])
        supplied = question.get("supplied")
        engine_mode, mode_params = route_config(mode)
        backend.reset_usage()
        started = time.perf_counter()
        record: dict[str, Any] = {
            "id": str(question["id"]),
            "family": question.get("family"),
            "split": question.get("split"),
            "mode": mode,
            "query": query,
            "supplied": supplied,
            "gold": question.get("gold", []),
            "gold_in_corpus": set(str(x) for x in question.get("gold", [])) <= corpus_ids,
            "gold_missing_from_corpus": sorted(set(str(x) for x in question.get("gold", [])) - corpus_ids),
            "expected": question.get("expected"),
            "expected_groups": question.get("expected_groups"),
            "unanswerable": bool(question.get("unanswerable", False)),
            "scope_out": bool(question.get("scope_out", False)),
            "initial_need": question.get("initial_need"),
            "complexity": question.get("complexity"),
            "features": features_by_id[str(question["id"])],
            "params": mode_params,
        }
        try:
            response = jsonable(
                systems[mode].respond(query, supplied=supplied, mode=engine_mode, params=mode_params)
            )
            if not isinstance(response, dict):
                raise TypeError(f"System.respond must return a mapping, got {type(response).__name__}")
            record.update(response)
            trace = response.get("trace") or []
            router_steps = [step for step in trace if isinstance(step, dict) and step.get("stage") == "router"]
            supplement_steps = [
                step for step in trace
                if isinstance(step, dict) and step.get("stage") == "retrieval" and step.get("action") == "supplement"
            ]
            record["effective_route"] = router_steps[-1].get("action") if router_steps else response.get("initial_action")
            record["adaptive_fallback"] = any(
                step.get("reason") == "supplied_evidence_gap_triggered_retrieval_fallback"
                for step in router_steps
            )
            record["supplement_attempted"] = bool(supplement_steps)
            record["supplement_reason"] = supplement_steps[-1].get("reason") if supplement_steps else None
            record["initial_retrieval_executed"] = any(
                isinstance(step, dict) and step.get("stage") == "retrieval" and step.get("action") == "initial"
                for step in trace
            )
            record["status"] = "ok"
        except Exception as exc:  # failures stay distinct from a clinical abstention
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)[:1000]
        record["latency_seconds"] = time.perf_counter() - started
        record["usage"] = jsonable(backend.usage())
        if record["status"] == "ok":
            record["grading"] = grade_case(question, record)
        return record

    with output.open("a", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, q, mode): (q, mode) for q, mode in tasks}
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                print(
                    f"{record['split']} {record['id']} {record['mode']} "
                    f"{record['status']} rounds={record.get('rounds')} "
                    f"latency={record['latency_seconds']:.1f}s",
                    flush=True,
                )

    metadata = {
        "split": args.split,
        "modes": args.modes,
        "workers": args.workers,
        "questions": len(questions),
        "tasks_scheduled": len(tasks),
        "question_sha256": sha256(questions_path),
        "evidence_sha256": sha256(corpus_path),
        "params": params,
        "elapsed_seconds": time.perf_counter() - start_all,
        "python": sys.version,
        "chat_model": os.getenv("CHAT_MODEL", "qwen3.6-35b-a3b"),
        "chat_replay_cache": str(cache_path) if not args.disable_replay_cache else None,
        "replay_backend": not args.disable_replay_cache,
    }
    mode_slug = "-".join(args.modes)
    metadata_path = output.with_suffix(output.suffix + f".{args.split}.{mode_slug}.meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

