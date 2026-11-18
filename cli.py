"""Command-line entry point for baseline and improved clinical engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from clinical import VERSION as CLINICAL_VERSION
from clinical import System as ClinicalSystem
from clinical import redact_query_for_log


def _read_params(path: Optional[str]) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("params file must contain a JSON object")
    return value


def _append_audit(path: str, query: str, mode: str, engine: str, version: str,
                  result: dict, elapsed_ms: float, signature: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_query = redact_query_for_log(query)
    event = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "system_version": version,
        "engine": engine,
        "embedding_signature": signature,
        "query": safe_query,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "mode": mode,
        "initial_action": result.get("initial_action"),
        "rounds": result.get("rounds"),
        "citations": result.get("citations", []),
        "sources": result.get("sources", []),
        "abstain": result.get("abstain"),
        "partial": result.get("partial", False),
        "elapsed_ms": round(elapsed_ms, 3),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evidence-grounded Q&A for general adult blood-pressure measurement and diagnosis."
    )
    parser.add_argument("--query", help="Non-personal question; if omitted, read stdin.")
    parser.add_argument("--mode", choices=("reuse", "single", "multi", "adaptive", "adaptive_nocrit"), default="adaptive")
    parser.add_argument("--engine", choices=("clinical", "improved"), default="clinical",
                        help="Inference implementation; the frozen clinical engine remains the default.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="For the improved engine only, answer supported requested parts and label remaining gaps.")
    parser.add_argument("--supplied", help="Comma-separated IDs already loaded from the approved evidence corpus.")
    parser.add_argument("--params", help="Optional JSON file with routing, retrieval, and response-policy overrides.")
    parser.add_argument("--corpus", default="data/evidence.jsonl")
    parser.add_argument("--index", default="index")
    parser.add_argument("--audit-log", help="Optional local JSONL audit log; common direct identifiers are redacted.")
    parser.add_argument("--version", action="version",
                        version=f"clinical={CLINICAL_VERSION} improved=0.2.0")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    query = (args.query if args.query is not None else sys.stdin.read()).strip()
    if not query:
        parser.error("provide --query or nonempty stdin")
    if args.allow_partial and args.engine != "improved":
        parser.error("--allow-partial requires --engine improved")
    supplied = [value.strip() for value in args.supplied.split(",") if value.strip()] if args.supplied else None
    try:
        params = _read_params(args.params)
        if args.engine == "improved":
            from improved_clinical import System, VERSION

            if args.allow_partial:
                params["allow_partial"] = True
        else:
            System, VERSION = ClinicalSystem, CLINICAL_VERSION
            if params.get("allow_partial"):
                raise ValueError("allow_partial is available only with --engine improved")

        from model_backend import Backend

        backend = Backend()
        system = System(backend, corpus_path=args.corpus, index_dir=args.index)
        started = time.perf_counter()
        result = system.respond(query, supplied=supplied, mode=args.mode, params=params)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if args.audit_log:
            _append_audit(
                args.audit_log,
                query,
                args.mode,
                args.engine,
                VERSION,
                result,
                elapsed_ms,
                system.embedding_signature,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        version = CLINICAL_VERSION if args.engine == "clinical" else "0.2.0"
        print(json.dumps({"error": str(exc), "system_version": version, "engine": args.engine},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
