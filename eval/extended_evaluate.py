"""Evaluation harness for the independent improved 7.3 engine.

Gold labels are copied only into the offline result/grader; System.respond sees
only the query and user-supplied evidence IDs.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        obj=json.loads(line)
        if not isinstance(obj,dict): raise ValueError(f"Expected object at {path}:{n}")
        rows.append(obj)
    return rows

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def jsonable(x: Any) -> Any:
    if hasattr(x,"model_dump"): return x.model_dump()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if hasattr(x,"tolist"): return x.tolist()
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    return str(x)

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,default=ROOT)
    ap.add_argument("--questions",type=Path,required=True,help="JSONL path, absolute or relative to project")
    ap.add_argument("--split",default="all",help="split field, or all")
    ap.add_argument("--modes",nargs="+",default=["single","multi","adaptive","adaptive_nocrit"])
    ap.add_argument("--ids",nargs="*")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--params",type=Path)
    ap.add_argument("--allow-partial",action="store_true")
    args=ap.parse_args()
    if not 1 <= args.workers <= 2: ap.error("--workers must be 1 or 2")
    project=args.project.resolve(); os.chdir(project); sys.path.insert(0,str(project))
    import improved_clinical as engine
    from eval.grading import grade_case
    from eval.replay_backend import ReplayBackend
    module_path=Path(engine.__file__).resolve()
    engine_sha=sha256(module_path)
    engine_version=f"extended:{getattr(engine,'VERSION','unknown')}:{engine_sha}"
    qpath=args.questions if args.questions.is_absolute() else project/args.questions
    questions=read_jsonl(qpath)
    if args.split != "all": questions=[q for q in questions if str(q.get("split","all"))==args.split]
    if args.ids is not None:
        wanted=set(args.ids); questions=[q for q in questions if str(q.get("id")) in wanted]
        missing=wanted-{str(q.get("id")) for q in questions}
        if missing: raise ValueError(f"Unknown IDs: {sorted(missing)}")
    if not questions: raise ValueError("No questions selected")
    if len({str(q.get("id")) for q in questions}) != len(questions): raise ValueError("IDs must be unique")
    corpus_path=project/"data"/"evidence.jsonl"
    corpus=read_jsonl(corpus_path); corpus_ids={str(r.get("id")) for r in corpus}
    params=json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
    if args.allow_partial: params["allow_partial"]=True
    backend=ReplayBackend(cache_path=str(project/"results"/"extended_chat_cache.sqlite3"),engine_version=engine_version)
    systems={m:engine.System(backend,corpus_path="data/evidence.jsonl",index_dir="index") for m in args.modes}
    output=args.output if args.output.is_absolute() else project/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if output.exists():
        for row in read_jsonl(output):
            if row.get("status")=="ok": completed.add((str(row.get("id")),str(row.get("mode"))))
    tasks=[(q,m) for q in questions for m in args.modes if (str(q["id"]),m) not in completed]
    def run(q: dict[str,Any],mode: str) -> dict[str,Any]:
        supplied=q.get("supplied")
        gold=q.get("gold",q.get("evidence_gold",[]))
        grade_question=dict(q); grade_question["gold"]=gold
        record={"id":str(q["id"]),"split":q.get("split",args.split),"mode":mode,
            "family":q.get("family",q.get("challenge_family")),"query":str(q["query"]),
            "supplied":supplied,"gold":gold,"gold_in_corpus":set(map(str,gold or []))<=corpus_ids,
            "gold_missing_from_corpus":sorted(set(map(str,gold or []))-corpus_ids),
            "expected_groups":q.get("expected_groups"),"expected":q.get("expected"),
            "unanswerable":bool(q.get("unanswerable",False)),"scope_out":bool(q.get("scope_out",False)),
            "initial_need":q.get("initial_need"),"complexity":q.get("complexity"),
            "source_ids":q.get("source_ids"),"challenge_family":q.get("challenge_family"),"params":params or None}
        backend.reset_usage(); t=time.perf_counter()
        try:
            call_params=dict(params) if mode in ("adaptive","adaptive_nocrit") and params else None
            response=jsonable(systems[mode].respond(str(q["query"]),supplied=supplied,mode=mode,params=call_params))
            if not isinstance(response,dict): raise TypeError("System.respond must return a mapping")
            record.update(response)
            trace=response.get("trace") or []
            router=[s for s in trace if isinstance(s,dict) and s.get("stage")=="router"]
            supplements=[s for s in trace if isinstance(s,dict) and s.get("stage")=="retrieval" and s.get("action")=="supplement"]
            record["effective_route"]=router[-1].get("action") if router else response.get("initial_action")
            record["adaptive_fallback"]=any(s.get("reason")=="supplied_evidence_gap_triggered_retrieval_fallback" for s in router)
            record["supplement_attempted"]=bool(supplements)
            record["supplement_reason"]=supplements[-1].get("reason") if supplements else None
            record["initial_retrieval_executed"]=int(response.get("rounds",0) or 0)>0
            record["initial_retrieval_trace_seen"]=any(isinstance(s,dict) and s.get("stage")=="retrieval" and s.get("action")=="initial" for s in trace)
            record["status"]="ok"
            record["grading"]=grade_case(grade_question,record)
        except Exception as exc:
            record.update(status="error",error_type=type(exc).__name__,error=str(exc)[:1200])
        record["latency_seconds"]=time.perf_counter()-t
        record["usage"]=jsonable(backend.usage())
        return record
    with output.open("a",encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map={pool.submit(run,q,m):(q,m) for q,m in tasks}
            for fut in concurrent.futures.as_completed(future_map):
                rec=fut.result(); out.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n"); out.flush()
                print(f"{rec['id']} {rec['mode']} {rec['status']} rounds={rec.get('rounds')} wall={rec['latency_seconds']:.1f}s",flush=True)
    meta={"questions_file":str(qpath),"question_sha256":sha256(qpath),"evidence_sha256":sha256(corpus_path),
        "engine_file":str(module_path),"engine_sha256":engine_sha,"engine_version":engine_version,
        "split":args.split,"modes":args.modes,"workers":args.workers,"selected_questions":len(questions),
        "scheduled":len(tasks),"params":params,"output":str(output)}
    output.with_suffix(output.suffix+".meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=="__main__": main()



