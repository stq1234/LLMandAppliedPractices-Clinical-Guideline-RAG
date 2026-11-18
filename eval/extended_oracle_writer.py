"""All-gold evidence oracle: hold supplied IDs fixed to separate writer and critic."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
def read_jsonl(p:Path): return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonable(x:Any):
    if hasattr(x,"model_dump"): return x.model_dump()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if hasattr(x,"tolist"): return x.tolist()
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    return str(x)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,default=ROOT)
    ap.add_argument("--questions",type=Path,required=True)
    ap.add_argument("--split",default="test")
    ap.add_argument("--ids",nargs="*")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    if not 1<=args.workers<=2: ap.error("--workers must be 1 or 2")
    project=args.project.resolve(); os.chdir(project); sys.path.insert(0,str(project))
    import improved_clinical as engine
    from eval.grading import grade_case
    from eval.replay_backend import ReplayBackend
    qpath=args.questions if args.questions.is_absolute() else project/args.questions
    questions=read_jsonl(qpath)
    if args.split!="all": questions=[q for q in questions if str(q.get("split","all"))==args.split]
    questions=[q for q in questions if not q.get("scope_out",False) and not q.get("unanswerable",False)]
    if args.ids is not None:
        wanted=set(args.ids); questions=[q for q in questions if str(q.get("id")) in wanted]
        missing=wanted-{str(q.get("id")) for q in questions}
        if missing: raise ValueError(f"Unknown answerable IDs: {sorted(missing)}")
    if not questions: raise ValueError("No answerable questions selected")
    corpus_path=project/"data"/"evidence.jsonl"
    corpus_ids={str(x["id"]) for x in read_jsonl(corpus_path)}
    for q in questions:
        gold=q.get("gold",q.get("evidence_gold",[]))
        if not gold or not set(map(str,gold))<=corpus_ids: raise ValueError(f"Missing/invalid all-gold IDs in {q.get('id')}")
    module_path=Path(engine.__file__).resolve(); engine_sha=sha256(module_path)
    backend=ReplayBackend(cache_path=str(project/"results"/"extended_chat_cache.sqlite3"),
        engine_version=f"extended:{getattr(engine,'VERSION','unknown')}:{engine_sha}")
    modes=("allgold_critic","allgold_nocrit")
    systems={m:engine.System(backend,corpus_path="data/evidence.jsonl",index_dir="index") for m in modes}
    output=args.output if args.output.is_absolute() else project/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if output.exists():
        for r in read_jsonl(output):
            if r.get("status")=="ok": completed.add((str(r.get("id")),str(r.get("mode"))))
    tasks=[(q,m) for q in questions for m in modes if (str(q["id"]),m) not in completed]
    def run(q,m):
        gold=q.get("gold",q.get("evidence_gold",[])); ids=list(map(str,gold))
        grade_q=dict(q); grade_q["gold"]=ids
        engine_mode="reuse" if m=="allgold_critic" else "adaptive_nocrit"
        params=None if m=="allgold_critic" else {"force_action":"reuse"}
        backend.reset_usage(); t=time.perf_counter()
        rec={"id":str(q["id"]),"split":q.get("split",args.split),"family":q.get("family",q.get("challenge_family")),
            "mode":m,"oracle_condition":"all_gold_supplied_fixed_writer_critic_separation",
            "query":str(q["query"]),"original_supplied":q.get("supplied"),"supplied":ids,"gold":ids,
            "expected_groups":q.get("expected_groups"),"unanswerable":False,"scope_out":False,
            "complexity":q.get("complexity"),"params":params}
        try:
            response=jsonable(systems[m].respond(str(q["query"]),supplied=ids,mode=engine_mode,params=params))
            if not isinstance(response,dict): raise TypeError("System.respond must return a mapping")
            rec.update(response); rec["status"]="ok"; rec["grading"]=grade_case(grade_q,rec)
        except Exception as exc: rec.update(status="error",error_type=type(exc).__name__,error=str(exc)[:1200])
        rec["latency_seconds"]=time.perf_counter()-t; rec["usage"]=jsonable(backend.usage())
        return rec
    with output.open("a",encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run,q,m):(q,m) for q,m in tasks}
            for fut in concurrent.futures.as_completed(futures):
                r=fut.result(); out.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n"); out.flush()
                print(f"oracle {r['id']} {r['mode']} {r['status']} abstain={r.get('abstain')} rounds={r.get('rounds')}",flush=True)
    meta={"questions_file":str(qpath),"question_sha256":sha256(qpath),"evidence_sha256":sha256(corpus_path),
        "engine_sha256":engine_sha,"split":args.split,"selected_questions":len(questions),"modes":list(modes),"workers":args.workers,
        "output":str(output),"interpretation":"gold IDs are deliberately supplied only in this oracle condition; this does not estimate normal retrieval performance"}
    output.with_suffix(output.suffix+".meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
