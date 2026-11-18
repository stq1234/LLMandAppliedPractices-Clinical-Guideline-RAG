"""Fixed-evidence writer-only arm, bypassing router/retrieval/critic/validator."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
def readj(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonable(x:Any):
 if hasattr(x,"model_dump"):return x.model_dump()
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 if hasattr(x,"tolist"):return x.tolist()
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return str(x)
def render_claims(claims):
 lines=[];citation_ids=[]
 for claim in claims:
  if not isinstance(claim,dict):continue
  text=str(claim.get("text","")).strip()
  raw=claim.get("citations") or []
  ids=[]
  for c in raw:
   if isinstance(c,dict) and c.get("id") is not None:ids.append(str(c["id"]))
  ids=list(dict.fromkeys(ids));citation_ids.extend(ids)
  if text:lines.append(text+(" "+" ".join(f"[{x}]" for x in ids) if ids else ""))
 return "\n".join(lines),list(dict.fromkeys(citation_ids))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project",type=Path,default=ROOT);ap.add_argument("--questions",type=Path,required=True);ap.add_argument("--split",default="test");ap.add_argument("--ids",nargs="*");ap.add_argument("--output",type=Path,required=True);ap.add_argument("--workers",type=int,default=2);a=ap.parse_args()
 if not 1<=a.workers<=2:ap.error("--workers must be 1 or 2")
 proj=a.project.resolve();os.chdir(proj);sys.path.insert(0,str(proj));import improved_clinical as engine
 from eval.replay_backend import ReplayBackend
 from eval.grading import grade_case
 qp=a.questions if a.questions.is_absolute() else proj/a.questions;questions=readj(qp)
 if a.split!="all":questions=[q for q in questions if str(q.get("split","all"))==a.split]
 questions=[q for q in questions if not q.get("scope_out",False) and not q.get("unanswerable",False)]
 if a.ids is not None:
  wanted=set(a.ids);questions=[q for q in questions if str(q.get("id")) in wanted]
  missing=wanted-{str(q["id"]) for q in questions}
  if missing:raise ValueError(f"Unknown answerable IDs: {sorted(missing)}")
 if not questions:raise ValueError("No answerable questions selected")
 cp=proj/"data"/"evidence.jsonl";corpus=readj(cp);corpus_ids={str(r["id"]) for r in corpus}
 for q in questions:
  g=q.get("gold",q.get("evidence_gold",[]))
  if not g or not set(map(str,g))<=corpus_ids:raise ValueError(f"Missing gold evidence in corpus for {q.get('id')}")
 backend=ReplayBackend(cache_path=str(proj/"results"/"extended_chat_cache.sqlite3"),engine_version=f"extended:{getattr(engine,'VERSION','unknown')}:{sha(Path(engine.__file__).resolve())}")
 system=engine.System(backend,corpus_path="data/evidence.jsonl",index_dir="index")
 out=a.output if a.output.is_absolute() else proj/a.output;out.parent.mkdir(parents=True,exist_ok=True)
 completed=set()
 if out.exists():
  for r in readj(out):
   if r.get("status")=="ok":completed.add(str(r.get("id")))
 todo=[q for q in questions if str(q["id"]) not in completed]
 def run(q):
  gold=list(map(str,q.get("gold",q.get("evidence_gold",[]))));query=str(q["query"]);grade_q=dict(q);grade_q["gold"]=gold
  backend.reset_usage();t=time.perf_counter();rec={"id":str(q["id"]),"split":q.get("split",a.split),"mode":"allgold_writer_only","oracle_condition":"all_gold_fixed_writer_only_no_critic_no_retrieval","query":query,"family":q.get("family",q.get("challenge_family")),"challenge_family":q.get("challenge_family"),"complexity":q.get("complexity"),"supplied":gold,"gold":gold,"expected_groups":q.get("expected_groups"),"expected":q.get("expected"),"unanswerable":False,"scope_out":False,"initial_need":q.get("initial_need"),"rounds":0,"first_evidence":gold,"evidence":gold,"initial_action":"reuse","engine":"improved","engine_version":getattr(engine,"VERSION","unknown"),"semantic_critic":False}
  try:
   facets,fallback=system._plan(query)
   records=system._resolve_supplied(gold)
   if any(not system._population_compatible(query,r["population"],r.get("population_scope")) for r in records):raise ValueError("Gold records are population-incompatible with query")
   claims=jsonable(system._write_claims(query,facets,records));answer,citations=render_claims(claims)
   rec.update(answer=answer,claims=claims,citations=citations,abstain=not bool(answer.strip()),sources=system._sources_for_ids(citations,records),trace=[{"stage":"oracle_input","action":"gold_fixed","retrieved":gold},{"stage":"planner","action":"plan","facets":facets,"fallback":fallback},{"stage":"writer","action":"writer_only","retrieved":citations}],diagnostics={"mode":"allgold_writer_only","semantic_critic":False,"router_bypassed":True,"retriever_bypassed":True,"claim_validation_bypassed":True,"plan_facets":facets,"plan_fallback":fallback,"writer_raw_claims":claims})
   rec["status"]="ok";rec["grading"]=grade_case(grade_q,rec)
  except Exception as exc:rec.update(status="error",error_type=type(exc).__name__,error=str(exc)[:1200])
  rec["latency_seconds"]=time.perf_counter()-t;rec["usage"]=jsonable(backend.usage());return rec
 with out.open("a",encoding="utf-8") as f:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
   futures={pool.submit(run,q):q for q in todo}
   for future in concurrent.futures.as_completed(futures):
    r=future.result();f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n");f.flush();print(f"writer-only {r['id']} {r['status']} abstain={r.get('abstain')}",flush=True)
 meta={"questions_file":str(qp),"question_sha256":sha(qp),"evidence_sha256":sha(cp),"engine_sha256":sha(Path(engine.__file__).resolve()),"split":a.split,"n_answerable":len(questions),"scheduled":len(todo),"workers":a.workers,"output":str(out),"condition":"same frozen System._plan and System._write_claims prompts on all and only gold IDs; no route, retrieval, critic, semantic validation or claim-anchor validation; gold is used only in this oracle arm"}
 out.with_suffix(out.suffix+".meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
