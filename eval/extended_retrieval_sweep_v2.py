"""No-generation initial/facet retrieval sweep with query-visible authority filters."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from clinical import System,_tokens
from model_backend import Backend
DEPTHS=(3,5,8)
SOURCE_PATTERNS=((r"NICE|NG136",{"NICE_NG136"}),(r"AHA|ACC",{"AHA_ACC_2025"}),(r"ISH",{"ISH_GLOBAL_2026"}),(r"USPSTF",{"USPSTF_SCREEN_2021","USPSTF_REVIEW_2021"}),(r"JAHA|DERAGON",{"JAHA_AOBP_REVIEW_2025"}))
def readj(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_ids(query):
 out=set()
 for pat,ids in SOURCE_PATTERNS:
  if re.search(pat,query,re.I):out.update(ids)
 return sorted(out)
def collect_facets(record):
 facets=[]
 for st in record.get("trace") or []:
  if not isinstance(st,dict):continue
  if st.get("stage")=="retrieval":
   qs=st.get("queries") or st.get("subqueries") or []
   if isinstance(qs,str):qs=[qs]
   # Only subneed/follow-up-stage queries are included; an initial main query is not a facet.
   action=str(st.get("action","")).lower()
   if action in {"supplement","facet","subquery","followup","follow_up"} or st.get("purpose")=="facet":
    facets.extend(str(q).strip() for q in qs if str(q).strip())
   elif qs and st.get("round",0)>0:
    facets.extend(str(q).strip() for q in qs if str(q).strip())
  if st.get("stage")=="planner":
   qs=st.get("facets") or st.get("subqueries") or st.get("queries") or []
   if isinstance(qs,str):qs=[qs]
   facets.extend(str(q).strip() for q in qs if str(q).strip())
 return list(dict.fromkeys(facets))
def eligible_idxs(system,query,source_filter=None):
 return [i for i,r in enumerate(system.records) if system._population_compatible(query,r["population"],r.get("population_scope")) and (source_filter is None or r["source_id"] in source_filter)]
def rankings(system,query,vec,source_filter=None):
 ids=eligible_idxs(system,query,source_filter)
 if not ids:return {m:[] for m in ("lexical","dense","hybrid")},0
 tok=_tokens(query); lex=system._lexical_bm25(tok,ids); dense=np.asarray(system.vectors[ids]@vec,dtype=np.float32)
 lexical=[i for i in sorted(ids,key=lambda i:(-lex.get(i,0.0),system.records[i]["id"]))]
 dense_order=[i for _,i in sorted(((-float(s),i) for i,s in zip(ids,dense)),key=lambda x:(x[0],system.records[x[1]]["id"]))]
 scores={i:float(dense[pos])+0.15*float(lex.get(i,0.0))+(0.025 if system.records[i]["kind"]=="guideline" else 0.0) for pos,i in enumerate(ids)}
 hybrid=sorted(ids,key=lambda i:(-scores[i],system.records[i]["id"]))
 return {"lexical":[system.records[i]["id"] for i in lexical],"dense":[system.records[i]["id"] for i in dense_order],"hybrid":[system.records[i]["id"] for i in hybrid]},len(ids)
def metric(rows,method,k):
 if not rows:return {"n":0,"complete_n":0,"complete_rate":None,"macro_gold_recall":None}
 rec=[]; complete=0
 for r in rows:
  gold=set(r["gold"]); got=set(r["rankings"][method][:k]); rec.append(len(gold&got)/len(gold) if gold else 1.0); complete+=bool(gold) and gold<=got
 return {"n":len(rows),"complete_n":int(complete),"complete_rate":complete/len(rows),"macro_gold_recall":float(np.mean(rec))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--questions",type=Path,required=True);ap.add_argument("--trace-file",type=Path);ap.add_argument("--trace-mode",choices=("single","multi","adaptive","adaptive_nocrit"),default="multi");ap.add_argument("--split",default="all");ap.add_argument("--output-prefix",type=Path,required=True);args=ap.parse_args()
 qpath=args.questions if args.questions.is_absolute() else ROOT/args.questions
 tracepath=args.trace_file if args.trace_file and args.trace_file.is_absolute() else ROOT/args.trace_file if args.trace_file else None
 questions=readj(qpath)
 if args.split!="all":questions=[q for q in questions if str(q.get("split","all"))==args.split]
 selected=[q for q in questions if not q.get("scope_out",False) and not q.get("unanswerable",False) and (q.get("gold") or q.get("evidence_gold"))]
 trace_by_id={}
 if tracepath and tracepath.exists():
  for r in readj(tracepath):
   if r.get("mode")==args.trace_mode and r.get("status")=="ok":trace_by_id[str(r["id"])]=r
 corpus_path=ROOT/"data"/"evidence.jsonl"; backend=Backend();system=System(backend,corpus_path=corpus_path,index_dir=ROOT/"index")
 facets={str(q["id"]):collect_facets(trace_by_id.get(str(q["id"]),{})) for q in selected}
 queries=[]
 for q in selected:queries.append(str(q["query"]));queries.extend(facets[str(q["id"])])
 unique=list(dict.fromkeys(x for x in queries if x));vecs=backend.encode(unique,query=True);vec_by={q:vecs[i] for i,q in enumerate(unique)}
 cases=[];facet_rows=[]
 for q in selected:
  qid=str(q["id"]);query=str(q["query"]);gold=list(map(str,q.get("gold",q.get("evidence_gold",[])) or []));sf=source_ids(query);sfset=set(sf) if sf else None
  unfiltered,cnt=rankings(system,query,vec_by[query]);filtered,fcnt=rankings(system,query,vec_by[query],sfset)
  allrank={**unfiltered,"named_source_filter":filtered["hybrid"]};counts={"unfiltered":cnt,"named_source_filter":fcnt}
  fs=[];union={"unfiltered":defaultdict(set),"named_source_filter":defaultdict(set)}
  for i,fq in enumerate(facets[qid],1):
   fr,fn=rankings(system,fq,vec_by[fq]);ff,ffn=rankings(system,fq,vec_by[fq],sfset)
   details={"index":i,"query":fq,"candidate_count":fn,"named_source_candidate_count":ffn,
    "gold_ranks_unfiltered":{g:(fr["hybrid"].index(g)+1 if g in fr["hybrid"] else None) for g in gold},
    "gold_ranks_named_source":{g:(ff["hybrid"].index(g)+1 if g in ff["hybrid"] else None) for g in gold},
    "top8_unfiltered":fr["hybrid"][:8],"top8_named_source":ff["hybrid"][:8]}
   fs.append(details)
   for k in DEPTHS:
    union["unfiltered"][k].update(fr["hybrid"][:k]);union["named_source_filter"][k].update(ff["hybrid"][:k])
  fsum={}
  for k in DEPTHS:
   fsum[str(k)]={}
   for mode in ("unfiltered","named_source_filter"):
    hits=sorted(set(gold)&union[mode][k]);fsum[str(k)][mode]={"gold_ids":hits,"gold_recall":len(hits)/len(gold) if gold else 1.0,"complete":set(gold)<=union[mode][k]}
   facet_rows.append({"id":qid,"family":q.get("family",q.get("challenge_family")),"complexity":q.get("complexity"),"top_k":k,"facet_count":len(fs),"unfiltered_recall":fsum[str(k)]["unfiltered"]["gold_recall"],"unfiltered_complete":fsum[str(k)]["unfiltered"]["complete"],"filtered_recall":fsum[str(k)]["named_source_filter"]["gold_recall"],"filtered_complete":fsum[str(k)]["named_source_filter"]["complete"]})
  cases.append({"id":qid,"family":q.get("family",q.get("challenge_family")),"complexity":q.get("complexity"),"query":query,"gold":gold,"named_source_ids_from_query":sf,"source_filter_applied":bool(sf),"candidate_counts":counts,"rankings":allrank,"gold_ranks":{m:{g:(o.index(g)+1 if g in o else None) for g in gold} for m,o in allrank.items()},"facet_queries":facets[qid],"facet_by_facet":fs,"facet_union_summary":fsum})
 methods=("lexical","dense","hybrid","named_source_filter");summary={m:{str(k):metric(cases,m,k) for k in DEPTHS} for m in methods}
 facets_cases=[r for r in cases if r["facet_queries"]]
 facet_summary={}
 for k in DEPTHS:
  facet_summary[str(k)]={"n_selected":len(cases),"n_with_facets":len(facets_cases)}
  for mode in ("unfiltered","named_source_filter"):
   vals=[r["facet_union_summary"][str(k)][mode] for r in facets_cases]
   facet_summary[str(k)][mode]={"complete_n":sum(x["complete"] for x in vals),"complete_rate":(sum(x["complete"] for x in vals)/len(vals) if vals else None),"macro_gold_recall":(float(np.mean([x["gold_recall"] for x in vals])) if vals else None)}
 prefix=args.output_prefix if args.output_prefix.is_absolute() else ROOT/args.output_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
 result={"study":"No-generation retrieval candidate sweep","split":args.split,"question_count":len(selected),"question_sha256":sha(qpath),"evidence_sha256":sha(corpus_path),"trace_file":str(tracepath) if tracepath else None,"trace_mode":args.trace_mode,"trace_sha256":sha(tracepath) if tracepath and tracepath.exists() else None,"chat_calls":backend.usage().get("chat_calls",0),"query_vectors":backend.usage().get("query_vectors",0),"depths":DEPTHS,"methods":methods,"filter_rule":"Filter only to source IDs explicitly named in query; if none named, use all population-compatible evidence. Gold is used only for post-ranking scoring.","facet_denominator_note":"Facet union metrics use only answerable questions with a logged follow-up/facet query; n_with_facets is shown explicitly. Missing traces are not treated as zero recall.","initial_rank_summary":summary,"facet_union_summary":facet_summary,"cases":cases}
 out_json=prefix.with_suffix(".json");out_csv=prefix.with_name(prefix.name+"_cases.csv");facet_csv=prefix.with_name(prefix.name+"_facets.csv")
 out_json.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 with out_csv.open("w",encoding="utf-8-sig",newline="") as f:
  fields=["id","family","complexity","source_filter_applied","named_source_ids_from_query","candidate_counts","gold_ranks"]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in cases:w.writerow({k:(json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(dict,list)) else r[k]) for k in fields})
 with facet_csv.open("w",encoding="utf-8-sig",newline="") as f:
  fields=["id","family","complexity","top_k","facet_count","unfiltered_recall","unfiltered_complete","filtered_recall","filtered_complete"]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(facet_rows)
 print(json.dumps({"initial":summary,"facets":facet_summary,"n_questions":len(cases),"n_with_facets":len(facets_cases),"chat_calls":result["chat_calls"],"query_vectors":result["query_vectors"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()

