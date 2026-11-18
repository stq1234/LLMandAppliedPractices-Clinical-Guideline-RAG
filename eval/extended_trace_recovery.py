"""Summarize first/follow-up evidence coverage from new engine traces."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def readj(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def ids(value):
 if isinstance(value,dict):
  for k in ("ids","evidence_ids","items","citations"):
   if k in value:return ids(value[k])
  for k in ("id","evidence_id","document_id","source_id"):
   if value.get(k) is not None:return [str(value[k])]
  return []
 if isinstance(value,str):return [value]
 if isinstance(value,list):
  out=[]
  for x in value:out.extend(ids(x))
  return out
 return []
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--questions",type=Path,required=True);ap.add_argument("--evaluation",type=Path,required=True);ap.add_argument("--split",required=True);ap.add_argument("--mode",choices=("single","multi","adaptive","adaptive_nocrit"),required=True);ap.add_argument("--output-prefix",type=Path,required=True);a=ap.parse_args()
 qpath=a.questions if a.questions.is_absolute() else ROOT/a.questions;epath=a.evaluation if a.evaluation.is_absolute() else ROOT/a.evaluation
 qmap={str(q["id"]):q for q in readj(qpath)};logs=[r for r in readj(epath) if str(r.get("split"))==a.split and r.get("mode")==a.mode and r.get("status")=="ok"]
 rows=[];stage_rows=[]
 for r in logs:
  q=qmap[str(r["id"])];gold=set(map(str,q.get("gold",q.get("evidence_gold",[])) or []));g=r.get("grading") or {}
  if not g.get("answerable") or not gold:continue
  first=set(ids(r.get("first_evidence")));final=set(ids(r.get("evidence")));current=set(first);steps=[]
  for ix,st in enumerate([s for s in (r.get("trace") or []) if isinstance(s,dict) and s.get("stage")=="retrieval" and s.get("action")=="supplement"],1):
   returned=set(ids(st.get("retrieved")));new=returned-current;new_gold=sorted(new&gold);current.update(returned)
   step={"id":str(r["id"]),"family":q.get("family",q.get("challenge_family")),"mode":a.mode,"stage":ix,"reason":st.get("reason"),"queries":st.get("queries") or [],"retrieved_ids":sorted(returned),"new_gold_ids":new_gold,"gold_ids_found_after_stage":sorted(current&gold),"gold_recall_after_stage":len(current&gold)/len(gold),"gold_complete_after_stage":gold<=current}
   steps.append(step);stage_rows.append(step)
  rows.append({"id":str(r["id"]),"family":q.get("family",q.get("challenge_family")),"complexity":q.get("complexity"),"gold_ids":sorted(gold),"first_ids":sorted(first),"first_missing_gold":sorted(gold-first),"first_gold_recall":len(first&gold)/len(gold),"first_gold_complete":gold<=first,"supplement_count":len(steps),"supplement_steps":steps,"final_ids":sorted(final),"final_gold_recall":len(final&gold)/len(gold),"final_gold_complete":gold<=final,"supplement_added_gold":bool((final-first)&gold),"supplement_rescued_supported_answer":bool(g.get("rescued_supported_keypoint_match")),"expected_match":g.get("expected_match"),"abstain":g.get("abstain"),"citation_ids":g.get("citation_ids"),"citation_coverage":g.get("citation_coverage"),"citations_complete":g.get("citations_complete"),"rounds":r.get("rounds")})
 n=len(rows);supp=[r for r in rows if r["supplement_count"]>0];summary={"split":a.split,"mode":a.mode,"n_answerable_with_gold":n,"n_first_gold_complete":sum(r["first_gold_complete"] for r in rows),"first_gold_complete_rate":sum(r["first_gold_complete"] for r in rows)/n if n else None,"n_with_supplement":len(supp),"n_supplement_added_gold":sum(r["supplement_added_gold"] for r in supp),"n_supplement_rescued_supported_answer":sum(r["supplement_rescued_supported_answer"] for r in supp),"n_supplement_nonproductive_gold":sum(not r["supplement_added_gold"] for r in supp),"mean_first_gold_recall":sum(r["first_gold_recall"] for r in rows)/n if n else None,"mean_final_gold_recall":sum(r["final_gold_recall"] for r in rows)/n if n else None,"n_supplement_stages":len(stage_rows),"family":{}}
 for fam in sorted({str(r["family"] or "unknown") for r in rows}):
  rr=[r for r in rows if str(r["family"] or "unknown")==fam];ss=[r for r in rr if r["supplement_count"]]
  summary["family"][fam]={"n":len(rr),"first_complete_n":sum(r["first_gold_complete"] for r in rr),"supplement_n":len(ss),"supplement_added_gold_n":sum(r["supplement_added_gold"] for r in ss),"rescued_supported_answer_n":sum(r["supplement_rescued_supported_answer"] for r in ss)}
 prefix=a.output_prefix if a.output_prefix.is_absolute() else ROOT/a.output_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
 prefix.with_suffix(".json").write_text(json.dumps({"summary":summary,"questions":rows,"stages":stage_rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 for suffix,data,fields in [("_questions.csv",rows,["id","family","complexity","gold_ids","first_ids","first_missing_gold","first_gold_recall","first_gold_complete","supplement_count","supplement_added_gold","supplement_rescued_supported_answer","final_ids","final_gold_recall","final_gold_complete","expected_match","abstain","citation_ids","citation_coverage","citations_complete","rounds"]),("_stages.csv",stage_rows,["id","family","mode","stage","reason","queries","retrieved_ids","new_gold_ids","gold_ids_found_after_stage","gold_recall_after_stage","gold_complete_after_stage"] )]:
  with prefix.with_name(prefix.name+suffix).open("w",encoding="utf-8-sig",newline="") as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for r in data:w.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k) for k in fields})
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
