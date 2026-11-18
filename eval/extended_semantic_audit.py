"""Offline keypoint audit for numeric, authority, and directional answer content."""
from __future__ import annotations
import argparse,csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
AUTH=re.compile(r"(?i)NICE(?:\s+NG\s*136)?|AHA\s*/\s*ACC|ACC\s*/\s*AHA|ISH(?:\s+2026)?|USPSTF|ESC|JAHA|Deragon")
NUM=re.compile(r"(?<![\w.])(?:≥|≤|>=|<=|>|<|=)?\s*[+-]?\d+(?:\.\d+)?(?:\s*(?:–|—|至|到|~|～|-)\s*[+-]?\d+(?:\.\d+)?)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?(?:\s*(?:–|—|至|到|~|～|-)\s*[+-]?\d+(?:\.\d+)?)?)?\s*(?:mmHg|mmol/L|mg/dL|kg/m²|kg/m2|mm|cm|kg|mg|μg|mcg|g|bpm|天|日|周|星期|月|年|小时|分钟|%|次)?",re.I)
DIR=re.compile(r"(?:不低于|不超过|至少|至多|最多|高于|低于|大于|小于|超过|应当|建议|不建议|避免|优先|排除|不得|必须|可考虑|可提供|诊断为|归为|定义为|normal|elevated|stage\s*\d|hypertension|recommend(?:ed)?|should|must|avoid|exclude|at least|at most|greater than|less than)",re.I)
def readj(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def vals(x):
 if isinstance(x,str): return [x]
 if isinstance(x,list): return [str(v) for v in x]
 return []
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,default=ROOT); ap.add_argument("--questions",type=Path,required=True); ap.add_argument("--evaluation",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
 proj=args.project.resolve(); sys.path.insert(0,str(proj)); from eval.grading import _candidate_match, _strip_provenance
 qp=args.questions if args.questions.is_absolute() else proj/args.questions; ep=args.evaluation if args.evaluation.is_absolute() else proj/args.evaluation; op=args.output if args.output.is_absolute() else proj/args.output
 qs={str(x["id"]):x for x in readj(qp)}; corpus={str(x["id"]):x for x in readj(proj/"data"/"evidence.jsonl")}; rows=[]
 for result in readj(ep):
  q=qs.get(str(result.get("id")))
  if q is None: continue
  answer=str(result.get("answer") or "")
  analysis_answer=_strip_provenance(answer,result)
  groups=q.get("expected_groups") or []
  group_checks=[]
  for i,g in enumerate(groups,1):
   options=vals(g); group_checks.append({"group":i,"acceptable_phrases":options,"literal_match":any(_candidate_match(x,analysis_answer) for x in options)})
  gold=q.get("gold",q.get("evidence_gold",[])) or []
  gold_docs=[corpus[g] for g in map(str,gold) if str(g) in corpus]
  searchable=" ".join([str(q.get("query") or ""),json.dumps(groups,ensure_ascii=False),analysis_answer])
  rows.append({"id":str(q["id"]),"split":q.get("split"),"mode":result.get("mode"),"family":q.get("family",q.get("challenge_family")),
   "status":result.get("status"),"abstain":result.get("abstain"),"answerable":not bool(q.get("scope_out",False) or q.get("unanswerable",False)),"scope_out":bool(q.get("scope_out",False)),"unanswerable":bool(q.get("unanswerable",False)),"expected_match":(result.get("grading") or {}).get("expected_match"),
   "query":q.get("query"),"answer":answer,"gold_ids":gold,"gold_sources":[d.get("source_id") for d in gold_docs],
   "gold_titles_locators":[{"title":d.get("title"),"locator":d.get("locator"),"source_id":d.get("source_id"),"authority":d.get("authority"),"population":d.get("population"),"text":d.get("text")} for d in gold_docs],
   "acceptable_point_checks":group_checks,"observed_numeric_expressions":sorted(set(m.group(0).strip() for m in NUM.finditer(analysis_answer))),
   "query_or_key_numeric_expressions":sorted(set(m.group(0).strip() for m in NUM.finditer(searchable))),
   "answer_authority_mentions":sorted(set(m.group(0) for m in AUTH.finditer(analysis_answer))),
   "query_key_authority_mentions":sorted(set(m.group(0) for m in AUTH.finditer(searchable))),
   "answer_direction_terms":sorted(set(m.group(0) for m in DIR.finditer(analysis_answer))),
   "query_key_direction_terms":sorted(set(m.group(0) for m in DIR.finditer(searchable))),
   "citation_ids":(result.get("grading") or {}).get("citation_ids"),"evidence_ids":(result.get("grading") or {}).get("final_evidence_ids")})
 op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps({"scope":"lexical screening, not clinical semantic adjudication","limitations":"Literal expected-phrase matching can miss valid paraphrases and can pass a phrase without checking negation, attribution to the correct authority, threshold direction, units, or interactions. Numeric expressions and authority/direction terms are extraction aids; human verification must compare each answer against its frozen gold evidence. Gold labels are unchanged.","n":len(rows),"rows":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 csv_path=op.with_suffix(".csv"); fields=["id","split","mode","family","status","answerable","scope_out","unanswerable","abstain","expected_match","query","answer","gold_ids","gold_sources","gold_titles_locators","acceptable_point_checks","observed_numeric_expressions","query_or_key_numeric_expressions","answer_authority_mentions","query_key_authority_mentions","answer_direction_terms","query_key_direction_terms","citation_ids","evidence_ids"]
 with csv_path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:(json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(list,dict)) else row[k]) for k in fields})
 print(f"audited_rows={len(rows)} json={op} csv={csv_path}")
if __name__=="__main__":main()



