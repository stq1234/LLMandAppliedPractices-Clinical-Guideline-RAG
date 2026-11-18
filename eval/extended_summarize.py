"""Split- and family-aware summaries for extended 7.3 runs."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"eval"))
from evaluate import read_jsonl
from grading import grade_case
from summarize import summarize_group,paired_comparisons,supplement_diagnostics

def deduplicate(raw):
 attempts=defaultdict(list)
 for r in raw:attempts[(str(r.get("split")),str(r.get("id")),str(r.get("mode")))].append(r)
 out=[]
 for history in attempts.values():
  valid=[r for r in history if r.get("status")=="ok"]
  row=dict(valid[-1] if valid else history[-1]);row["_attempt_count"]=len(history);row["_error_attempt_count"]=sum(x.get("status")!="ok" for x in history);row["_recovered_after_error"]=bool(valid and any(x.get("status")!="ok" for x in history))
  if row.get("status")=="ok": row["grading"]=grade_case(row,row)
  out.append(row)
 return out

def enriched_summary(rows):
 s=summarize_group(rows)
 valid=[r for r in rows if r.get("status")=="ok"]
 s["split"]=rows[0].get("split") if rows else None
 s["operational_success_n"]=sum(bool((r.get("grading") or {}).get("operational_success")) for r in valid)
 s["operational_success_rate"]=s["operational_success_n"]/len(valid) if valid else None
 s["correct_refusal_n"]=sum(bool((r.get("grading") or {}).get("correct_refusal")) for r in valid)
 s["answered_and_supported_n"]=sum(bool((r.get("grading") or {}).get("supported_keypoint_match")) for r in valid)
 s["family"]=rows[0].get("family",rows[0].get("challenge_family")) if rows else None
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--outdir",type=Path,required=True);a=ap.parse_args()
 raw=read_jsonl(a.input);rows=deduplicate(raw);splits=sorted({str(r.get("split")) for r in rows});out=a.outdir;out.mkdir(parents=True,exist_ok=True)
 modes=[];families=[]
 for split in splits:
  for mode in sorted({str(r.get("mode")) for r in rows if str(r.get("split"))==split}):
   group=[r for r in rows if str(r.get("split"))==split and str(r.get("mode"))==mode]
   modes.append(enriched_summary(group))
   family_values=sorted({str(r.get("family",r.get("challenge_family")) or "unknown") for r in group})
   for fam in family_values:
    fg=[r for r in group if str(r.get("family",r.get("challenge_family")) or "unknown")==fam]
    families.append(enriched_summary(fg))
 pairs=[]
 for split in splits:
  split_rows=[r for r in rows if str(r.get("split"))==split]
  if len({r.get("mode") for r in split_rows})>1:pairs.extend(paired_comparisons(split_rows,split))
 supplements=[]
 for split in splits:
  for mode in sorted({str(r.get("mode")) for r in rows if str(r.get("split"))==split and str(r.get("mode")) in ("adaptive","adaptive_nocrit")}):
   supplements.append(supplement_diagnostics(rows,split,mode))
 detail={"modes":modes,"families":families,"paired_comparisons":pairs,"supplement_diagnostics":supplements,"input":str(a.input)}
 (out/"summary.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 for name,data in (("summary.csv",modes),("family_summary.csv",families),("paired_gains_losses.csv",pairs),("supplement_diagnostics.csv",supplements)):
  fields=list(dict.fromkeys(k for r in data for k in r))
  with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 print(json.dumps({"split_mode_n":len(modes),"family_rows":len(families),"paired_comparisons":len(pairs),"outdir":str(out)},ensure_ascii=False))
if __name__=="__main__":main()

