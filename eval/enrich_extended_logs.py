"""Derive consistent route/supplement flags from recorded System trace rows."""
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 n=0
 with a.input.open(encoding="utf-8") as src,a.output.open("w",encoding="utf-8") as dst:
  for line in src:
   if not line.strip():continue
   r=json.loads(line); trace=r.get("trace") or []
   routers=[s for s in trace if isinstance(s,dict) and s.get("stage")=="router"]
   supplements=[s for s in trace if isinstance(s,dict) and s.get("stage")=="retrieval" and s.get("action")=="supplement"]
   r["effective_route"]=routers[-1].get("action") if routers else r.get("initial_action")
   r["adaptive_fallback"]=any(s.get("reason")=="supplied_evidence_gap_triggered_retrieval_fallback" for s in routers)
   r["supplement_attempted"]=bool(supplements)
   r["supplement_reason"]=supplements[-1].get("reason") if supplements else None
   r["initial_retrieval_executed"]=int(r.get("rounds",0) or 0)>0
   r["initial_retrieval_trace_seen"]=any(isinstance(s,dict) and s.get("stage")=="retrieval" and s.get("action")=="initial" for s in trace)
   dst.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n");n+=1
 print(f"rows={n} output={a.output}")
if __name__=="__main__":main()

