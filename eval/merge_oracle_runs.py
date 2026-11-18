"""Assemble fixed-evidence critic and writer-only rows, excluding fallback pilot."""
import argparse,json
from pathlib import Path
def read(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--critic",type=Path,required=True);ap.add_argument("--writer-only",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 c=[r for r in read(a.critic) if r.get("mode")=="allgold_critic" and r.get("status")=="ok" and int(r.get("rounds",0) or 0)==0]
 w=[r for r in read(a.writer_only) if r.get("mode")=="allgold_writer_only" and r.get("status")=="ok" and int(r.get("rounds",0) or 0)==0]
 ci={str(r["id"]) for r in c};wi={str(r["id"]) for r in w}
 if ci!=wi:raise ValueError(f"critic/writer-only ID mismatch: critic_only={sorted(ci-wi)} writer_only={sorted(wi-ci)}")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",encoding="utf-8") as f:
  for r in sorted(c+w,key=lambda x:(str(x.get("id")),str(x.get("mode")))):f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
 print(f"fixed_evidence_ids={len(ci)} critic_rows={len(c)} writer_only_rows={len(w)} excluded_route_fallback_rows={sum(r.get('mode')=='allgold_nocrit' and int(r.get('rounds',0) or 0)>0 for r in read(a.critic))} output={a.output}")
if __name__=="__main__":main()
