"""Plot newly computed 7.3 retrieval, supplementation, and oracle diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

K=(3,5,8)
COLORS={"lexical":"#8c6bb1","dense":"#3182bd","hybrid":"#238b45","named_source_filter":"#e6550d"}
METHODS={"lexical":"词项检索","dense":"稠密检索","hybrid":"混合排序","named_source_filter":"查询显式机构过滤"}

def load(root,name):
 path=root/name
 if not path.exists():
  if name=="summary_test.json":path=root/"extended_test_summary_final"/"summary.json"
  elif name=="summary_challenge.json":path=root/"extended_challenge_summary_final"/"summary.json"
  elif name=="oracle_summary_test.json":path=root/"extended_oracle_summary_test"/"summary.json"
  elif name=="oracle_summary_challenge.json":path=root/"extended_oracle_summary_challenge"/"summary.json"
 with path.open(encoding="utf-8") as f:return json.load(f)
def style():
 plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":11,"axes.titlesize":14,"axes.labelsize":12,"legend.fontsize":10,"figure.dpi":130,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
def save(fig,out,name):
 fig.savefig(out/name,bbox_inches="tight",facecolor="white");plt.close(fig)
def fig_topk(data,out):
 style();fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),sharey=True,constrained_layout=True)
 for ax,split,label in zip(axes,("test","challenge"),("测试集","挑战集")):
  p=load(data,f"extended_retrieval_{split}_multi.json");n=p["question_count"]
  for method,name in METHODS.items():
   vals=[p["initial_rank_summary"][method][str(k)]["complete_rate"] for k in K]
   ax.plot(K,vals,marker="o",linewidth=2.3,markersize=6,color=COLORS[method],label=name)
   for k,v in zip(K,vals):
    c=p["initial_rank_summary"][method][str(k)]["complete_n"]
    ax.annotate(f"{c}/{n}",(k,v),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8,color=COLORS[method])
  ax.set_title(f"{label}（n={n}）");ax.set_xlabel("初始候选深度 K");ax.set_xticks(K);ax.set_ylim(0.55,1.08);ax.grid(axis="y",alpha=.28)
 axes[0].set_ylabel("完整金标集合覆盖率")
 axes[1].legend(frameon=False,loc="lower right")
 fig.suptitle("固定语料上的初始检索深度与金标覆盖",fontsize=17,fontweight="bold")
 fig.text(.5,-.025,"按相同的80条证据库进行无生成候选排序；完整覆盖指全部必需证据ID进入Top-K。来源过滤只用于明确点名机构的问题。",ha="center",fontsize=10,color="#444")
 save(fig,out,"图7_初始检索TopK覆盖.png")
def recall(gold,order,k):
 g=set(gold);return len(g&set(order[:k]))/len(g) if g else 1.0
def fig_source_filter(data,out):
 style();fig,axes=plt.subplots(1,2,figsize=(13.5,5.6),sharey=True,constrained_layout=True)
 cats=("下降","不变","提高");catcolors={"下降":"#d73027","不变":"#bdbdbd","提高":"#1a9850"}
 for ax,split,label in zip(axes,("test","challenge"),("测试集","挑战集")):
  p=load(data,f"extended_retrieval_{split}_multi.json");cases=[r for r in p["cases"] if r["source_filter_applied"]];n=len(cases)
  bottoms=np.zeros(3)
  for ci,c in enumerate(cats):
   counts=[]
   for k in K:
    num=0
    for row in cases:
     r0=recall(row["gold"],row["rankings"]["hybrid"],k)
     r1=recall(row["gold"],row["rankings"]["named_source_filter"],k)
     delta=r1-r0
     cat="提高" if delta>1e-9 else "下降" if delta<-1e-9 else "不变"
     num+=cat==c
    counts.append(num)
   heights=np.array(counts,dtype=float)/n if n else np.zeros(3)
   ax.bar(np.arange(3),heights,bottom=bottoms,color=catcolors[c],label=f"召回{c}",width=.66)
   for x,(h,cnt) in enumerate(zip(heights,counts)):
    if cnt:ax.text(x,bottoms[x]+h/2,f"{cnt}",ha="center",va="center",fontsize=9,color="white" if c!="不变" else "#333",fontweight="bold")
   bottoms+=heights
  ax.set_title(f"{label}（显式来源题 n={n}）");ax.set_xticks(range(3),[f"K={k}" for k in K]);ax.set_ylim(0,1.06);ax.grid(axis="y",alpha=.25)
 axes[0].set_ylabel("题目比例（柱内数字为题数）")
 axes[1].legend(frameon=False,loc="lower right")
 fig.suptitle("查询显式机构来源过滤对逐题金标召回的影响",fontsize=17,fontweight="bold")
 fig.text(.5,-.025,"仅统计题目明确点名来源、实际应用过滤的样本；提高/下降按同题混合检索对照计算。未提名来源的题不纳入本图。",ha="center",fontsize=9.5,color="#444")
 save(fig,out,"图8_来源过滤逐题变化.png")
def fig_facets(data,out):
 style();fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True,constrained_layout=True)
 series=[("test","multi","测试集·强制多轮","#2171b5"),("test","adaptive","测试集·自适应","#6baed6"),("challenge","multi","挑战集·强制多轮","#238b45"),("challenge","adaptive","挑战集·自适应","#74c476")]
 for ax,split,title in zip(axes,("test","challenge"),("测试集","挑战集")):
  for ds,mode,label,color in series:
   if ds!=split:continue
   p=load(data,f"extended_retrieval_{ds}_{mode}.json");f=p["facet_union_summary"]
   if not f or not f.get("3",{}).get("n_with_facets",0):continue
   n=f["3"]["n_with_facets"]
   vals=[f[str(k)]["named_source_filter"]["complete_rate"] for k in K]
   ax.plot(K,vals,marker="o",linewidth=2.4,markersize=6,color=color,label=f"{label}（n={n}）")
   for k,v in zip(K,vals):
    c=f[str(k)]["named_source_filter"]["complete_n"]
    offset=(0,11) if mode=="multi" else (0,-17)
    ax.annotate(f"{c}/{n}",(k,v),xytext=offset,textcoords="offset points",ha="center",fontsize=8,color=color)
  ax.set_title(title);ax.set_xticks(K);ax.set_xlabel("每条子查询独立候选深度 K");ax.set_ylim(.4,1.08);ax.grid(axis="y",alpha=.25);ax.legend(frameon=False,loc="lower right")
 axes[0].set_ylabel("各题所有必需证据的并集合覆盖率")
 fig.suptitle("生成子查询的独立检索可行性",fontsize=17,fontweight="bold")
 fig.text(.5,-.025,"对实际补检trace中的子查询分别按同一检索器重排并合并Top-K；使用查询显式来源过滤。该图是候选可行性模拟，不是系统实际返回列表。n为有补检子查询的可答题数。",ha="center",fontsize=9.3,color="#444")
 save(fig,out,"图9_子查询TopK可行覆盖.png")
def fig_supplement(data,out):
 style();fig,axes=plt.subplots(1,2,figsize=(13.5,5.7),constrained_layout=True)
 specs=[("test","multi","测试集·强制多轮"),("test","adaptive","测试集·自适应"),("challenge","multi","挑战集·强制多轮"),("challenge","adaptive","挑战集·自适应")]
 labels=[];first=[];final=[];added=[];rescued=[];den=[]
 for split,mode,label in specs:
  p=load(data,f"extended_trace_{split}_{mode}.json")["summary"]
  labels.append(label);first.append(p["mean_first_gold_recall"]);final.append(p["mean_final_gold_recall"]);added.append(p["n_supplement_added_gold"]);rescued.append(p["n_supplement_rescued_supported_answer"]);den.append(p["n_with_supplement"])
 x=np.arange(len(labels));w=.34
 axes[0].bar(x-w/2,first,w,label="首轮证据平均召回",color="#6baed6")
 axes[0].bar(x+w/2,final,w,label="补检后平均召回",color="#2171b5")
 axes[0].set_xticks(x,labels,rotation=15,ha="right");axes[0].set_ylim(0,1.08);axes[0].set_ylabel("金标证据召回率");axes[0].set_title("金标证据覆盖");axes[0].grid(axis="y",alpha=.25);axes[0].legend(frameon=False)
 for bars,vals in zip(axes[0].containers,(first,final)):
  axes[0].bar_label(bars,labels=[f"{v:.2f}" for v in vals],padding=2,fontsize=8)
 axes[1].bar(x-w/2,added,w,label="新增至少一条金标",color="#31a354")
 axes[1].bar(x+w/2,rescued,w,label="救回关键点且引文完整",color="#756bb1")
 axes[1].set_xticks(x,[f"{l}\n补检n={n}" for l,n in zip(labels,den)],rotation=15,ha="right");axes[1].set_ylim(0,max(4,max(added,default=0)+1));axes[1].set_ylabel("题数");axes[1].set_title("实际补检增量与答案救回");axes[1].grid(axis="y",alpha=.25);axes[1].legend(frameon=False)
 for bars,vals in zip(axes[1].containers,(added,rescued)):axes[1].bar_label(bars,labels=[str(v) for v in vals],padding=2,fontsize=8)
 fig.suptitle("实际补检是否找回缺失金标并改善答案",fontsize=17,fontweight="bold")
 fig.text(.5,-.02,"依照新引擎trace中的first_evidence、supplement retrieved IDs及评分结果；分母为可答且有金标题数。新增金标不等价于答案救回。",ha="center",fontsize=9.5,color="#444")
 save(fig,out,"图10_实际补检增量与答案救回.png")
def fig_oracle(data,out):
 style();fig,axes=plt.subplots(1,2,figsize=(13.2,5.5),sharey=True,constrained_layout=True)
 for ax,outcome,title in zip(axes,("supported","citation"),("严格关键点与引用联合通过","完整金标引用ID覆盖")):
  positions=[];values=[];colors=[];labels=[];counts=[];x=0
  for split,n,label in (("test",22,"测试集"),("challenge",20,"挑战集")):
   p=load(data,f"oracle_summary_{split}.json")
   rows={r["mode"]:r for r in p["modes"]}
   for mode,mlab,color in (("allgold_critic","完整核证",("#3182bd" if split=="test" else "#74a9cf")),("allgold_writer_only","仅writer",("#31a354" if split=="test" else "#78c679"))):
    r=rows[mode]
    if outcome=="supported":num=int(r["answered_and_supported_n"]);den=int(r["answerable_n"])
    else:num=round(float(r["complete_gold_citation_rate"])*int(r["answerable_n"]));den=int(r["answerable_n"])
    positions.append(x);values.append(num/den if den else 0);colors.append(color);labels.append(f"{label}\n{mlab}");counts.append(f"{num}/{den}");x+=1
   x+=.6
  bars=ax.bar(positions,values,color=colors,width=.72)
  ax.bar_label(bars,labels=counts,padding=3,fontsize=9)
  ax.set_xticks(positions,labels);ax.set_ylim(0,1.06);ax.set_title(title);ax.grid(axis="y",alpha=.25)
 axes[0].set_ylabel("严格文字/ID口径通过率")
 fig.suptitle("完整金标证据条件下的writer与核证诊断",fontsize=17,fontweight="bold")
 fig.text(.5,-.04,"Oracle条件：每题直接提供全部金标证据。Writer臂绕过critic、路由与验证；引文覆盖来自原始writer输出ID，未验证锚点。所有keypoint仍按冻结expected_groups作机械文字匹配。",ha="center",fontsize=9.0,color="#444")
 save(fig,out,"图11_Oracle固定证据Writer核证诊断.png")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data-dir",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 for fn in (fig_topk,fig_source_filter,fig_facets,fig_supplement,fig_oracle):fn(a.data_dir,a.out_dir)
 print("generated",*[p.name for p in sorted(a.out_dir.glob("图[7-9]*.png"))],*[p.name for p in sorted(a.out_dir.glob("图1*.png"))])
if __name__=="__main__":main()


