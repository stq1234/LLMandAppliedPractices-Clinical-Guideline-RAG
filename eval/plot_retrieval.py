"""Plot first-pass evidence coverage on the frozen questions."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(source, dest):
    data=json.loads(source.read_text(encoding="utf-8"))["summary"]
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans"],
                         "axes.unicode_minus":False, "axes.spines.top":False,
                         "axes.spines.right":False, "font.size":10})
    methods={"lexical":"BM25", "dense":"稠密检索", "hybrid":"混合检索", "named_sources":"混合检索＋来源约束"}
    fig, axes=plt.subplots(1,2,figsize=(10.5,3.5),constrained_layout=True,sharey=True)
    depths=[1,3,5,8,10]
    for ax,(group,label) in zip(axes,[("single","单证据问题"),("multi","多证据问题")]):
        matrix=np.array([[data[group][m][str(k)]["complete"] for k in depths] for m in methods])
        shade=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=12,aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value=matrix[row,col]
                ax.text(col,row,str(value),ha="center",va="center",fontsize=11,
                        color="white" if value>=8 else "#102b41")
        ax.set_title(label+"（11 题）",loc="left",fontsize=12)
        ax.set_xticks(range(len(depths)),depths)
        ax.set_yticks(range(len(methods)),list(methods.values()))
        ax.set_xlabel("候选条目数 K")
        ax.tick_params(length=0,pad=6)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    bar=fig.colorbar(shade,ax=axes,shrink=.90,ticks=[0,3,6,9,11])
    bar.set_label("金标齐全的题数")
    fig.savefig(dest,dpi=200,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("dest",type=Path)
    args=parser.parse_args()
    draw(args.source,args.dest)
