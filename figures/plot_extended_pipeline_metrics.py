"""Plot the frozen 0.2.0 test and challenge outcomes from the adjacent CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "extended_pipeline_metrics.csv"
OUTPUT = HERE / "图6_扩展测试与挑战集证据覆盖.png"
MODE_LABELS = {
    "single": "单轮",
    "multi": "双轮",
    "adaptive": "自适应",
    "adaptive_nocrit": "自适应\n（无语义核证）",
}
COHORTS = (("test", "原测试集", "#4477AA"), ("challenge", "独立挑战集", "#EE7733"))
MODES = ("single", "multi", "adaptive", "adaptive_nocrit")


def read_rows() -> dict[tuple[str, str], dict[str, str]]:
    with DATA.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["cohort"], row["mode"]): row for row in rows}
    expected = {(cohort, mode) for cohort, _, _ in COHORTS for mode in MODES}
    if set(indexed) != expected:
        raise ValueError("The plotting table must contain the 2 × 4 fixed conditions.")
    return indexed


def main() -> None:
    data = read_rows()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    x = np.arange(len(MODES))
    width = 0.34

    panels = (
        ("strict_answer_n", "answerable_n", "可答题严格联合通过率"),
        ("final_context_full_n", "answerable_n", "最终上下文金标全覆盖率"),
    )
    for ax, (num_key, den_key, title) in zip(axes, panels):
        for offset, (cohort, label, color) in zip((-width / 2, width / 2), COHORTS):
            values = []
            labels = []
            for mode in MODES:
                row = data[(cohort, mode)]
                n, d = int(row[num_key]), int(row[den_key])
                values.append(n / d if d else 0.0)
                labels.append(f"{n}/{d}")
            bars = ax.bar(x + offset, values, width, color=color, label=label, edgecolor="white")
            for bar, label in zip(bars, labels):
                ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, pad=12, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6), ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.set_xticks(x, [MODE_LABELS[mode] for mode in MODES], ha="center")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("比例")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("冻结模型权重下的新旧查询集表现", fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "两组各含4道正确拒答题（2道库内无答案、2道范围外），此图只显示20/22道可答题。",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.14, 0.98, 0.90), w_pad=2.0)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
