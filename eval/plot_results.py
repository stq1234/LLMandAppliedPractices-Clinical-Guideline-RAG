"""Render the three Chinese evaluation figures from summarize.py output.

Usage: python plot_results.py --summary summary.json --outdir figures
Metrics are descriptive; expected-keypoint matching is not semantic clinical
adjudication. The answerable-performance panel also requires complete gold-ID
citations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "reuse": "给定证据",
    "single": "固定单轮",
    "multi": "固定双轮",
    "adaptive": "规则自适应",
    "adaptive_nocrit": "自适应无语义核证",
    "adaptive_learned": "dev校准阈值",
}


def _configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 240
    plt.rcParams["font.size"] = 10


def _mode_rows(summary: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = [row for row in summary.get("modes", []) if row.get("split") == split and row.get("n_valid", 0)]
    preferred = ["reuse", "single", "multi", "adaptive", "adaptive_nocrit", "adaptive_learned"]
    rank = {mode: i for i, mode in enumerate(preferred)}
    return sorted(rows, key=lambda row: rank.get(row.get("mode"), 100))


def plot_modes(summary: dict[str, Any], outdir: Path) -> Path:
    rows = _mode_rows(summary, "test")
    if not rows:
        raise ValueError("summary.json has no valid test-mode rows")
    labels = [LABELS.get(str(r["mode"]), str(r["mode"])) for r in rows]
    x = np.arange(len(rows))
    width = 0.23
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.4, 6.1), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.28, top=0.86, wspace=0.20)

    values = [r.get("supported_keypoint_match_rate") for r in rows]
    lows = [r.get("supported_keypoint_match_ci95_low") for r in rows]
    highs = [r.get("supported_keypoint_match_ci95_high") for r in rows]
    y = np.array([np.nan if v is None else float(v) for v in values])
    low = np.array([0.0 if v is None else float(v) for v in lows])
    high = np.array([0.0 if v is None else float(v) for v in highs])
    err = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
    ax0.bar(x, y, color="#2C7FB8", width=0.62, zorder=2)
    ax0.errorbar(x, y, yerr=err, fmt="none", ecolor="#222222", capsize=3, lw=1, zorder=3)
    for i, row in enumerate(rows):
        ax0.text(i, min(1.02, y[i] + 0.04), f"n={row.get('answerable_n', 0)}", ha="center", va="bottom", fontsize=8)
    ax0.set_title("可答题：关键点匹配且金标引用齐全")
    ax0.set_ylabel("比例（95% Wilson 区间）")
    ax0.set_ylim(0, 1.13)
    ax0.set_xticks(x, labels, rotation=25, ha="right")
    ax0.grid(axis="y", alpha=0.25, zorder=0)

    series = [
        ("unanswerable_correct_refusal_rate", "unanswerable_refusal_ci95_low", "unanswerable_refusal_ci95_high", "范围内不可答正确拒答", "#41AB5D", "unanswerable_n"),
        ("scope_out_correct_refusal_rate", "scope_out_refusal_ci95_low", "scope_out_refusal_ci95_high", "范围外正确拒答", "#F28E2B", "scope_out_n"),
    ]
    for j, (key, lo_key, hi_key, label, color, n_key) in enumerate(series):
        vals = np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])
        los = np.array([vals[i] if r.get(lo_key) is None else float(r[lo_key]) for i, r in enumerate(rows)])
        his = np.array([vals[i] if r.get(hi_key) is None else float(r[hi_key]) for i, r in enumerate(rows)])
        err = np.vstack([np.maximum(0, vals - los), np.maximum(0, his - vals)])
        xpos = x + (j - 0.5) * width
        ax1.bar(xpos, vals, width=width, label=label, color=color, zorder=2)
        ax1.errorbar(xpos, vals, yerr=err, fmt="none", ecolor="#222222", capsize=2, lw=0.8, zorder=3)
        for i, row in enumerate(rows):
            if vals[i] == vals[i]:
                ax1.text(xpos[i], min(1.02, vals[i] + 0.04), f"n={row.get(n_key, 0)}", ha="center", fontsize=7, rotation=90)
    ax1.set_title("拒答边界：不可答与范围外分别统计")
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("正确拒答比例（95% Wilson 区间）")
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="lower center", bbox_to_anchor=(0.75, 0.035), ncol=2)
    fig.suptitle("测试集策略表现（逐题配对评测）", fontsize=14)
    path = outdir / "图1_测试集回答与拒答表现.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_frontier(summary: dict[str, Any], outdir: Path) -> Path:
    rows = [r for r in _mode_rows(summary, "test") if r.get("supported_keypoint_match_rate") is not None]
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    rounds = [float(r.get("rounds_mean") or 0.0) for r in rows]
    norm = plt.Normalize(min(rounds, default=0), max(rounds, default=1) or 1)
    sc = None
    for row, avg_rounds in zip(rows, rounds):
        tokens = float(row.get("prompt_tokens_mean") or 0) + float(row.get("completion_tokens_mean") or 0)
        value = float(row["supported_keypoint_match_rate"])
        latency = float(row.get("latency_seconds_mean") or 0)
        sc = ax.scatter(tokens, value, s=90, c=[avg_rounds], cmap="viridis", norm=norm, edgecolor="black", linewidth=0.7, zorder=3)
        label = LABELS.get(str(row["mode"]), str(row["mode"]))
        ax.annotate(f"{label}\n{latency:.1f} 秒", (tokens, value), xytext=(7, 7), textcoords="offset points", fontsize=8)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("平均检索轮数")
    ax.set_xlabel("平均输入与输出 tokens / 题")
    ax.set_ylabel("可答题：关键点匹配且金标引用齐全比例（放大区间；全量见图1）")
    ax.set_title("测试集回答质量与调用代价")
    upper = max(0.20, min(1.0, np.ceil(max([float(r["supported_keypoint_match_rate"]) for r in rows] + [0.0]) * 10) / 10 + 0.05))
    ax.set_ylim(0, upper)
    ax.grid(alpha=0.25)
    path = outdir / "图2_测试集质量与资源代价.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_supplement(summary: dict[str, Any], outdir: Path) -> Path:
    rows = [r for r in summary.get("supplement_diagnostics", []) if r.get("split") == "test"]
    rows = [r for r in rows if r.get("mode") in ("adaptive", "adaptive_learned")]
    if not rows:
        raise ValueError("summary.json has no adaptive test supplement diagnostics")
    keys = [
        ("first_round_missing_gold_n", "首轮缺必需证据"),
        ("second_round_trace_n", "执行第二轮"),
        ("second_round_added_required_gold_n", "新增金标证据"),
        ("second_round_rescued_supported_keypoint_n", "挽回可支持回答"),
        ("second_round_no_new_required_gold_n", "未新增金标证据"),
        ("missed_supplement_recovered_by_paired_forced_multi_n", "漏补但双轮可救回"),
        ("adequate_gold_evidence_but_abstained_n", "依据齐全仍拒答"),
    ]
    modes = [str(r["mode"]) for r in rows]
    labels = [LABELS.get(m, m) for m in modes]
    x = np.arange(len(keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.5, 6.0), constrained_layout=True)
    for j, row in enumerate(rows):
        vals = [int(row.get(key, 0) or 0) for key, _ in keys]
        xpos = x + (j - (len(rows) - 1) / 2) * width
        bars = ax.bar(xpos, vals, width=width, label=labels[j], color=("#3182BD" if modes[j] == "adaptive" else "#FD8D3C"), zorder=2)
        for bar, val in zip(bars, vals):
            if val:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.08, str(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [name for _, name in keys], rotation=18, ha="right")
    ax.set_ylabel("题数")
    ax.set_title("首轮证据缺口、第二轮增量与失误分解（测试集）")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    path = outdir / "图3_补检证据增量与失误.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_initial_routing(summary: dict[str, Any], outdir: Path) -> Path:
    rows = [
        row for row in _mode_rows(summary, "test")
        if row.get("mode") in ("adaptive", "adaptive_learned")
    ]
    if not rows:
        raise ValueError("summary.json has no adaptive test routing rows")
    labels = [LABELS.get(str(row["mode"]), str(row["mode"])) for row in rows]
    x = np.arange(len(rows))
    width = 0.22
    series = [
        ("initial_need_retrieval_required_n", "需要首轮检索", "#3182BD"),
        ("initial_retrieval_executed_when_required_n", "实际执行首轮检索", "#31A354"),
        ("missed_initial_retrieval_n", "需要查证但未检索", "#DE2D26"),
        ("initial_retrieval_without_need_n", "当前证据足够或范围外却检索", "#F28E2B"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    for j, (key, label, color) in enumerate(series):
        vals = [int(row.get(key, 0) or 0) for row in rows]
        xpos = x + (j - (len(series)-1)/2) * width
        bars = ax.bar(xpos, vals, width=width, label=label, color=color, zorder=2)
        for bar, val in zip(bars, vals):
            if val:
                ax.text(bar.get_x()+bar.get_width()/2, val+0.08, str(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("题数（测试集）")
    ax.set_title("首轮检索门控：需求标签与实际路由")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    path = outdir / "图4_首轮检索需求与路由.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    _configure_fonts()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_modes(summary, args.outdir),
        plot_cost_frontier(summary, args.outdir),
        plot_supplement(summary, args.outdir),
        plot_initial_routing(summary, args.outdir),
    ]
    for path in paths:
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

