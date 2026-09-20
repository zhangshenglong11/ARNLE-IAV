#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ha_na_threshold_sensitivity.py

目的
----
审稿意见 R1-M6：build_ha_na_paired_model_fixed.py 中 classify_joint_state()
使用固定的 0.60 阈值判定 HA / NA 是否"接近目标宿主优势状态"，未做敏感性分析、
未说明阈值依据。本脚本对同一份配对表，在多组阈值（默认 0.5/0.6/0.7/0.8）下
重新执行完全相同的联合状态分类规则，比较各类别占比随阈值的变化，输出：

1. 每个阈值下的联合状态构成表（长表，便于画堆叠柱状图）。
2. 阈值敏感性汇总（宽表：每行一个阈值，每列一个联合状态类别的占比）。
3. 敏感性折线图（PNG）。

分类规则与 build_ha_na_paired_model_fixed.py 的 classify_joint_state() 完全一致
（仅把硬编码的 0.6 换成可配置阈值），以保证阈值=0.6 时的结果与正文报告数字一致，
可作为交叉核对。

用法示例
--------
python ha_na_threshold_sensitivity.py \
  --paired-csv /path/to/H5N1_artiodactyla_to_primates_paired.csv \
  --label "H5N1 artiodactyla_to_primates" \
  --thresholds 0.5 0.6 0.7 0.8 \
  --outdir /path/to/out
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def safe_name(text: str) -> str:
    import re
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return re.sub(r"_+", "_", text).strip("_") or "combo"


def classify_joint_state_at_threshold(ha_frac, na_frac, in_transition_ha, in_transition_na, threshold: float) -> str:
    """Reproduces build_ha_na_paired_model_fixed.py's classify_joint_state()
    logic exactly, with the hardcoded 0.6 replaced by `threshold`."""
    ha_high = pd.notna(ha_frac) and float(ha_frac) >= threshold
    na_high = pd.notna(na_frac) and float(na_frac) >= threshold

    if ha_high and na_high:
        return "HA和NA均接近人源优势状态"
    if ha_high and not na_high:
        return "仅HA接近人源优势状态"
    if na_high and not ha_high:
        return "仅NA接近人源优势状态"
    if bool(in_transition_ha) or bool(in_transition_na):
        return "位于至少一个蛋白的中间样本带但优势位点比例不高"
    return "未显示明显联合人源优势状态"


def run_sensitivity(df: pd.DataFrame, thresholds: List[float], ha_col: str, na_col: str,
                     ha_trans_col: Optional[str], na_trans_col: Optional[str]) -> pd.DataFrame:
    ha = df[ha_col]
    na = df[na_col]
    in_trans_ha = df[ha_trans_col] if ha_trans_col and ha_trans_col in df.columns else pd.Series(False, index=df.index)
    in_trans_na = df[na_trans_col] if na_trans_col and na_trans_col in df.columns else pd.Series(False, index=df.index)

    rows = []
    for t in thresholds:
        states = [
            classify_joint_state_at_threshold(h, n, ih, ina, t)
            for h, n, ih, ina in zip(ha, na, in_trans_ha, in_trans_na)
        ]
        counts = pd.Series(states).value_counts()
        total = counts.sum()
        for state, n_state in counts.items():
            rows.append({"threshold": t, "joint_state": state, "n": int(n_state), "fraction": float(n_state / total)})
    return pd.DataFrame(rows)


def plot_sensitivity(long_df: pd.DataFrame, label: str, out_png: str) -> None:
    # Use English category labels for the plot only (CSV keeps the original
    # Chinese labels) to avoid CJK glyph-missing warnings/tofu boxes on
    # machines without a CJK-capable matplotlib font configured.
    EN_LABELS = {
        "HA和NA均接近人源优势状态": "HA & NA both target-major",
        "仅HA接近人源优势状态": "HA only target-major",
        "仅NA接近人源优势状态": "NA only target-major",
        "位于至少一个蛋白的中间样本带但优势位点比例不高": "In transition band, low target-major",
        "未显示明显联合人源优势状态": "No clear joint target-major state",
    }
    plot_df = long_df.copy()
    plot_df["joint_state"] = plot_df["joint_state"].map(lambda s: EN_LABELS.get(s, s))
    wide = plot_df.pivot_table(index="threshold", columns="joint_state", values="fraction", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for state in wide.columns:
        ax.plot(wide.index, wide[state], marker="o", label=state)
    ax.set_xlabel("Joint-state threshold")
    ax.set_ylabel("Fraction of paired isolates")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity analysis for the HA-NA joint-state 0.60 threshold "
                     "(addresses reviewer comment R1-M6)."
    )
    ap.add_argument("--paired-csv", required=True, help="Output of build_ha_na_paired_model_fixed.py")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--ha-col", default="HA_target_major_fraction")
    ap.add_argument("--na-col", default="NA_target_major_fraction")
    ap.add_argument("--ha-transition-col", default="HA_in_transition_band")
    ap.add_argument("--na-transition-col", default="NA_in_transition_band")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.paired_csv)
    for col in (args.ha_col, args.na_col):
        if col not in df.columns:
            raise SystemExit(f"Column {col!r} not found in {args.paired_csv}. Available: {list(df.columns)}")

    long_df = run_sensitivity(df, args.thresholds, args.ha_col, args.na_col, args.ha_transition_col, args.na_transition_col)
    long_path = os.path.join(args.outdir, f"{safe_name(args.label)}_threshold_sensitivity_long.csv")
    long_df.to_csv(long_path, index=False)

    wide_df = long_df.pivot_table(index="threshold", columns="joint_state", values="fraction", fill_value=0.0).reset_index()
    wide_path = os.path.join(args.outdir, f"{safe_name(args.label)}_threshold_sensitivity_wide.csv")
    wide_df.to_csv(wide_path, index=False)

    png_path = os.path.join(args.outdir, f"{safe_name(args.label)}_threshold_sensitivity.png")
    plot_sensitivity(long_df, args.label, png_path)

    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")
    print(f"Saved: {png_path}")
    print("\nComposition at each threshold:")
    print(wide_df.to_string(index=False))


if __name__ == "__main__":
    main()
