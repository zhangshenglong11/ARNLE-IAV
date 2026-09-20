#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classic_marker_site_recall_check.py

目的
----
审稿意见 R2-M3 / R4-M3：全文没有以教科书级宿主适应标志位点（如 PB2-627、
PB2-701、HA 受体结合位点 H3-226/228、H1-190/225，以及 NP 的 MxA 逃逸位点）
作为阳性对照，无法判断候选位点筛查方法本身的特异性/灵敏度。

本脚本检查这些经典位点是否出现在候选位点全集（不只是最终入选并公开报告的
位点，而是候选位点筛选前——即通过"来源/目标主要氨基酸不同 + 计数>50"
预筛之后、统计检验之前——的完整表格）中，对每个方向组合分别报告：

1. 该位点是否进入候选池（即是否通过预筛）。
2. 若进入候选池，其效应量、trend_p、trend_q_BH_within_analysis、
   evidence_level（如表中已有该列）等统计结果。
3. 若未进入候选池，尝试的原因（若原始表中保留了 source_major_aa /
   target_major_aa / count 等预筛依据列，会一并报告，帮助判断是"预筛就被
   过滤掉"还是"预筛通过但统计检验未达标"）。

默认经典位点表基于常见文献报道，位点编号以你实际数据的编号体系为准——
**编号体系是否与经典文献一致，需要你确认**（审稿人三同样指出编号体系不透明
的问题）；如与标准编号不同，请用 --marker-table 提供你自己的映射表。

用法示例
--------
python classic_marker_site_recall_check.py \
  --candidate-pool-csv /path/to/all_candidate_sites_before_filtering.csv \
  --outdir /path/to/out

# 自定义经典位点表（CSV: protein,position,marker_name,reference）
python classic_marker_site_recall_check.py \
  --candidate-pool-csv /path/to/all_candidate_sites_before_filtering.csv \
  --marker-table /path/to/my_markers.csv \
  --outdir /path/to/out
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd


DEFAULT_MARKERS = pd.DataFrame([
    # protein, position, marker_name, reference_note
    ("PB2", "627", "PB2 E627K — 聚合酶活性与哺乳动物宿主适应经典位点", "Subbarao et al. 1993; Hatta et al. 2001"),
    ("PB2", "701", "PB2 D701N — 与 ANP32 相互作用、哺乳动物适应", "Gabriel et al. 2005"),
    ("HA", "226", "HA Q226L（H3 numbering）— 受体结合偏好转换", "Rogers & Paulson 1983; Connor et al. 1994"),
    ("HA", "228", "HA G228S（H3 numbering）— 受体结合偏好转换", "Connor et al. 1994"),
    ("HA", "190", "HA E190D（H1 numbering）— 受体结合偏好转换", "Glaser et al. 2005"),
    ("HA", "225", "HA G225D（H1 numbering）— 受体结合偏好转换，2009 H1N1 相关", "Glaser et al. 2005; Chutinimitkul et al. 2010"),
    ("NP", "100", "NP 位点区域 — 已报道影响人源 MxA 逃逸的候选区域之一（示例位点，需按你数据实际编号核对）", "Manz et al. 2013; Riegger et al. 2015"),
], columns=["protein", "position", "marker_name", "reference_note"])


def load_marker_table(path: Optional[str]) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        required = {"protein", "position", "marker_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"--marker-table is missing columns: {missing}")
        return df
    return DEFAULT_MARKERS.copy()


def normalize_position(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def check_recall(
    candidate_pool: pd.DataFrame,
    markers: pd.DataFrame,
    protein_col: str,
    position_col: str,
    analysis_col: Optional[str],
) -> pd.DataFrame:
    pool = candidate_pool.copy()
    pool[position_col] = pool[position_col].map(normalize_position)
    if protein_col in pool.columns:
        pool[protein_col] = pool[protein_col].astype(str).str.upper().str.strip()

    report_cols = [c for c in [
        "analysis_label", "direction", "subtype", "source_major_aa", "target_major_aa",
        "n_total_sequences", "delta_target_near_minus_source_near",
        "effect_size_delta_target_minus_source", "trend_p", "trend_q_BH_within_analysis",
        "evidence_level", "logistic_beta",
    ] if c in pool.columns]

    rows = []
    for _, marker in markers.iterrows():
        m_protein = str(marker["protein"]).upper().strip()
        m_position = normalize_position(marker["position"])

        if protein_col in pool.columns:
            hits = pool[(pool[protein_col] == m_protein) & (pool[position_col] == m_position)]
        else:
            hits = pool[pool[position_col] == m_position]

        if len(hits) == 0:
            rows.append({
                "protein": m_protein,
                "position": m_position,
                "marker_name": marker["marker_name"],
                "reference_note": marker.get("reference_note", ""),
                "recalled_in_candidate_pool": False,
                "n_matching_rows": 0,
            })
            continue

        for _, hit in hits.iterrows():
            row = {
                "protein": m_protein,
                "position": m_position,
                "marker_name": marker["marker_name"],
                "reference_note": marker.get("reference_note", ""),
                "recalled_in_candidate_pool": True,
                "n_matching_rows": len(hits),
            }
            for c in report_cols:
                row[c] = hit.get(c, None)
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Check whether classic host-adaptation marker sites are recalled "
                     "in the candidate site pool (addresses reviewer comments R2-M3 / R4-M3)."
    )
    ap.add_argument("--candidate-pool-csv", required=True,
                     help="Candidate site table BEFORE statistical filtering (or the final "
                          "candidate table if a pre-filter version isn't available — see note below)")
    ap.add_argument("--marker-table", default=None,
                     help="Optional CSV with columns protein,position,marker_name[,reference_note] "
                          "to override the built-in default marker list")
    ap.add_argument("--protein-col", default="protein")
    ap.add_argument("--position-col", default="position")
    ap.add_argument("--analysis-col", default="analysis_label")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    pool = pd.read_csv(args.candidate_pool_csv)
    if args.position_col not in pool.columns:
        raise SystemExit(
            f"Column {args.position_col!r} not found in candidate pool table. "
            f"Available columns: {list(pool.columns)}"
        )

    markers = load_marker_table(args.marker_table)
    report = check_recall(pool, markers, args.protein_col, args.position_col, args.analysis_col)

    out_path = os.path.join(args.outdir, "classic_marker_recall_report.csv")
    report.to_csv(out_path, index=False)

    n_markers = markers["marker_name"].nunique() if "marker_name" in markers.columns else len(markers)
    n_recalled = report.loc[report["recalled_in_candidate_pool"], "marker_name"].nunique()
    print(f"Saved: {out_path}")
    print(f"Classic markers checked: {n_markers}")
    print(f"Classic markers recalled in candidate pool (at least once): {n_recalled}")
    if n_recalled < n_markers:
        missing = sorted(set(markers["marker_name"]) - set(report.loc[report["recalled_in_candidate_pool"], "marker_name"]))
        print("Not recalled:")
        for m in missing:
            print(f"  - {m}")
    print(
        "\n注意：本脚本检查的是你提供表格中的位点编号是否命中经典位点编号；"
        "如果你的下游编号与 H3/H1/N2 等标准编号不一致（审稿人三 R3-m4 已指出这一问题），"
        "命中结果可能不准确，建议先完成位点编号换算后再运行本脚本。"
    )


if __name__ == "__main__":
    main()
