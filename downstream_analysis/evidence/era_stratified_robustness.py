#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era_stratified_robustness_check.py

目的
----
审稿意见 R4-M5：公共数据库中年份、谱系、地区分布不均衡，候选位点检出结果
可能反映谱系/年代漂移而非真正的宿主适应信号；讨论已承认这一限制，但没有做
年代分层的稳健性检验。

本脚本读取某个代表性组合的样本级表（需包含采样年份列、宿主列、以及至少两列
PCA 坐标），按滑动年份窗口切分成若干子集，在每个子集内重新计算一个与
figure3_transition_band_quantification_svg.py 一致的信噪比风格的过渡带分离度
统计量（来源/目标质心的分离度相对于组内离散度之比），比较该统计量在不同年代
子集间是否稳定：

1. 每个年代窗口的分离度统计量、样本量。
2. 跨窗口的一致性指标：
   - 分离度符号是否一致（全部为正，或全部为负）
   - 分离度变异系数（CV = std / mean，越小越稳定）
   - 与全量数据分离度的 Pearson 相关（若窗口数 >= 3，做窗口分离度 vs 窗口
     中位年份的趋势检验，判断是否存在随时间单调漂移的证据）
3. 折线图：各年代窗口的分离度及其自助法置信区间。

用法示例
--------
python era_stratified_robustness_check.py \
  --pca-csv /path/to/pairwise_pca_coordinates_with_year.csv \
  --source-host artiodactyla --target-host primates \
  --year-col collection_year \
  --window-size 5 --step 2 \
  --outdir /path/to/out --label "NP_H1N1_artiodactyla_to_primates"

若样本量在窗口内过少，脚本会跳过该窗口并在输出中注明。
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def infer_column(df: pd.DataFrame, candidates, label: str) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(f"Cannot infer {label} column. Tried: {candidates}")


def infer_pc_cols(df: pd.DataFrame, user_cols: Optional[List[str]]) -> List[str]:
    if user_cols:
        return user_cols
    candidates = [c for c in ["PC1", "PC2", "PC3"] if c in df.columns]
    if len(candidates) < 2:
        raise ValueError("Could not find at least two of PC1/PC2/PC3; pass --pc-cols explicitly.")
    return candidates


def separation_statistic(coords: np.ndarray, host: np.ndarray, source_host: str, target_host: str,
                          n_boot: int, alpha: float, seed: int):
    is_source = host == source_host
    is_target = host == target_host
    n_source, n_target = int(is_source.sum()), int(is_target.sum())
    if n_source < 5 or n_target < 5:
        return {"status": "insufficient_samples", "n_source": n_source, "n_target": n_target,
                "separation": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    source_centroid = coords[is_source].mean(axis=0)
    target_centroid = coords[is_target].mean(axis=0)
    axis = target_centroid - source_centroid
    norm = np.linalg.norm(axis)
    if norm == 0:
        return {"status": "degenerate_axis", "n_source": n_source, "n_target": n_target,
                "separation": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    axis_unit = axis / norm
    source_proj = (coords[is_source] - source_centroid) @ axis_unit
    target_proj = (coords[is_target] - source_centroid) @ axis_unit
    pooled_std = np.sqrt((np.var(source_proj, ddof=1) + np.var(target_proj, ddof=1)) / 2.0)
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return {"status": "zero_variance", "n_source": n_source, "n_target": n_target,
                "separation": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    observed = float((target_proj.mean() - source_proj.mean()) / pooled_std)

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        s_bs = source_proj[rng.integers(0, n_source, n_source)]
        t_bs = target_proj[rng.integers(0, n_target, n_target)]
        pooled_std_bs = np.sqrt((np.var(s_bs, ddof=1) + np.var(t_bs, ddof=1)) / 2.0)
        boot_vals[b] = (t_bs.mean() - s_bs.mean()) / pooled_std_bs if pooled_std_bs > 0 else np.nan
    valid = boot_vals[np.isfinite(boot_vals)]
    ci_low = float(np.quantile(valid, alpha / 2)) if len(valid) else np.nan
    ci_high = float(np.quantile(valid, 1 - alpha / 2)) if len(valid) else np.nan

    return {"status": "ok", "n_source": n_source, "n_target": n_target,
            "separation": observed, "ci_low": ci_low, "ci_high": ci_high}


def make_windows(years: np.ndarray, window_size: int, step: int):
    y_min, y_max = int(np.nanmin(years)), int(np.nanmax(years))
    windows = []
    start = y_min
    while start <= y_max:
        end = start + window_size - 1
        windows.append((start, end))
        start += step
    return windows


def main():
    ap = argparse.ArgumentParser(
        description="Year-window stratified robustness check for a representative "
                     "source->target combination (addresses reviewer comment R4-M5)."
    )
    ap.add_argument("--pca-csv", required=True,
                     help="Per-sample table with PCA coordinates, host label, and a collection-year column")
    ap.add_argument("--source-host", required=True)
    ap.add_argument("--target-host", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--host-col", default=None)
    ap.add_argument("--year-col", default=None, help="Column with a numeric collection year (auto-inferred if omitted)")
    ap.add_argument("--pc-cols", nargs="+", default=None)
    ap.add_argument("--window-size", type=int, default=5, help="Window width in years")
    ap.add_argument("--step", type=int, default=2, help="Step between consecutive window start years")
    ap.add_argument("--min-per-group", type=int, default=15, help="Minimum source/target samples required within a window")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.pca_csv)
    host_col = args.host_col or infer_column(df, ["host", "host_class", "host_group", "true_host", "label"], "host")
    year_col = args.year_col or infer_column(df, ["collection_year", "year", "sample_year"], "year")
    pc_cols = infer_pc_cols(df, args.pc_cols)

    host = df[host_col].astype(str).str.strip().str.lower().to_numpy()
    source_host = args.source_host.strip().lower()
    target_host = args.target_host.strip().lower()
    years = pd.to_numeric(df[year_col], errors="coerce").to_numpy()
    coords_all = df[pc_cols].to_numpy(dtype=float)

    windows = make_windows(years, args.window_size, args.step)
    rows = []
    for (start, end) in windows:
        mask = (years >= start) & (years <= end)
        if mask.sum() == 0:
            continue
        res = separation_statistic(
            coords_all[mask], host[mask], source_host, target_host,
            args.n_boot, args.alpha, args.seed,
        )
        res.update({"window_start": start, "window_end": end, "window_mid": (start + end) / 2.0})
        rows.append(res)

    result_df = pd.DataFrame(rows)
    result_df["passed_min_n"] = (result_df["n_source"] >= args.min_per_group) & (result_df["n_target"] >= args.min_per_group)

    # Overall (unwindowed) separation for reference
    overall = separation_statistic(coords_all, host, source_host, target_host, args.n_boot, args.alpha, args.seed)

    out_csv = os.path.join(args.outdir, "era_stratified_separation.csv")
    result_df.to_csv(out_csv, index=False)

    usable = result_df[(result_df["status"] == "ok") & result_df["passed_min_n"]]
    consistency = {}
    if len(usable) >= 2:
        signs = np.sign(usable["separation"])
        consistency["sign_consistent"] = bool((signs == signs.iloc[0]).all())
        consistency["cv"] = float(usable["separation"].std(ddof=1) / usable["separation"].mean()) if usable["separation"].mean() != 0 else np.nan
    if len(usable) >= 3:
        rho, p = stats.spearmanr(usable["window_mid"], usable["separation"])
        consistency["drift_spearman_rho"] = float(rho)
        consistency["drift_spearman_p"] = float(p)
    consistency["n_windows_usable"] = int(len(usable))
    consistency["n_windows_total"] = int(len(result_df))
    consistency["overall_separation"] = overall["separation"]

    import json
    with open(os.path.join(args.outdir, "era_stratified_consistency_summary.json"), "w", encoding="utf-8") as f:
        json.dump(consistency, f, ensure_ascii=False, indent=2)
    print(json.dumps(consistency, ensure_ascii=False, indent=2))

    # plot
    if len(result_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_df = result_df[result_df["status"] == "ok"]
        colors = ["tab:blue" if p else "lightgray" for p in plot_df["passed_min_n"]]
        ax.errorbar(
            plot_df["window_mid"], plot_df["separation"],
            yerr=[plot_df["separation"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["separation"]],
            fmt="o", ecolor="gray", capsize=3, color="tab:blue",
        )
        ax.axhline(overall["separation"], color="tab:red", linestyle="--", label="overall (unwindowed)")
        ax.set_xlabel(f"Window midpoint year (width={args.window_size}, step={args.step})")
        ax.set_ylabel("Separation statistic (source vs target)")
        ax.set_title(args.label, fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png_path = os.path.join(args.outdir, "era_stratified_separation.png")
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {png_path}")

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
