#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ha_na_correlation_permutation_bootstrap.py

目的
----
审稿意见 R1-M5 / R2-m3 / R4-M6：HA--NA 同一分离株配对的 target-major fraction
相关系数（如 0.959、0.151、-0.105 等）目前没有报告置换检验 P 值或置信区间，
无法判断相关系数本身、以及不同组合之间差异的统计学意义。

本脚本读取 build_ha_na_paired_model_fixed.py 的输出配对表（含
HA_target_major_fraction 和 NA_target_major_fraction 两列），对每个
组合分别计算：

1. Pearson 和 Spearman 相关系数。
2. 置换检验：打乱 NA 一侧与 HA 的配对关系后重新计算相关系数，重复
   n-permutation 次，得到双侧经验 P 值。
3. Bootstrap 置信区间：对配对（HA, NA）联合有放回重抽样，重新计算相关系数，
   重复 n-boot 次后取百分位数区间。
4. 配对散点图（PNG），并标注相关系数、置换 P 值、bootstrap CI 和配对数 n。

用法示例
--------
# 单个组合
python ha_na_correlation_permutation_bootstrap.py \
  --paired-csv /path/to/H5N1_artiodactyla_to_primates_paired.csv \
  --label "H5N1 artiodactyla_to_primates" \
  --outdir /path/to/out

# 批量：给一个 CSV，每行是一个组合的 paired 表路径与标签
python ha_na_correlation_permutation_bootstrap.py \
  --manifest /path/to/manifest.csv \
  --outdir /path/to/out
（manifest.csv 需要两列：paired_csv, label）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

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


def load_ha_na_pairs(paired_csv: str, ha_col: Optional[str], na_col: Optional[str]) -> np.ndarray:
    df = pd.read_csv(paired_csv)
    ha_col = ha_col or "HA_target_major_fraction"
    na_col = na_col or "NA_target_major_fraction"
    if ha_col not in df.columns or na_col not in df.columns:
        raise ValueError(
            f"Columns {ha_col!r} / {na_col!r} not found in {paired_csv}. "
            f"Available: {list(df.columns)}"
        )
    sub = df[[ha_col, na_col]].dropna()
    return sub[ha_col].to_numpy(dtype=float), sub[na_col].to_numpy(dtype=float)


def permutation_test_correlation(x: np.ndarray, y: np.ndarray, method: str, n_permutation: int, seed: int):
    rng = np.random.default_rng(seed)
    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
    observed_r = float(corr_func(x, y)[0])

    n = len(x)
    perm_r = np.empty(n_permutation)
    for i in range(n_permutation):
        y_perm = y[rng.permutation(n)]
        perm_r[i] = corr_func(x, y_perm)[0]

    # two-sided empirical p-value
    p = float((np.sum(np.abs(perm_r) >= abs(observed_r)) + 1) / (n_permutation + 1))
    return observed_r, p, perm_r


def bootstrap_ci_correlation(x: np.ndarray, y: np.ndarray, method: str, n_boot: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
    n = len(x)
    boot_r = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot_r[i] = np.nan
            continue
        boot_r[i] = corr_func(xb, yb)[0]
    valid = boot_r[np.isfinite(boot_r)]
    if len(valid) == 0:
        return np.nan, np.nan
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return lo, hi


def analyze_one_combo(paired_csv: str, label: str, outdir: str, ha_col, na_col,
                       n_permutation: int, n_boot: int, alpha: float, seed: int) -> dict:
    x, y = load_ha_na_pairs(paired_csv, ha_col, na_col)
    n_pairs = len(x)
    result = {"label": label, "paired_csv": paired_csv, "n_pairs": n_pairs}

    if n_pairs < 4:
        result["status"] = "too_few_pairs"
        return result

    for method in ("pearson", "spearman"):
        r_obs, p_perm, _ = permutation_test_correlation(x, y, method, n_permutation, seed)
        ci_lo, ci_hi = bootstrap_ci_correlation(x, y, method, n_boot, alpha, seed + 1)
        result[f"{method}_r"] = r_obs
        result[f"{method}_permutation_p"] = p_perm
        result[f"{method}_ci_low"] = ci_lo
        result[f"{method}_ci_high"] = ci_hi
    result["status"] = "ok"

    # scatter plot
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(x, y, s=14, alpha=0.5, edgecolor="none")
    ax.set_xlabel("HA target-major fraction")
    ax.set_ylabel("NA target-major fraction")
    ax.set_title(label, fontsize=10)
    txt = (
        f"Pearson r={result['pearson_r']:.3f} (P={result['pearson_permutation_p']:.4f})\n"
        f"95% CI [{result['pearson_ci_low']:.3f}, {result['pearson_ci_high']:.3f}]\n"
        f"n={n_pairs}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig_path = os.path.join(outdir, f"{safe_name(label)}_ha_na_scatter.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    result["scatter_png"] = fig_path

    return result


def main():
    ap = argparse.ArgumentParser(
        description="Permutation test + bootstrap CI for HA-NA target-major-fraction "
                     "correlation (addresses reviewer comments R1-M5 / R2-m3 / R4-M6)."
    )
    ap.add_argument("--paired-csv", default=None, help="Single combination's paired CSV (output of build_ha_na_paired_model_fixed.py)")
    ap.add_argument("--label", default=None, help="Label for the single combination, e.g. 'H5N1 artiodactyla_to_primates'")
    ap.add_argument("--manifest", default=None, help="CSV with columns paired_csv,label for batch mode")
    ap.add_argument("--ha-col", default=None, help="Override HA fraction column name")
    ap.add_argument("--na-col", default=None, help="Override NA fraction column name")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-permutation", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    jobs = []
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        for _, row in manifest.iterrows():
            jobs.append((str(row["paired_csv"]), str(row["label"])))
    elif args.paired_csv and args.label:
        jobs.append((args.paired_csv, args.label))
    else:
        raise SystemExit("Provide either --manifest, or both --paired-csv and --label.")

    results = []
    for paired_csv, label in jobs:
        res = analyze_one_combo(
            paired_csv, label, args.outdir, args.ha_col, args.na_col,
            args.n_permutation, args.n_boot, args.alpha, args.seed,
        )
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if k != "scatter_png"}, ensure_ascii=False, indent=2))

    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.outdir, "ha_na_correlation_permutation_bootstrap_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
