#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
balanced_core_band_permutation_test.py

目的
----
审稿意见 R1-M1 / R4-M1 / R4-m2：Figure 3 的候选过渡带统计支持（随机标签置换 +
阈值敏感性）只覆盖了几何规则（geometric rule），而 Figure 4 及位点频率分析实际
使用的是另一套"平衡核规则"（balanced-core rule，定义见
middle_band_site_trajectory_analysis_v4.py），后者此前没有做过同等的统计检验。

本脚本为 balanced-core 规则补上：
1. 随机标签置换检验（负对照）——打乱来源/目标宿主标签后重新定义 balanced-core
   三层分层，重新计算带结构统计量，得到经验 P 值。
2. 阈值敏感性分析——在 core_keep_fraction / source_upper_q / target_lower_q /
   core_min_side_n 等参数的多组取值下，比较候选样本集合的稳定性（Jaccard 指数），
   与 figure3_transition_band_quantification_svg.py 的敏感性分析报告格式一致，
   便于两套规则并排比较。

统计量定义
----------
balanced-core 规则本身不像几何规则那样有正交距离阈值，因此这里参照
figure3_transition_band_quantification_svg.py 的思路，采用一个信噪比风格的
"轴分离度"统计量：

    separation = (target 侧样本在投影轴上的均值 - source 侧样本投影均值)
                 / 两组投影值的合并标准差

而不是直接比较三层分层与投影位置的等级相关——后者由于 balanced-core 分层本身
是"先按投影排序、再切三等分"得到的，任何轴（无论是否携带真实宿主信号）都会给出
接近 1 的等级相关，无法区分真实结构与置换背景，因此不适合作为置换检验的统计量。
置换检验中，每次置换后会连同来源/目标质心一起重新计算投影轴（而不是复用观测轴），
这一点与 figure3 脚本的 permutation_negative_control 一致，是保证检验有效性的关键。

用法示例
--------
python balanced_core_band_permutation_test.py \
  --pca-csv /path/to/pairwise_pca_coordinates.csv \
  --source-host artiodactyla --target-host primates \
  --outdir /path/to/out/NP_H1N1_artiodactyla_to_primates \
  --n-permutation 1000 --seed 2026 \
  --do-sensitivity

输入表要求：包含宿主列（自动从常见列名中推断）和至少两列 PCA 坐标（默认尝试
PC1/PC2/PC3，可用 --pc-cols 手动指定）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Column inference (mirrors house style in figure3_transition_band_quantification_svg.py)
# -----------------------------

def infer_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(f"Cannot infer {label} column. Tried: {', '.join(candidates)}")


def infer_host_col(df: pd.DataFrame) -> str:
    return infer_column(
        df,
        ["host", "host_class", "host_group", "true_host", "true_label", "label", "host_label"],
        "host",
    )


def infer_pc_cols(df: pd.DataFrame, user_cols: Optional[List[str]]) -> List[str]:
    if user_cols:
        missing = [c for c in user_cols if c not in df.columns]
        if missing:
            raise ValueError(f"User-specified PC columns not found: {missing}")
        return user_cols
    candidates = [c for c in ["PC1", "PC2", "PC3"] if c in df.columns]
    if len(candidates) < 2:
        raise ValueError("Could not find at least two of PC1/PC2/PC3; pass --pc-cols explicitly.")
    return candidates


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


# -----------------------------
# Balanced-core band construction
# (mirrors the algorithm in middle_band_site_trajectory_analysis_v4.py)
# -----------------------------

@dataclass
class BandParams:
    source_upper_q: float = 0.90
    target_lower_q: float = 0.10
    core_keep_fraction: float = 1.0
    core_min_side_n: int = 20


def project_onto_axis(coords: np.ndarray, source_centroid: np.ndarray, target_centroid: np.ndarray) -> np.ndarray:
    axis = target_centroid - source_centroid
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Source and target centroids coincide; cannot define an axis.")
    axis_unit = axis / norm
    return (coords - source_centroid) @ axis_unit


def build_balanced_core_layers(
    projection: np.ndarray,
    is_source: np.ndarray,
    is_target: np.ndarray,
    params: BandParams,
) -> Optional[np.ndarray]:
    """Return an array of layer codes (0/1/2) for samples inside the balanced-core
    band, and NaN for samples outside it. Mirrors v4's two-stage construction:
    an initial tail-quantile transition band, narrowed to an equal-count core
    around its midpoint."""
    source_proj = projection[is_source]
    target_proj = projection[is_target]
    if len(source_proj) < 5 or len(target_proj) < 5:
        return None

    lower = np.quantile(source_proj, params.source_upper_q)
    upper = np.quantile(target_proj, params.target_lower_q)
    if lower >= upper:
        # degenerate: source and target tails already overlap/cross
        lo, hi = min(lower, upper), max(lower, upper)
    else:
        lo, hi = lower, upper

    in_band = (projection >= lo) & (projection <= hi)
    band_idx = np.where(in_band)[0]
    if len(band_idx) < 2 * params.core_min_side_n:
        return None

    mid = (lo + hi) / 2.0
    band_proj = projection[band_idx]
    left_idx = band_idx[band_proj <= mid]
    right_idx = band_idx[band_proj > mid]
    n_side = int(min(len(left_idx), len(right_idx)) * params.core_keep_fraction)
    n_side = max(n_side, params.core_min_side_n)
    if len(left_idx) < params.core_min_side_n or len(right_idx) < params.core_min_side_n:
        return None

    # keep samples closest to the midpoint on each side
    left_sorted = left_idx[np.argsort(-projection[left_idx])]  # closest to mid first (largest, since <= mid)
    right_sorted = right_idx[np.argsort(projection[right_idx])]  # closest to mid first (smallest, since > mid)
    left_core = left_sorted[:n_side]
    right_core = right_sorted[:n_side]
    core_idx = np.concatenate([left_core, right_core])

    # quantile-balanced 3-way split within the core, ordered by projection
    core_proj = projection[core_idx]
    order = np.argsort(core_proj)
    core_idx_ordered = core_idx[order]
    n = len(core_idx_ordered)
    third = n // 3
    layer_of_core = np.empty(n, dtype=int)
    layer_of_core[:third] = 0
    layer_of_core[third:2 * third] = 1
    layer_of_core[2 * third:] = 2

    layers = np.full(projection.shape[0], np.nan)
    layers[core_idx_ordered] = layer_of_core
    return layers


def band_order_score(projection: np.ndarray, layers: np.ndarray, is_source: np.ndarray, is_target: np.ndarray) -> Tuple[float, int]:
    """Observed statistic for the permutation test.

    NOTE: layer codes (0/1/2) are constructed by *sorting* samples along the
    projection axis and cutting into thirds, so Spearman(layer, projection)
    would be close to 1 for ANY axis, real or random — that statistic cannot
    distinguish signal from noise and is not used here. Instead we use a
    signal-to-noise style separation ratio of the axis itself: how far apart
    the source/target centroids are, relative to the pooled within-group
    spread along that axis. A genuinely structured source->target axis
    should show much larger separation than an axis built from two random
    subsets of the same pool (which will sit close to the shared centroid).
    """
    mask = ~np.isnan(layers)
    n = int(mask.sum())
    if n < 6:
        return np.nan, n
    source_proj = projection[is_source]
    target_proj = projection[is_target]
    if len(source_proj) < 2 or len(target_proj) < 2:
        return np.nan, n
    pooled_std = np.sqrt((np.var(source_proj, ddof=1) + np.var(target_proj, ddof=1)) / 2.0)
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return np.nan, n
    separation = (np.mean(target_proj) - np.mean(source_proj)) / pooled_std
    return float(separation), n


# -----------------------------
# Permutation test
# -----------------------------

def run_permutation_test(
    coords: np.ndarray,
    host: np.ndarray,
    source_host: str,
    target_host: str,
    params: BandParams,
    n_permutation: int,
    seed: int,
) -> dict:
    """Permutation null: for each iteration, shuffle which samples count as
    source/target, then RECOMPUTE the source->target axis, projection, and
    balanced-core layers from scratch using the shuffled labels (mirrors
    figure3_transition_band_quantification_svg.py's permutation_negative_control,
    which recomputes calculate_transition_geometry under each permutation rather
    than reusing the observed axis). This is essential: because balanced-core
    layers are constructed by sorting samples along the projection axis, reusing
    a fixed axis would make the order score close to 1 under any permutation and
    the test would be uninformative."""
    is_source = host == source_host
    is_target = host == target_host
    eligible = is_source | is_target
    eligible_idx = np.where(eligible)[0]
    n_source = int(is_source.sum())
    n_target = int(is_target.sum())
    if n_source < 5 or n_target < 5:
        return {
            "status": "insufficient_samples",
            "observed_score": np.nan,
            "n_in_band": 0,
            "n_permutation_valid": 0,
            "empirical_p": np.nan,
        }

    source_centroid = coords[is_source].mean(axis=0)
    target_centroid = coords[is_target].mean(axis=0)
    observed_projection = project_onto_axis(coords, source_centroid, target_centroid)
    layers = build_balanced_core_layers(observed_projection, is_source, is_target, params)
    if layers is None:
        return {
            "status": "insufficient_band",
            "observed_score": np.nan,
            "n_in_band": 0,
            "n_permutation_valid": 0,
            "empirical_p": np.nan,
        }
    observed_score, n_in_band = band_order_score(observed_projection, layers, is_source, is_target)

    rng = np.random.default_rng(seed)
    perm_scores = []
    for _ in range(n_permutation):
        shuffled = rng.permutation(eligible_idx)
        perm_is_source = np.zeros(coords.shape[0], dtype=bool)
        perm_is_target = np.zeros(coords.shape[0], dtype=bool)
        perm_is_source[shuffled[:n_source]] = True
        perm_is_target[shuffled[n_source:n_source + n_target]] = True

        perm_source_centroid = coords[perm_is_source].mean(axis=0)
        perm_target_centroid = coords[perm_is_target].mean(axis=0)
        try:
            perm_projection = project_onto_axis(coords, perm_source_centroid, perm_target_centroid)
        except ValueError:
            continue
        perm_layers = build_balanced_core_layers(perm_projection, perm_is_source, perm_is_target, params)
        if perm_layers is None:
            continue
        score, _ = band_order_score(perm_projection, perm_layers, perm_is_source, perm_is_target)
        if np.isfinite(score):
            perm_scores.append(score)

    perm_scores = np.array(perm_scores, dtype=float)
    n_valid = len(perm_scores)
    if n_valid == 0 or not np.isfinite(observed_score):
        empirical_p = np.nan
    else:
        # one-sided: is the observed order score higher than random background?
        empirical_p = float((np.sum(perm_scores >= observed_score) + 1) / (n_valid + 1))

    return {
        "status": "ok",
        "observed_score": observed_score,
        "n_in_band": n_in_band,
        "n_permutation_requested": n_permutation,
        "n_permutation_valid": n_valid,
        "permutation_score_mean": float(np.mean(perm_scores)) if n_valid else np.nan,
        "permutation_score_p95": float(np.quantile(perm_scores, 0.95)) if n_valid else np.nan,
        "empirical_p": empirical_p,
    }


# -----------------------------
# Sensitivity analysis
# -----------------------------

DEFAULT_SENSITIVITY_GRID = {
    "source_upper_q": [0.85, 0.90, 0.95],
    "target_lower_q": [0.05, 0.10, 0.15],
    "core_keep_fraction": [0.7, 1.0],
}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def run_sensitivity_analysis(
    projection: np.ndarray,
    host: np.ndarray,
    source_host: str,
    target_host: str,
    default_params: BandParams,
    grid: dict,
) -> pd.DataFrame:
    is_source = host == source_host
    is_target = host == target_host

    default_layers = build_balanced_core_layers(projection, is_source, is_target, default_params)
    default_set = set(np.where(~np.isnan(default_layers))[0]) if default_layers is not None else set()

    rows = []
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in _cartesian(grid.values())]
    for combo in combos:
        params = BandParams(
            source_upper_q=combo.get("source_upper_q", default_params.source_upper_q),
            target_lower_q=combo.get("target_lower_q", default_params.target_lower_q),
            core_keep_fraction=combo.get("core_keep_fraction", default_params.core_keep_fraction),
            core_min_side_n=default_params.core_min_side_n,
        )
        layers = build_balanced_core_layers(projection, is_source, is_target, params)
        band_set = set(np.where(~np.isnan(layers))[0]) if layers is not None else set()
        score, n_in_band = band_order_score(projection, layers, is_source, is_target) if layers is not None else (np.nan, 0)
        row = dict(combo)
        row["n_in_band"] = n_in_band
        row["order_score"] = score
        row["jaccard_vs_default"] = jaccard(band_set, default_set)
        rows.append(row)
    return pd.DataFrame(rows)


def _cartesian(value_lists):
    import itertools
    return list(itertools.product(*value_lists))


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Permutation test and threshold sensitivity analysis for the "
                     "balanced-core transition-band rule (addresses reviewer comments "
                     "R1-M1 / R4-M1 / R4-m2)."
    )
    ap.add_argument("--pca-csv", required=True, help="CSV with per-sample PCA coordinates and host labels")
    ap.add_argument("--source-host", required=True)
    ap.add_argument("--target-host", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--host-col", default=None, help="Host column name (auto-inferred if omitted)")
    ap.add_argument("--pc-cols", nargs="+", default=None, help="PCA coordinate columns, e.g. PC1 PC2 PC3")
    ap.add_argument("--source-upper-q", type=float, default=0.90)
    ap.add_argument("--target-lower-q", type=float, default=0.10)
    ap.add_argument("--core-keep-fraction", type=float, default=1.0)
    ap.add_argument("--core-min-side-n", type=int, default=20)
    ap.add_argument("--n-permutation", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--do-sensitivity", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.pca_csv)
    host_col = args.host_col or infer_host_col(df)
    pc_cols = infer_pc_cols(df, args.pc_cols)

    host = df[host_col].astype(str).str.strip().str.lower().to_numpy()
    source_host = args.source_host.strip().lower()
    target_host = args.target_host.strip().lower()

    coords = df[pc_cols].to_numpy(dtype=float)
    is_source = host == source_host
    is_target = host == target_host
    if is_source.sum() < 5 or is_target.sum() < 5:
        raise SystemExit(
            f"Too few samples: source(n={int(is_source.sum())}) target(n={int(is_target.sum())}). "
            "Need at least 5 each to define centroids."
        )

    source_centroid = coords[is_source].mean(axis=0)
    target_centroid = coords[is_target].mean(axis=0)
    projection = project_onto_axis(coords, source_centroid, target_centroid)

    default_params = BandParams(
        source_upper_q=args.source_upper_q,
        target_lower_q=args.target_lower_q,
        core_keep_fraction=args.core_keep_fraction,
        core_min_side_n=args.core_min_side_n,
    )

    perm_result = run_permutation_test(
        coords, host, source_host, target_host,
        default_params, args.n_permutation, args.seed,
    )
    perm_result_out = dict(perm_result)
    perm_result_out.update({
        "source_host": source_host,
        "target_host": target_host,
        "rule": "balanced_core",
        "params": asdict(default_params),
    })
    perm_path = os.path.join(args.outdir, "balanced_core_permutation_test.json")
    with open(perm_path, "w", encoding="utf-8") as f:
        json.dump(perm_result_out, f, ensure_ascii=False, indent=2)
    print(json.dumps(perm_result_out, ensure_ascii=False, indent=2))

    if args.do_sensitivity:
        sens_df = run_sensitivity_analysis(
            projection, host, source_host, target_host, default_params, DEFAULT_SENSITIVITY_GRID
        )
        sens_path = os.path.join(args.outdir, "balanced_core_sensitivity.csv")
        sens_df.to_csv(sens_path, index=False)
        print(f"[sensitivity] wrote {len(sens_df)} parameter combinations to {sens_path}")
        print(f"[sensitivity] median Jaccard vs default: {sens_df['jaccard_vs_default'].median():.3f}")


if __name__ == "__main__":
    main()
