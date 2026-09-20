#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
site_effect_size_ci_and_logistic_or.py

目的
----
审稿意见 R1-M4 / R1-m1：效应量（target-near 与 source-near 目标宿主常见氨基酸
比例之差）目前没有置信区间；logistic 趋势检验只报 trend_p 与 BH q，不报回归系数
β1 或比值比（OR）。本脚本读取与 site_trajectory_logistic_trend_fdr.py 完全相同
的 site-state 长表输入，对每个候选位点（analysis_label × position）额外计算：

1. 效应量的 bootstrap 置信区间——在 source-near / target-near 两层内分别做
   有放回重抽样，重新计算比例差，重复 n-boot 次后取百分位数区间。
2. logistic 回归系数 β1 的 Wald 95% 置信区间，以及对应的比值比
   OR = exp(β1) 及其置信区间 [exp(CI_low), exp(CI_high)]。

输出表以 (analysis_label, position) 为键，可直接与
site_trajectory_logistic_trend_fdr.py 的输出表合并（如 pandas merge），
不修改、不依赖对方脚本已保存的中间结果。

用法示例
--------
python site_effect_size_ci_and_logistic_or.py \
  --site-state /path/to/site_state_long.csv \
  --output /path/to/site_effect_ci_and_or.csv \
  --n-boot 2000 --alpha 0.05 --seed 2026

之后可用如下方式与原表合并：
    trend = pd.read_csv("candidate_sites_with_trend_fdr.csv")
    extra = pd.read_csv("site_effect_ci_and_or.csv")
    merged = trend.merge(extra, on=["analysis_label", "position"], how="left")
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: statsmodels is required. Install with: pip install statsmodels") from exc


LAYER_CODE = {
    "source-near": 0, "source_near": 0, "source near": 0, "source": 0, "near_source": 0,
    "center": 1, "centre": 1, "middle": 1, "mid": 1, "central": 1,
    "target-near": 2, "target_near": 2, "target near": 2, "target": 2, "near_target": 2,
}


def read_csv_keep_na(path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def require_columns(df: pd.DataFrame, required, table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}. Available: {list(df.columns)}")


def normalize_position(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_aa(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def layer_to_code(value) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1, 2) else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if int(value) in (0, 1, 2) else None
    text = str(value).strip().lower()
    return LAYER_CODE.get(text)


def clean_long_input(df: pd.DataFrame) -> pd.DataFrame:
    required = ["analysis_label", "sequence_id", "transition_layer", "position", "aa", "target_major_aa"]
    require_columns(df, required, "site-state long table")
    out = df.copy()
    out["analysis_label"] = out["analysis_label"].astype(str).str.strip()
    out["sequence_id"] = out["sequence_id"].astype(str).str.strip()
    out["position"] = out["position"].map(normalize_position)
    out["aa"] = out["aa"].map(normalize_aa)
    out["target_major_aa"] = out["target_major_aa"].map(normalize_aa)
    out["layer_code"] = out["transition_layer"].map(layer_to_code)
    bad = out["layer_code"].isna().sum()
    if bad:
        examples = out.loc[out["layer_code"].isna(), "transition_layer"].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"{bad} rows have unrecognized transition_layer values. Examples: {examples}")
    out["layer_code"] = out["layer_code"].astype(int)
    out = out[(out["analysis_label"] != "") & (out["sequence_id"] != "") & (out["position"] != "") & (out["aa"] != "")]
    return out


def bootstrap_effect_size_ci(binary_by_layer: dict, n_boot: int, alpha: float, rng: np.random.Generator):
    """binary_by_layer: {0: np.array of 0/1, 1: ..., 2: ...} — resample each
    layer independently (matches how the observed proportions are computed
    independently per layer) and recompute delta = freq(2) - freq(0)."""
    src = binary_by_layer.get(0)
    tgt = binary_by_layer.get(2)
    if src is None or tgt is None or len(src) == 0 or len(tgt) == 0:
        return np.nan, np.nan, np.nan, 0

    deltas = np.empty(n_boot)
    n_src, n_tgt = len(src), len(tgt)
    for b in range(n_boot):
        src_bs = src[rng.integers(0, n_src, n_src)]
        tgt_bs = tgt[rng.integers(0, n_tgt, n_tgt)]
        deltas[b] = tgt_bs.mean() - src_bs.mean()
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    se = float(np.std(deltas, ddof=1))
    return lo, hi, se, n_boot


def fit_logistic_with_ci(y: pd.Series, x: pd.Series, min_total: int, min_per_layer: int, alpha: float):
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)
    valid = y.notna() & x.notna()
    y, x = y[valid], x[valid]

    result = {
        "logistic_beta_refit": np.nan, "logistic_beta_ci_low": np.nan, "logistic_beta_ci_high": np.nan,
        "odds_ratio": np.nan, "odds_ratio_ci_low": np.nan, "odds_ratio_ci_high": np.nan,
        "logistic_refit_status": "not_attempted",
    }
    if len(y) < min_total:
        result["logistic_refit_status"] = "too_few_total_sequences"
        return result
    layer_counts = x.value_counts()
    for layer in (0, 1, 2):
        if int(layer_counts.get(layer, 0)) < min_per_layer:
            result["logistic_refit_status"] = "too_few_sequences_in_one_or_more_layers"
            return result
    if y.nunique(dropna=True) < 2 or x.nunique(dropna=True) < 2:
        result["logistic_refit_status"] = "no_variation"
        return result

    X = sm.add_constant(x.to_numpy(dtype=float), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = sm.Logit(y.to_numpy(dtype=float), X)
            res = model.fit(disp=False, maxiter=200)
        beta = float(res.params[1])
        ci = res.conf_int(alpha=alpha)
        ci_low, ci_high = float(ci[1][0]), float(ci[1][1])
        if not all(math.isfinite(v) for v in (beta, ci_low, ci_high)):
            result["logistic_refit_status"] = "non_finite_fit"
            return result
        # Heuristic guard against quasi-complete separation: statsmodels can
        # converge "successfully" to a huge, numerically unstable beta when a
        # layer has an extreme (near 0 or near 1) target-major proportion with
        # a small sample size. A Wald CI this wide on the log-odds scale is not
        # a usable estimate even though the fit nominally succeeded.
        WIDE_CI_THRESHOLD = 15.0  # log-odds scale; beta itself is usually |beta| < ~5 for real effects
        is_unstable = (ci_high - ci_low) > WIDE_CI_THRESHOLD or abs(beta) > WIDE_CI_THRESHOLD
        result.update({
            "logistic_beta_refit": beta,
            "logistic_beta_ci_low": ci_low,
            "logistic_beta_ci_high": ci_high,
            "odds_ratio": float(np.exp(beta)) if abs(beta) < 700 else float("inf"),
            "odds_ratio_ci_low": float(np.exp(ci_low)) if ci_low > -700 else 0.0,
            "odds_ratio_ci_high": float(np.exp(ci_high)) if ci_high < 700 else float("inf"),
            "logistic_refit_status": "unstable_wide_ci_likely_quasi_separation" if is_unstable else "ok",
        })
        return result
    except PerfectSeparationError:
        result["logistic_refit_status"] = "perfect_separation"
        return result
    except np.linalg.LinAlgError:
        result["logistic_refit_status"] = "singular_matrix"
        return result
    except Exception as exc:  # noqa: BLE001
        result["logistic_refit_status"] = f"fit_failed:{type(exc).__name__}"
        return result


def summarize_site_ci(group: pd.DataFrame, n_boot: int, alpha: float, min_total: int, min_per_layer: int, rng: np.random.Generator) -> dict:
    g = group.copy()
    g["target_major_binary"] = (g["aa"] == g["target_major_aa"]).astype(int)

    row = {
        "analysis_label": str(g["analysis_label"].iloc[0]),
        "position": normalize_position(g["position"].iloc[0]),
    }

    binary_by_layer = {
        code: g.loc[g["layer_code"] == code, "target_major_binary"].to_numpy()
        for code in (0, 1, 2)
    }
    lo, hi, se, n_boot_used = bootstrap_effect_size_ci(binary_by_layer, n_boot, alpha, rng)
    row["effect_size_delta_target_minus_source"] = (
        float(binary_by_layer[2].mean() - binary_by_layer[0].mean())
        if len(binary_by_layer[0]) and len(binary_by_layer[2]) else np.nan
    )
    row["effect_size_ci_low"] = lo
    row["effect_size_ci_high"] = hi
    row["effect_size_bootstrap_se"] = se
    row["effect_size_n_boot"] = n_boot_used

    row.update(fit_logistic_with_ci(
        y=g["target_major_binary"], x=g["layer_code"],
        min_total=min_total, min_per_layer=min_per_layer, alpha=alpha,
    ))
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap CI for site-level effect size, and logistic beta / "
                     "odds-ratio with Wald CI (addresses reviewer comments R1-M4 / R1-m1)."
    )
    ap.add_argument("--site-state", required=True, help="Same long-format site-state CSV used by site_trajectory_logistic_trend_fdr.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-total", type=int, default=30)
    ap.add_argument("--min-per-layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    site_state = read_csv_keep_na(args.site_state)
    long_df = clean_long_input(site_state)

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, group in long_df.groupby(["analysis_label", "position"], dropna=False, sort=True):
        rows.append(summarize_site_ci(group, args.n_boot, args.alpha, args.min_total, args.min_per_layer, rng))

    result = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    n_sites = len(result)
    n_ci_ok = int(result["effect_size_ci_low"].notna().sum())
    n_or_ok = int((result["logistic_refit_status"] == "ok").sum())
    print(f"Saved: {out_path}")
    print(f"Sites analyzed: {n_sites}")
    print(f"Effect-size bootstrap CI computed: {n_ci_ok}")
    print(f"Logistic OR + CI computed: {n_or_ok}")


if __name__ == "__main__":
    main()
