#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transition-band candidate-site trajectory analysis.

The workflow defines an initial transition band from source and target
projection tails, optionally constructs a balanced core band around the
geometric midpoint, divides the balanced band into source-near, center, and
target-near strata, and summarizes residue trajectories for ranked candidate
sites. It can also export projection-ordered aligned FASTA files for external
inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_version(acc: str) -> str:
    s = "" if pd.isna(acc) else str(acc).strip()
    return s.split(".", 1)[0] if s else s


def find_site_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cs = str(c).strip()
        if cs.isdigit():
            cols.append(cs)
    cols.sort(key=lambda x: int(x))
    return cols


def clean_residue(x) -> str:
    if pd.isna(x):
        return "X"
    s = str(x).strip()
    if not s:
        return "X"
    return s[0] if len(s) > 1 else s


def ordered_sequence_from_row(row: pd.Series, site_cols: List[str]) -> str:
    return "".join(clean_residue(row[c]) for c in site_cols)


def normalize_host_col(df: pd.DataFrame) -> pd.Series:
    for c in ["host_norm", "host"]:
        if c in df.columns:
            return df[c].astype(str).str.strip()
    return pd.Series([""] * len(df), index=df.index)


def parse_bayes_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Bayes matrix looks invalid: {path}")
    aa_col = df.columns[0]
    df = df.rename(columns={aa_col: "AA"})
    df["AA"] = df["AA"].astype(str).str.strip()
    df = df.set_index("AA")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def major_aa_from_bayes(bayes_df: pd.DataFrame, position: str, allow_gap_major: bool = False) -> Tuple[str, float]:
    if position not in bayes_df.columns:
        return ("", np.nan)
    s = bayes_df[position].dropna().copy()
    if s.empty:
        return ("", np.nan)
    if not allow_gap_major:
        s = s.loc[[idx for idx in s.index if idx != "-"]]
        if s.empty:
            return ("-", np.nan)
    aa = str(s.idxmax())
    val = float(s.max())
    return aa, val


def residue_freq(series: pd.Series, residue: str) -> float:
    if residue == "" or len(series) == 0:
        return np.nan
    return float((series == residue).sum()) / float(len(series))


def value_counts_freq(series: pd.Series) -> Dict[str, float]:
    if len(series) == 0:
        return {}
    vc = series.value_counts(dropna=False)
    total = float(len(series))
    return {str(k): float(v) / total for k, v in vc.items()}


def classify_trajectory_pattern(s: float, c: float, t: float, min_delta: float, jump_ratio: float) -> str:
    vals = [s, c, t]
    if any(pd.isna(v) for v in vals):
        return "insufficient_data"

    delta = t - s
    monotonic = (s <= c <= t)
    if delta < min_delta:
        return "weak_or_flat"

    left = c - s
    right = t - c

    if monotonic:
        if left > 0 and right > 0:
            if right > left * jump_ratio:
                return "late_jump_to_target_major"
            if left > right * jump_ratio:
                return "early_jump_to_target_major"
            return "smooth_gradient_to_target_major"
        if left <= 0 < right:
            return "late_jump_to_target_major"
        if left > 0 >= right:
            return "early_jump_to_target_major"
        return "monotonic_increase"

    if t > s:
        return "mixed_increase"
    return "no_clear_target_major_increase"


def write_fasta(df: pd.DataFrame, site_cols: List[str], out_fa: Path, header_cols: List[str]) -> None:
    with out_fa.open("w", encoding="utf-8") as fw:
        for _, row in df.iterrows():
            header_parts = []
            for c in header_cols:
                if c in row.index:
                    header_parts.append(f"{c}={row[c]}")
            header = " | ".join(header_parts)
            seq = ordered_sequence_from_row(row, site_cols)
            fw.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fw.write(seq[i:i+60] + "\n")


def require_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def resolve_meta_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def compute_projection_geometry(df: pd.DataFrame, source_host: str, target_host: str, host_col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pc_cols = ["PC1", "PC2", "PC3"]
    require_columns(df, pc_cols, "merged dataframe")

    source_df = df.loc[df[host_col] == source_host].copy()
    target_df = df.loc[df[host_col] == target_host].copy()
    if len(source_df) == 0 or len(target_df) == 0:
        raise ValueError("After merge, source or target host has zero samples. Cannot compute centroids.")

    cs = source_df[pc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0).values.astype(float)
    ct = target_df[pc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0).values.astype(float)
    axis = ct - cs
    axis_norm_sq = float(np.dot(axis, axis))
    axis_norm = float(np.sqrt(axis_norm_sq))
    if not np.isfinite(axis_norm_sq) or axis_norm_sq <= 0:
        raise ValueError("Source and target centroids are identical or invalid; cannot define projection axis.")

    X = df[pc_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    XC = X - cs
    proj_t = np.dot(XC, axis) / axis_norm_sq
    proj_point = cs + np.outer(proj_t, axis)
    orth_dist = np.sqrt(np.sum((X - proj_point) ** 2, axis=1))

    out = df.copy()
    out["projection_t"] = proj_t
    out["projection_orthogonal_distance"] = orth_dist
    out["projection_axis_distance_from_source_centroid"] = proj_t * axis_norm
    out["projection_axis_distance_from_target_centroid"] = (1.0 - proj_t) * axis_norm

    geom = {
        "source_centroid_PC1": float(cs[0]),
        "source_centroid_PC2": float(cs[1]),
        "source_centroid_PC3": float(cs[2]),
        "target_centroid_PC1": float(ct[0]),
        "target_centroid_PC2": float(ct[1]),
        "target_centroid_PC3": float(ct[2]),
        "axis_norm": axis_norm,
        "axis_norm_sq": axis_norm_sq,
    }
    return out, geom


def quantile_safe(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def decide_transition_band(
    merged: pd.DataFrame,
    host_col: str,
    source_host: str,
    target_host: str,
    source_upper_q: float,
    target_lower_q: float,
    band_expand_fraction: float = 0.0,
    manual_low: float | None = None,
    manual_high: float | None = None,
) -> Dict[str, float]:
    src_t = pd.to_numeric(
        merged.loc[merged[host_col] == source_host, "projection_t"], errors="coerce"
    ).dropna()
    tgt_t = pd.to_numeric(
        merged.loc[merged[host_col] == target_host, "projection_t"], errors="coerce"
    ).dropna()

    if len(src_t) == 0 or len(tgt_t) == 0:
        raise ValueError("Cannot define transition band because one host has zero projection_t values.")

    src_upper = quantile_safe(src_t, source_upper_q)
    tgt_lower = quantile_safe(tgt_t, target_lower_q)

    src_min = float(src_t.min())
    src_max = float(src_t.max())
    src_median = float(src_t.median())
    tgt_min = float(tgt_t.min())
    tgt_max = float(tgt_t.max())
    tgt_median = float(tgt_t.median())

    if manual_low is not None and manual_high is not None:
        low = float(manual_low)
        high = float(manual_high)
        mode = "manual_transition_band"
        relation = "manual"
    else:
        low = float(min(src_upper, tgt_lower))
        high = float(max(src_upper, tgt_lower))
        if src_upper < tgt_lower:
            relation = "gap_between_tails"
        elif src_upper > tgt_lower:
            relation = "overlap_between_tails"
        else:
            relation = "tail_touch_point"
        mode = "tail_overlap_or_gap_band"

    if band_expand_fraction > 0:
        width = high - low
        expand = width * float(band_expand_fraction)
        low -= expand
        high += expand

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(
            f"Invalid transition band after calculation: low={low}, high={high}. "
            f"Try manual bounds or adjust source_upper_q/target_lower_q."
        )

    return {
        "transition_low": float(low),
        "transition_high": float(high),
        "transition_mode": mode,
        "transition_relation": relation,
        "source_projection_min": src_min,
        "source_projection_median": src_median,
        "source_projection_max": src_max,
        "source_projection_upper_quantile_q": float(source_upper_q),
        "source_projection_upper_quantile_value": float(src_upper),
        "target_projection_min": tgt_min,
        "target_projection_median": tgt_median,
        "target_projection_max": tgt_max,
        "target_projection_lower_quantile_q": float(target_lower_q),
        "target_projection_lower_quantile_value": float(tgt_lower),
    }


def build_balanced_core_band(
    transition_all: pd.DataFrame,
    core_keep_fraction: float,
    min_side_n: int,
    disable_balanced_core: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if len(transition_all) < 3:
        raise ValueError("Too few samples in initial transition band to build balanced core band.")

    low = float(transition_all["projection_t"].min())
    high = float(transition_all["projection_t"].max())
    midpoint = (low + high) / 2.0

    left = transition_all.loc[transition_all["projection_t"] <= midpoint].sort_values("projection_t").copy()
    right = transition_all.loc[transition_all["projection_t"] > midpoint].sort_values("projection_t").copy()

    left_n = int(len(left))
    right_n = int(len(right))
    smaller_side_n = min(left_n, right_n)

    if disable_balanced_core or smaller_side_n < max(1, min_side_n):
        info = {
            "core_band_mode": "disabled_or_fallback_to_initial_band",
            "initial_transition_low": low,
            "initial_transition_high": high,
            "initial_transition_midpoint": midpoint,
            "initial_left_side_n": left_n,
            "initial_right_side_n": right_n,
            "smaller_side_n": smaller_side_n,
            "core_keep_fraction": float(core_keep_fraction),
            "balanced_side_k": smaller_side_n,
            "balanced_core_low": low,
            "balanced_core_high": high,
            "balanced_core_n": int(len(transition_all)),
            "balanced_left_kept_n": left_n,
            "balanced_right_kept_n": right_n,
        }
        return transition_all.copy(), info

    if not (0 < core_keep_fraction <= 1.0):
        raise ValueError("core_keep_fraction must be in (0,1]")

    k = max(min_side_n, int(np.floor(smaller_side_n * core_keep_fraction)))
    k = min(k, smaller_side_n)

    # keep the k samples closest to midpoint on each side
    left_keep = left.nlargest(k, "projection_t").copy()
    right_keep = right.nsmallest(k, "projection_t").copy()
    core_df = pd.concat([left_keep, right_keep], axis=0).sort_values(["projection_t", "projection_orthogonal_distance", "accession_norm"]).copy()

    info = {
        "core_band_mode": "balanced_core_from_midpoint",
        "initial_transition_low": low,
        "initial_transition_high": high,
        "initial_transition_midpoint": midpoint,
        "initial_left_side_n": left_n,
        "initial_right_side_n": right_n,
        "smaller_side_n": smaller_side_n,
        "core_keep_fraction": float(core_keep_fraction),
        "balanced_side_k": int(k),
        "balanced_core_low": float(core_df["projection_t"].min()),
        "balanced_core_high": float(core_df["projection_t"].max()),
        "balanced_core_n": int(len(core_df)),
        "balanced_left_kept_n": int(len(left_keep)),
        "balanced_right_kept_n": int(len(right_keep)),
    }
    return core_df, info


def assign_groups(df: pd.DataFrame, grouping_mode: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if len(df) < 3:
        raise ValueError("Too few samples in balanced core band to assign three groups.")

    out = df.copy()
    if grouping_mode == "quantile_balanced":
        q1 = float(out["projection_t"].quantile(1.0 / 3.0))
        q2 = float(out["projection_t"].quantile(2.0 / 3.0))
        def _assign(t: float) -> str:
            if t <= q1:
                return "transition_source_near"
            if t <= q2:
                return "transition_center"
            return "transition_target_near"
        out["transition_band_group"] = out["projection_t"].map(_assign)
        meta = {
            "grouping_mode": grouping_mode,
            "group_boundary_1": q1,
            "group_boundary_2": q2,
        }
        return out, meta

    if grouping_mode == "equal_width":
        low = float(out["projection_t"].min())
        high = float(out["projection_t"].max())
        span = high - low
        b1 = low + span / 3.0
        b2 = low + 2.0 * span / 3.0
        def _assign(t: float) -> str:
            if t <= b1:
                return "transition_source_near"
            if t <= b2:
                return "transition_center"
            return "transition_target_near"
        out["transition_band_group"] = out["projection_t"].map(_assign)
        meta = {
            "grouping_mode": grouping_mode,
            "group_boundary_1": b1,
            "group_boundary_2": b2,
        }
        return out, meta

    raise ValueError(f"Unknown grouping_mode: {grouping_mode}")


# -----------------------------
# Main analysis
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise_csv", required=True)
    ap.add_argument("--master_table_csv", required=True)
    ap.add_argument("--sorted_diff_csv", required=True)
    ap.add_argument("--source_bayes_csv", required=True)
    ap.add_argument("--target_bayes_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--source_host", default="artiodactyla")
    ap.add_argument("--target_host", default="primates")
    ap.add_argument("--top_n", type=int, default=40)
    ap.add_argument("--source_upper_q", type=float, default=0.90,
                    help="Upper-tail quantile of the source projection distribution; default 0.90")
    ap.add_argument("--target_lower_q", type=float, default=0.10,
                    help="Lower-tail quantile of the target projection distribution; default 0.10")
    ap.add_argument("--band_expand_fraction", type=float, default=0.0,
                    help="Relative expansion applied to both sides of the initial transition band, for example 0.10")
    ap.add_argument("--transition_low", type=float, default=None,
                    help="Manually set the initial transition-band lower bound; used with transition_high to override automatic bounds")
    ap.add_argument("--transition_high", type=float, default=None,
                    help="Manually set the initial transition-band upper bound; used with transition_low to override automatic bounds")
    ap.add_argument("--disable_balanced_core", action="store_true",
                    help="Disable the balanced core band and stratify directly within the initial transition band")
    ap.add_argument("--core_keep_fraction", type=float, default=1.0,
                    help="Fraction of samples retained from the smaller side when constructing the balanced core band; default 1.0")
    ap.add_argument("--core_min_side_n", type=int, default=20,
                    help="Minimum samples required on each side for the balanced core band; otherwise fall back to the initial band")
    ap.add_argument("--grouping_mode", choices=["quantile_balanced", "equal_width"], default="quantile_balanced",
                    help="Method for dividing the balanced core band into three strata; default quantile_balanced")
    ap.add_argument("--min_delta", type=float, default=0.10,
                    help="Minimum frequency difference used to define a meaningful target-major trajectory change")
    ap.add_argument("--jump_ratio", type=float, default=1.5,
                    help="Ratio threshold used to classify early and late jumps")
    ap.add_argument("--min_group_size", type=int, default=5,
                    help="Advisory minimum sample count per transition stratum; not a hard filter")
    ap.add_argument("--allow_gap_major", action="store_true",
                    help="Allow '-' to be selected as the major state in Bayesian summaries")
    args = ap.parse_args()

    if not (0.0 < args.source_upper_q < 1.0):
        raise ValueError("source_upper_q must be in (0,1)")
    if not (0.0 < args.target_lower_q < 1.0):
        raise ValueError("target_lower_q must be in (0,1)")
    if (args.transition_low is None) ^ (args.transition_high is None):
        raise ValueError("transition_low and transition_high must be provided together")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    pairwise = pd.read_csv(args.pairwise_csv)
    master = pd.read_csv(args.master_table_csv, low_memory=False)
    sorted_diff = pd.read_csv(args.sorted_diff_csv)
    source_bayes = parse_bayes_matrix(args.source_bayes_csv)
    target_bayes = parse_bayes_matrix(args.target_bayes_csv)

    require_columns(pairwise, ["accession", "PC1", "PC2", "PC3"], "pairwise_csv")
    require_columns(master, ["accession"], "master_table_csv")
    require_columns(sorted_diff, ["position"], "sorted_diff_csv")

    site_cols = find_site_columns(master)
    if not site_cols:
        raise ValueError("No numeric site columns found in master table.")

    pairwise = pairwise.copy()
    master = master.copy()
    pairwise["accession_raw"] = pairwise["accession"].astype(str).str.strip()
    master["accession_raw"] = master["accession"].astype(str).str.strip()
    pairwise["accession_norm"] = pairwise["accession_raw"].map(strip_version)
    master["accession_norm"] = master["accession_raw"].map(strip_version)

    pairwise_host = normalize_host_col(pairwise)
    pairwise = pairwise.loc[pairwise_host.isin([args.source_host, args.target_host])].copy()
    pairwise["host_norm_effective"] = normalize_host_col(pairwise)
    master["host_norm_effective"] = normalize_host_col(master)

    pair_debug = pairwise[[c for c in ["accession_raw", "accession_norm", "title", "title_aln", "host", "host_norm_effective", "approach_score", "PC1", "PC2", "PC3"] if c in pairwise.columns]].copy()
    master_debug = master[[c for c in ["accession_raw", "accession_norm", "title", "title_aln", "host", "host_norm_effective"] if c in master.columns]].copy()
    merge_debug = pair_debug.merge(master_debug, on="accession_norm", how="outer", suffixes=("_pairwise", "_master"), indicator=True)
    merge_debug.to_csv(out_dir / "merge_debug_accession_status.csv", index=False)

    merged = pairwise.merge(master, on="accession_norm", how="inner", suffixes=("_pairwise", "_master"))
    host_effective_col = resolve_meta_col(merged, ["host_norm_effective_pairwise", "host_norm_effective", "host_norm", "host_pairwise"])
    if not host_effective_col:
        raise ValueError("Cannot resolve effective host column after merge.")
    merged.to_csv(out_dir / "merged_pairwise_master_inner.csv", index=False)

    merged = merged.loc[merged[host_effective_col].isin([args.source_host, args.target_host])].copy()
    merged, geom = compute_projection_geometry(merged, args.source_host, args.target_host, host_effective_col)

    band = decide_transition_band(
        merged=merged,
        host_col=host_effective_col,
        source_host=args.source_host,
        target_host=args.target_host,
        source_upper_q=args.source_upper_q,
        target_lower_q=args.target_lower_q,
        band_expand_fraction=args.band_expand_fraction,
        manual_low=args.transition_low,
        manual_high=args.transition_high,
    )

    initial_low = float(band["transition_low"])
    initial_high = float(band["transition_high"])
    merged["initial_transition_band_flag"] = merged["projection_t"].between(initial_low, initial_high, inclusive="both")
    initial_transition_all = merged.loc[merged["initial_transition_band_flag"]].copy()
    if len(initial_transition_all) < 3:
        raise ValueError(
            f"Too few samples in initial transition band ({len(initial_transition_all)}). "
            f"Try widening the band with source_upper_q/target_lower_q or band_expand_fraction."
        )

    core_band_df, core_info = build_balanced_core_band(
        transition_all=initial_transition_all,
        core_keep_fraction=args.core_keep_fraction,
        min_side_n=args.core_min_side_n,
        disable_balanced_core=args.disable_balanced_core,
    )

    core_band_df, group_meta = assign_groups(core_band_df, args.grouping_mode)

    merged["transition_band_flag"] = False
    merged["transition_band_group"] = ""
    merged.loc[core_band_df.index, "transition_band_flag"] = True
    merged.loc[core_band_df.index, "transition_band_group"] = core_band_df["transition_band_group"]

    pairwise_with_geom, _ = compute_projection_geometry(pairwise.copy(), args.source_host, args.target_host, "host_norm_effective")
    core_low = float(core_band_df["projection_t"].min())
    core_high = float(core_band_df["projection_t"].max())
    pairwise_with_geom["transition_band_flag"] = pairwise_with_geom["projection_t"].between(core_low, core_high, inclusive="both")
    pairwise_trans = pairwise_with_geom.loc[pairwise_with_geom["transition_band_flag"]].copy()
    missing_trans_norm = set(pairwise_trans["accession_norm"]) - set(core_band_df["accession_norm"])
    pairwise_trans.loc[pairwise_trans["accession_norm"].isin(missing_trans_norm)].to_csv(out_dir / "transition_band_missing_from_master.csv", index=False)

    # geometry summary tables
    pd.DataFrame([
        {"centroid": args.source_host, "PC1": geom["source_centroid_PC1"], "PC2": geom["source_centroid_PC2"], "PC3": geom["source_centroid_PC3"]},
        {"centroid": args.target_host, "PC1": geom["target_centroid_PC1"], "PC2": geom["target_centroid_PC2"], "PC3": geom["target_centroid_PC3"]},
    ]).to_csv(out_dir / "projection_centroids.csv", index=False)

    projection_distribution_rows = []
    for grp_name, sub in [
        (f"full_{args.source_host}", merged.loc[merged[host_effective_col] == args.source_host]),
        (f"full_{args.target_host}", merged.loc[merged[host_effective_col] == args.target_host]),
        ("initial_transition_band", initial_transition_all),
        ("balanced_core_band", core_band_df),
    ]:
        t = pd.to_numeric(sub["projection_t"], errors="coerce").dropna()
        projection_distribution_rows.append({
            "group": grp_name,
            "n_samples": int(len(t)),
            "t_min": float(t.min()) if len(t) else np.nan,
            "t_q05": float(t.quantile(0.05)) if len(t) else np.nan,
            "t_q10": float(t.quantile(0.10)) if len(t) else np.nan,
            "t_q25": float(t.quantile(0.25)) if len(t) else np.nan,
            "t_median": float(t.median()) if len(t) else np.nan,
            "t_q75": float(t.quantile(0.75)) if len(t) else np.nan,
            "t_q90": float(t.quantile(0.90)) if len(t) else np.nan,
            "t_q95": float(t.quantile(0.95)) if len(t) else np.nan,
            "t_max": float(t.max()) if len(t) else np.nan,
        })
    pd.DataFrame(projection_distribution_rows).to_csv(out_dir / "projection_distribution_summary.csv", index=False)

    summary_rows = []
    year_col = resolve_meta_col(merged, ["year_pairwise", "year_master", "year"])
    country_col = resolve_meta_col(merged, ["country_pairwise", "country_master", "country"])
    axis_col = resolve_meta_col(merged, ["axis_distance", "projection_orthogonal_distance"])
    dsrc_col = resolve_meta_col(merged, ["distance_to_source", "projection_axis_distance_from_source_centroid"])
    dtgt_col = resolve_meta_col(merged, ["distance_to_target", "projection_axis_distance_from_target_centroid"])

    for grp_name, sub in [
        ("full_source_host", merged.loc[merged[host_effective_col] == args.source_host]),
        ("full_target_host", merged.loc[merged[host_effective_col] == args.target_host]),
        ("initial_transition_band", initial_transition_all),
        ("balanced_core_band", core_band_df),
        ("transition_source_near", core_band_df.loc[core_band_df["transition_band_group"] == "transition_source_near"]),
        ("transition_center", core_band_df.loc[core_band_df["transition_band_group"] == "transition_center"]),
        ("transition_target_near", core_band_df.loc[core_band_df["transition_band_group"] == "transition_target_near"]),
    ]:
        years = pd.to_numeric(sub[year_col], errors="coerce") if year_col else pd.Series(dtype=float)
        summary_rows.append({
            "group": grp_name,
            "n_samples": int(len(sub)),
            "host_counts": json.dumps(sub[host_effective_col].value_counts(dropna=False).to_dict(), ensure_ascii=False),
            "projection_t_min": float(sub["projection_t"].min()) if len(sub) else np.nan,
            "projection_t_median": float(sub["projection_t"].median()) if len(sub) else np.nan,
            "projection_t_max": float(sub["projection_t"].max()) if len(sub) else np.nan,
            "approach_score_median": pd.to_numeric(sub.get("approach_score", pd.Series(dtype=float)), errors="coerce").median() if len(sub) else np.nan,
            "orthogonal_distance_median": pd.to_numeric(sub[axis_col], errors="coerce").median() if (len(sub) and axis_col) else np.nan,
            "distance_to_source_median": pd.to_numeric(sub[dsrc_col], errors="coerce").median() if (len(sub) and dsrc_col) else np.nan,
            "distance_to_target_median": pd.to_numeric(sub[dtgt_col], errors="coerce").median() if (len(sub) and dtgt_col) else np.nan,
            "year_min": years.min() if len(years) else np.nan,
            "year_max": years.max() if len(years) else np.nan,
            "country_top10": json.dumps(sub[country_col].astype(str).value_counts().head(10).to_dict(), ensure_ascii=False) if country_col and len(sub) else json.dumps({}, ensure_ascii=False),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "transition_band_group_summary.csv", index=False)

    # Ordered band for MEGA
    transition_ordered = core_band_df.sort_values(["projection_t", "projection_orthogonal_distance", "accession_norm"]).copy()

    residue_preview_positions = []
    if "position" in sorted_diff.columns:
        residue_preview_positions = [str(int(x)) if not pd.isna(x) else "" for x in sorted_diff["position"].head(args.top_n).tolist()]
    residue_preview_positions = [p for p in residue_preview_positions if p in site_cols]

    mega_meta_cols = [
        "accession_norm",
        host_effective_col,
        "projection_t",
        "approach_score",
        "transition_band_group",
        "projection_orthogonal_distance",
        "axis_distance",
        "distance_to_source",
        "distance_to_target",
        "title_pairwise",
        "title_aln_master",
    ]
    mega_meta_cols = [c for c in mega_meta_cols if c in transition_ordered.columns]
    mega_meta = transition_ordered[mega_meta_cols].copy()
    rename_map = {
        "accession_norm": "accession",
        host_effective_col: "host",
        "title_pairwise": "title",
        "title_aln_master": "title_aln",
    }
    mega_meta = mega_meta.rename(columns=rename_map)
    for p in residue_preview_positions:
        mega_meta[f"site_{p}"] = transition_ordered[p].map(clean_residue)
    mega_meta.to_csv(out_dir / "transition_band_ordered_metadata.csv", index=False)

    fasta_header_cols = [c for c in [
        "accession_norm", host_effective_col, "projection_t", "approach_score", "transition_band_group",
        "projection_orthogonal_distance", "axis_distance", "distance_to_source", "distance_to_target"
    ] if c in transition_ordered.columns]
    write_fasta(transition_ordered, site_cols, out_dir / "transition_band_samples_ordered_for_MEGA.fasta", fasta_header_cols)
    transition_ordered.to_csv(out_dir / "transition_band_samples.csv", index=False)
    initial_transition_all.to_csv(out_dir / "initial_transition_band_samples.csv", index=False)

    # Candidate positions
    if "sort" not in sorted_diff.columns:
        sorted_diff["sort"] = np.nan
    sorted_diff = sorted_diff.copy()
    sorted_diff["position_str"] = sorted_diff["position"].apply(lambda x: str(int(x)) if not pd.isna(x) else "")
    top_df = sorted_diff.head(args.top_n).copy()
    top_df["rank"] = np.arange(1, len(top_df) + 1)

    missing_site_rows = []
    traj_rows = []

    group_map = {
        "full_source_host": merged.loc[merged[host_effective_col] == args.source_host],
        "full_target_host": merged.loc[merged[host_effective_col] == args.target_host],
        "initial_transition_band": initial_transition_all,
        "balanced_core_band": core_band_df,
        "transition_source_near": core_band_df.loc[core_band_df["transition_band_group"] == "transition_source_near"],
        "transition_center": core_band_df.loc[core_band_df["transition_band_group"] == "transition_center"],
        "transition_target_near": core_band_df.loc[core_band_df["transition_band_group"] == "transition_target_near"],
    }

    for _, r in top_df.iterrows():
        pos = r["position_str"]
        rank = int(r["rank"])
        sort_score = float(r["sort"]) if not pd.isna(r["sort"]) else np.nan
        if pos not in site_cols:
            missing_site_rows.append({
                "position": pos,
                "rank": rank,
                "sorted_diff_score": sort_score,
                "reason": "position_missing_in_master_table",
            })
            continue

        source_major_aa, source_major_prob = major_aa_from_bayes(source_bayes, pos, args.allow_gap_major)
        target_major_aa, target_major_prob = major_aa_from_bayes(target_bayes, pos, args.allow_gap_major)

        rec = {
            "position": pos,
            "rank": rank,
            "is_in_top20": rank <= 20,
            "sorted_diff_score": sort_score,
            "source_major_aa": source_major_aa,
            "source_major_bayes_prob": source_major_prob,
            "target_major_aa": target_major_aa,
            "target_major_bayes_prob": target_major_prob,
            "same_major_aa": source_major_aa == target_major_aa,
        }

        for grp_name, sub in group_map.items():
            residues = sub[pos].map(clean_residue) if pos in sub.columns else pd.Series(dtype=object)
            rec[f"{grp_name}_n"] = len(residues)
            rec[f"{grp_name}_source_major_freq"] = residue_freq(residues, source_major_aa)
            rec[f"{grp_name}_target_major_freq"] = residue_freq(residues, target_major_aa)
            rec[f"{grp_name}_top_residue_freqs_json"] = json.dumps(dict(sorted(value_counts_freq(residues).items(), key=lambda kv: kv[1], reverse=True)[:5]), ensure_ascii=False)

        s_freq = rec["transition_source_near_target_major_freq"]
        c_freq = rec["transition_center_target_major_freq"]
        t_freq = rec["transition_target_near_target_major_freq"]
        s_src = rec["transition_source_near_source_major_freq"]
        c_src = rec["transition_center_source_major_freq"]
        t_src = rec["transition_target_near_source_major_freq"]

        rec["delta_target_major_freq_transition"] = (t_freq - s_freq) if all(pd.notna([s_freq, t_freq])) else np.nan
        rec["delta_source_major_freq_transition"] = (s_src - t_src) if all(pd.notna([s_src, t_src])) else np.nan
        rec["monotonic_target_major_increase"] = bool(pd.notna(s_freq) and pd.notna(c_freq) and pd.notna(t_freq) and (s_freq <= c_freq <= t_freq))
        rec["monotonic_source_major_decrease"] = bool(pd.notna(s_src) and pd.notna(c_src) and pd.notna(t_src) and (s_src >= c_src >= t_src))
        rec["trajectory_pattern"] = classify_trajectory_pattern(s_freq, c_freq, t_freq, args.min_delta, args.jump_ratio)

        min_support = min(rec["transition_source_near_n"], rec["transition_center_n"], rec["transition_target_near_n"])
        rec["min_transition_group_n"] = min_support
        rec["support_ok"] = bool(min_support >= args.min_group_size)

        delta_t = rec["delta_target_major_freq_transition"] if pd.notna(rec["delta_target_major_freq_transition"]) else 0.0
        if rec["same_major_aa"]:
            traj_score = -1.0 + float(max(0.0, delta_t))
        else:
            traj_score = float(max(0.0, delta_t))
            if delta_t >= args.min_delta:
                if rec["monotonic_target_major_increase"]:
                    traj_score += 0.50
                if rec["monotonic_source_major_decrease"]:
                    traj_score += 0.20
                if rec["trajectory_pattern"] == "smooth_gradient_to_target_major":
                    traj_score += 0.20
                elif rec["trajectory_pattern"] in {"late_jump_to_target_major", "early_jump_to_target_major"}:
                    traj_score += 0.10
                if rec["support_ok"]:
                    traj_score += 0.10
        rec["trajectory_score"] = traj_score
        traj_rows.append(rec)

    pd.DataFrame(missing_site_rows).to_csv(out_dir / "missing_top_positions_in_master_table.csv", index=False)
    traj_df = pd.DataFrame(traj_rows)
    if len(traj_df) == 0:
        raise ValueError("No valid candidate positions remained after site matching.")

    traj_df.to_csv(out_dir / f"site_trajectory_top{args.top_n}.csv", index=False)

    ranked = traj_df.sort_values(
        by=["trajectory_score", "delta_target_major_freq_transition", "sorted_diff_score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked.to_csv(out_dir / f"site_trajectory_top{args.top_n}_ranked.csv", index=False)
    ranked_informative = ranked.loc[ranked["same_major_aa"] == False].copy()
    ranked_informative.to_csv(out_dir / f"site_trajectory_top{args.top_n}_ranked_informative.csv", index=False)

    heatmap_input_cols = [
        "position", "rank", "trajectory_score", "trajectory_pattern", "source_major_aa", "target_major_aa",
        "full_source_host_target_major_freq", "initial_transition_band_target_major_freq", "balanced_core_band_target_major_freq",
        "transition_source_near_target_major_freq", "transition_center_target_major_freq", "transition_target_near_target_major_freq",
        "full_target_host_target_major_freq",
        "full_source_host_source_major_freq", "initial_transition_band_source_major_freq", "balanced_core_band_source_major_freq",
        "transition_source_near_source_major_freq", "transition_center_source_major_freq", "transition_target_near_source_major_freq",
        "full_target_host_source_major_freq",
    ]
    heatmap_input_cols = [c for c in heatmap_input_cols if c in ranked.columns]
    ranked[heatmap_input_cols].to_csv(out_dir / "trajectory_heatmap_input.csv", index=False)

    summary = {
        "pairwise_csv": os.path.abspath(args.pairwise_csv),
        "master_table_csv": os.path.abspath(args.master_table_csv),
        "sorted_diff_csv": os.path.abspath(args.sorted_diff_csv),
        "source_bayes_csv": os.path.abspath(args.source_bayes_csv),
        "target_bayes_csv": os.path.abspath(args.target_bayes_csv),
        "source_host": args.source_host,
        "target_host": args.target_host,
        "top_n": args.top_n,
        "middle_definition": "balanced_core_transition_band_from_projection_tails",
        "source_upper_q": args.source_upper_q,
        "target_lower_q": args.target_lower_q,
        "band_expand_fraction": args.band_expand_fraction,
        "disable_balanced_core": bool(args.disable_balanced_core),
        "core_keep_fraction": args.core_keep_fraction,
        "core_min_side_n": args.core_min_side_n,
        "grouping_mode": args.grouping_mode,
        "transition_low": initial_low,
        "transition_high": initial_high,
        "balanced_core_low": core_low,
        "balanced_core_high": core_high,
        **group_meta,
        "n_pairwise_filtered": int(len(pairwise)),
        "n_master": int(len(master)),
        "n_merged_inner": int(len(merged)),
        "n_initial_transition_band_after_merge": int(len(initial_transition_all)),
        "n_transition_band_after_merge": int(len(core_band_df)),
        "n_transition_source_near": int((core_band_df["transition_band_group"] == "transition_source_near").sum()),
        "n_transition_center": int((core_band_df["transition_band_group"] == "transition_center").sum()),
        "n_transition_target_near": int((core_band_df["transition_band_group"] == "transition_target_near").sum()),
        "pairwise_only_accessions": int((merge_debug["_merge"] == "left_only").sum()),
        "master_only_accessions": int((merge_debug["_merge"] == "right_only").sum()),
        "both_accessions": int((merge_debug["_merge"] == "both").sum()),
        "top_positions_missing_in_master": int(len(missing_site_rows)),
        "valid_candidate_positions": int(len(traj_df)),
        **geom,
        **band,
        **core_info,
        "projection_t_full_source_median": float(merged.loc[merged[host_effective_col] == args.source_host, "projection_t"].median()),
        "projection_t_full_target_median": float(merged.loc[merged[host_effective_col] == args.target_host, "projection_t"].median()),
    }
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2, ensure_ascii=False)

    with (out_dir / "README_outputs.txt").open("w", encoding="utf-8") as fw:
        fw.write("Core output files for the balanced transition-band workflow\n")
        fw.write("=========================================\n")
        fw.write("projection_centroids.csv\n")
        fw.write("  Source and target centroid coordinates.\n\n")
        fw.write("projection_distribution_summary.csv\n")
        fw.write("  Projection summaries for source, target, initial band, and balanced core band.\n\n")
        fw.write("merge_debug_accession_status.csv\n")
        fw.write("  Accession-alignment status between pairwise and master tables.\n\n")
        fw.write("initial_transition_band_samples.csv\n")
        fw.write("  Samples in the initial transition band.\n\n")
        fw.write("transition_band_samples.csv\n")
        fw.write("  Samples in the balanced core band.\n\n")
        fw.write("transition_band_ordered_metadata.csv\n")
        fw.write("  Metadata for balanced-core samples ordered by projection_t.\n\n")
        fw.write("transition_band_samples_ordered_for_MEGA.fasta\n")
        fw.write("  Aligned FASTA for balanced-core samples ordered by projection_t.\n\n")
        fw.write(f"site_trajectory_top{args.top_n}_ranked_informative.csv\n")
        fw.write("  Prioritized candidate sites ranked by trajectory_score after removing same_major_aa sites.\n\n")
        fw.write("trajectory_heatmap_input.csv\n")
        fw.write("  Compact table suitable for downstream visualization.\n\n")
        fw.write("transition_band_group_summary.csv\n")
        fw.write("  Sample counts and host/country/year summaries for the initial band, balanced core band, and transition strata.\n")

    print("[DONE] analysis finished.")
    print(f"[DONE] outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
