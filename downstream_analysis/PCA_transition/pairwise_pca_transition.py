#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise PCA transition analysis.

The script extracts a source/target host pair from a sequence-level PCA table,
computes source and target centroids, normalized projection along the
source-to-target axis, orthogonal distance to that axis, distance balance, and
transition-state labels. It writes pairwise coordinates, bridge/intermediate
samples, summary statistics, static plots, and optional interactive HTML plots.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


HOST_ALIASES = {
    "primate": "primates",
    "primates": "primates",
    "human": "primates",
    "humans": "primates",
    "homo sapiens": "primates",
    "artiodactyla": "artiodactyla",
    "swine": "artiodactyla",
    "pig": "artiodactyla",
    "pigs": "artiodactyla",
    "porcine": "artiodactyla",
    "sus scrofa": "artiodactyla",
    "anseriformes": "aves",
    "galliformes": "aves",
    "aves": "aves",
    "avian": "aves",
    "bird": "aves",
    "birds": "aves",
}

HOST_ORDER = ["aves", "artiodactyla", "primates"]
HOST_LABELS = {
    "aves": "Aves",
    "artiodactyla": "Artiodactyla",
    "primates": "Primates",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_host(v: str) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return HOST_ALIASES.get(s, s)


def parse_pair(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"--host_pair must use source:target format; received: {spec}")
    source, target = spec.split(":", 1)
    source = normalize_host(source)
    target = normalize_host(target)
    if source == target:
        raise ValueError("source_host and target_host must differ.")
    return source, target


def require_columns(df: pd.DataFrame, cols: List[str], file_label: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{file_label} is missing required columns: {miss}")


def compute_centroid(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("Cannot compute a centroid for an empty array.")
    return arr.mean(axis=0)


def assign_transition_group(
    approach_score: np.ndarray,
    axis_distance: np.ndarray,
    dist_balance_abs: np.ndarray,
    bridge_axis_q: float = 0.60,
    bridge_balance_q: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Layering logic:
    - source_like: approach_score < 0.2
    - bridge_like: 0.2 <= approach_score <= 0.8 with relatively small
      axis_distance and distance_balance_abs
    - target_core: 0.8 < approach_score <= 1.2
    - target_far: approach_score > 1.2
    bridge_candidate_flag applies a stricter median-based filter.
    """
    axis_thr = float(np.quantile(axis_distance, bridge_axis_q))
    balance_thr = float(np.quantile(dist_balance_abs, bridge_balance_q))

    groups = []
    bridge_flag = []

    bridge_axis_thr_strict = float(np.quantile(axis_distance, 0.50))
    bridge_balance_thr_strict = float(np.quantile(dist_balance_abs, 0.50))

    for a, ad, db in zip(approach_score, axis_distance, dist_balance_abs):
        if not np.isfinite(a):
            groups.append("unknown")
            bridge_flag.append(False)
            continue

        if a < 0.2:
            groups.append("source_like")
        elif a <= 0.8:
            if ad <= axis_thr and db <= balance_thr:
                groups.append("bridge_like")
            else:
                groups.append("axis_mid_offtrack")
        elif a <= 1.2:
            groups.append("target_core")
        else:
            groups.append("target_far")

        flag = bool((0.3 <= a <= 0.7) and (ad <= bridge_axis_thr_strict) and (db <= bridge_balance_thr_strict))
        bridge_flag.append(flag)

    return np.asarray(groups, dtype=object), np.asarray(bridge_flag, dtype=bool), axis_thr, balance_thr


def compute_pairwise_metrics(df: pd.DataFrame, source_host: str, target_host: str):
    work = df.copy()

    if "host_norm" not in work.columns:
        if "host" not in work.columns:
            raise ValueError("The input table contains neither host_norm nor host; host labels cannot be resolved.")
        work["host_norm"] = work["host"].map(normalize_host)

    require_columns(work, ["PC1", "PC2", "PC3", "host_norm"], "pca_coordinates")

    work["PC1"] = pd.to_numeric(work["PC1"], errors="coerce")
    work["PC2"] = pd.to_numeric(work["PC2"], errors="coerce")
    work["PC3"] = pd.to_numeric(work["PC3"], errors="coerce")
    work = work.dropna(subset=["PC1", "PC2", "PC3"]).reset_index(drop=True)

    pair = work[work["host_norm"].astype(str).isin([source_host, target_host])].copy().reset_index(drop=True)
    if len(pair) == 0:
        raise ValueError(f"No samples were found for host pair {source_host} vs {target_host}.")

    source_mask = pair["host_norm"].astype(str).eq(source_host).to_numpy()
    target_mask = pair["host_norm"].astype(str).eq(target_host).to_numpy()

    n_source = int(source_mask.sum())
    n_target = int(target_mask.sum())
    if n_source == 0 or n_target == 0:
        raise ValueError(f"Insufficient samples for pairwise analysis: {source_host}={n_source}, {target_host}={n_target}.")

    coords = pair[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    source_coords = coords[source_mask]
    target_coords = coords[target_mask]

    source_centroid = compute_centroid(source_coords)
    target_centroid = compute_centroid(target_coords)

    axis_vec = target_centroid - source_centroid
    axis_len = float(np.linalg.norm(axis_vec))
    if not np.isfinite(axis_len) or axis_len <= 0:
        raise ValueError("Source and target centroids overlap; the transition axis cannot be defined.")
    axis_unit = axis_vec / axis_len

    rel = coords - source_centroid[None, :]
    proj_len = rel @ axis_unit
    proj_point = source_centroid[None, :] + np.outer(proj_len, axis_unit)
    axis_distance = np.linalg.norm(coords - proj_point, axis=1)

    approach_score = proj_len / axis_len
    dist_to_source = np.linalg.norm(coords - source_centroid[None, :], axis=1)
    dist_to_target = np.linalg.norm(coords - target_centroid[None, :], axis=1)
    dist_balance_abs = np.abs(dist_to_source - dist_to_target)

    transition_group, bridge_candidate_flag, bridge_axis_thr, bridge_balance_thr = assign_transition_group(
        approach_score=approach_score,
        axis_distance=axis_distance,
        dist_balance_abs=dist_balance_abs,
        bridge_axis_q=0.60,
        bridge_balance_q=0.60,
    )

    pair["distance_to_source"] = dist_to_source
    pair["distance_to_target"] = dist_to_target
    pair["approach_score"] = approach_score
    pair["axis_distance"] = axis_distance
    pair["distance_balance_abs"] = dist_balance_abs
    pair["transition_group"] = transition_group
    pair["bridge_candidate_flag"] = bridge_candidate_flag

    # Retained for compatibility with the earlier workflow.
    pair["intermediate_flag"] = pair["transition_group"].astype(str).isin(["bridge_like", "axis_mid_offtrack"])

    summary = {
        "source_host": source_host,
        "target_host": target_host,
        "n_rows": int(len(pair)),
        "n_source": n_source,
        "n_target": n_target,
        "source_centroid": source_centroid.tolist(),
        "target_centroid": target_centroid.tolist(),
        "axis_length": axis_len,
        "bridge_axis_distance_threshold": bridge_axis_thr,
        "bridge_distance_balance_threshold": bridge_balance_thr,
        "bridge_candidate_n": int(pair["bridge_candidate_flag"].sum()),
        "transition_group_counts": pair["transition_group"].astype(str).value_counts().to_dict(),
        "approach_score_summary": {
            "min": float(np.min(approach_score)),
            "q25": float(np.quantile(approach_score, 0.25)),
            "median": float(np.quantile(approach_score, 0.50)),
            "q75": float(np.quantile(approach_score, 0.75)),
            "max": float(np.max(approach_score)),
        },
    }
    return pair, summary


def _resolve_color_group_col(df: pd.DataFrame, color_by: str) -> str:
    c = str(color_by or "host").strip().lower()
    if c == "host":
        return "host_norm"
    if c == "subtype":
        for cand in ["subtype_plot", "subtype_final", "subtype", "H_subtype", "N_subtype"]:
            if cand in df.columns:
                return cand
        raise ValueError("color_by=subtype was requested, but no subtype_final, subtype, H_subtype, or N_subtype column was found.")
    if color_by not in df.columns:
        raise ValueError(f"Specified color_by column does not exist: {color_by}")
    return color_by


def save_static_plot(df: pd.DataFrame, out_png: Path, title: str, color_by: str = "host", highlight_bridge: bool = True) -> None:
    color_col = _resolve_color_group_col(df, color_by)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if color_col == "host_norm":
        groups = [g for g in HOST_ORDER if g in set(df[color_col].astype(str))]
        extras = [g for g in df[color_col].astype(str).dropna().unique().tolist() if g not in groups]
        groups.extend(sorted(extras))
    else:
        groups = sorted([g for g in df[color_col].astype(str).dropna().unique().tolist() if g and g != "nan"])

    for g in groups:
        tmp = df[df[color_col].astype(str) == str(g)]
        if len(tmp) == 0:
            continue
        label = HOST_LABELS.get(g, g) if color_col == "host_norm" else g
        ax.scatter(
            tmp["PC1"], tmp["PC2"], tmp["PC3"],
            s=10,
            alpha=0.22 if highlight_bridge else 0.45,
            label=f"{label} (n={len(tmp)})",
            depthshade=False,
        )

    if highlight_bridge and "bridge_candidate_flag" in df.columns:
        mid = df[df["bridge_candidate_flag"] == True]
        if len(mid) > 0:
            ax.scatter(
                mid["PC1"], mid["PC2"], mid["PC3"],
                s=34,
                alpha=0.95,
                marker="o",
                label=f"Bridge candidate (n={len(mid)})",
                depthshade=False,
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_distance_plot(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bg = df[df["bridge_candidate_flag"] != True]
    mid = df[df["bridge_candidate_flag"] == True]

    ax.scatter(
        bg["distance_to_source"], bg["distance_to_target"],
        s=12, alpha=0.35, label=f"Background (n={len(bg)})"
    )
    if len(mid) > 0:
        ax.scatter(
            mid["distance_to_source"], mid["distance_to_target"],
            s=28, alpha=0.9, label=f"Bridge candidate (n={len(mid)})"
        )

    ax.set_xlabel("Distance to source centroid")
    ax.set_ylabel("Distance to target centroid")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_approach_hist(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["approach_score"], bins=40)
    ax.axvline(0.2, linestyle="--", linewidth=1.0)
    ax.axvline(0.8, linestyle="--", linewidth=1.0)
    ax.axvline(1.2, linestyle="--", linewidth=1.0)
    ax.set_xlabel("approach_score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_interactive_plot(df: pd.DataFrame, out_html: Path, title: str, color_by: str = "host", highlight_bridge: bool = True) -> None:
    try:
        import plotly.express as px
    except Exception as e:
        raise RuntimeError(f"Plotly is required for interactive HTML output: {e}")

    color_col = _resolve_color_group_col(df, color_by)

    hover_cols = {}
    for c in [
        "accession", "title", "title_aln", "host", "host_norm", "subtype", "subtype_final",
        "H_subtype", "N_subtype", "feature_row", "approach_score",
        "distance_to_source", "distance_to_target", "axis_distance",
        "distance_balance_abs", "transition_group", "bridge_candidate_flag"
    ]:
        if c in df.columns:
            hover_cols[c] = True

    plot_df = df.copy()
    if highlight_bridge and "bridge_candidate_flag" in plot_df.columns:
        plot_df["_point_role"] = np.where(plot_df["bridge_candidate_flag"], "Bridge candidate", "Background")
    else:
        plot_df["_point_role"] = "Background"

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_col,
        symbol="_point_role" if "_point_role" in plot_df.columns else None,
        hover_data=hover_cols,
        title=title,
        opacity=0.60,
    )
    fig.update_traces(marker=dict(size=4))
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract a host pair from PCA coordinates and perform transition analysis.")
    p.add_argument("--pca_csv", required=True, help="Input PCA coordinate table, for example *_pca_coordinates.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--host_pair", required=True, help="Host pair in source:target format, for example artiodactyla:primates")
    p.add_argument("--color_by", default="host", help="Color field for static/interactive plots: host, subtype, or any column name")
    p.add_argument("--title", default="", help="Plot title")
    p.add_argument("--interactive_html", action="store_true", help="Generate an interactive HTML plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_host, target_host = parse_pair(args.host_pair)

    pca_csv = Path(args.pca_csv)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    df = pd.read_csv(pca_csv, dtype=str).fillna("")
    for c in ["PC1", "PC2", "PC3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pair_df, summary = compute_pairwise_metrics(df, source_host=source_host, target_host=target_host)

    stem = pca_csv.stem
    prefix = f"{stem}_{source_host}_to_{target_host}"

    pair_df.to_csv(out_dir / f"{prefix}_pairwise_coordinates.csv", index=False)
    pair_df.loc[pair_df["bridge_candidate_flag"] == True].to_csv(
        out_dir / f"{prefix}_bridge_candidates.csv", index=False
    )
    pair_df.loc[pair_df["transition_group"].astype(str).isin(["bridge_like", "axis_mid_offtrack"])].to_csv(
        out_dir / f"{prefix}_intermediate_like_samples.csv", index=False
    )

    with open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    title = args.title.strip() or f"{stem} | {source_host} -> {target_host}"
    save_static_plot(
        pair_df,
        out_png=out_dir / f"{prefix}_pairwise_3d.png",
        title=title,
        color_by=args.color_by,
        highlight_bridge=True,
    )
    save_distance_plot(
        pair_df,
        out_png=out_dir / f"{prefix}_distance_scatter.png",
        title=f"{title} | centroid distances",
    )
    save_approach_hist(
        pair_df,
        out_png=out_dir / f"{prefix}_approach_hist.png",
        title=f"{title} | approach_score",
    )

    if args.interactive_html:
        save_interactive_plot(
            pair_df,
            out_html=out_dir / f"{prefix}_pairwise_3d_interactive.html",
            title=title,
            color_by=args.color_by,
            highlight_bridge=True,
        )

    print(f"[OK] pairwise done: {prefix}")
    print(f"source={summary['source_host']} n={summary['n_source']}")
    print(f"target={summary['target_host']} n={summary['n_target']}")
    print(f"bridge_candidate_n={summary['bridge_candidate_n']}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
