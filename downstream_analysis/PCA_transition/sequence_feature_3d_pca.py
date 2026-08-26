#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence-level 3D PCA for ARNLE-IAV feature matrices.

This script merges per-sequence model features with the corresponding master
tables and produces global and subtype-specific 3D PCA coordinates, plots,
and explained-variance tables.

If a master table is a filtered subset while the NumPy feature matrix still
contains the full sample set, provide --feature_ref_tables. The script will
map subset rows back to feature rows using identifiers such as accession,
title_aln, or title.

Input feature arrays must be sequence-level matrices with shape
(n_sequences, n_features).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    pyo = None
    PLOTLY_AVAILABLE = False


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
HOST_COLORS = {
    "aves": "#1f77b4",
    "artiodactyla": "#ff7f0e",
    "primates": "#2ca02c",
}
EXTRA_COLORS = [
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#bab0ab",
]

DEFAULT_MIN_SUBTYPE_SAMPLES = 30
DEFAULT_MIN_HOST_SAMPLES = 5
ROW_ID_CANDIDATES = ["feature_row", "orig_row", "row_id", "source_row", "attn_row", "idx", "index"]
KEY_CANDIDATE_GROUPS = [
    ["accession", "title_aln"],
    ["accession", "title"],
    ["title_aln"],
    ["accession"],
    ["title"],
]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_host(v: str) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return HOST_ALIASES.get(s, s)


def normalize_h(v: str) -> str:
    s = str(v or "").strip().upper()
    m = re.search(r"(H\d{1,2})", s)
    return m.group(1) if m else ""


def normalize_n(v: str) -> str:
    s = str(v or "").strip().upper()
    m = re.search(r"(N\d{1,2})", s)
    return m.group(1) if m else ""


def normalize_full_subtype(v: str) -> str:
    s = str(v or "").strip().upper()
    m = re.search(r"(H\d{1,2}N\d{1,2})", s)
    return m.group(1) if m else ""


def normalize_key_piece(v: str) -> str:
    s = str(v or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def make_join_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = [df[c].astype(str).map(normalize_key_piece) for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "||" + p
    return out


def infer_protein(df: pd.DataFrame, table_path: Path) -> str:
    if "protein" in df.columns:
        vals = [str(x).strip() for x in df["protein"].dropna().astype(str).tolist() if str(x).strip()]
        if vals:
            return vals[0].upper()
    stem = table_path.stem.upper()
    for p in ["PB2", "PB1", "PA", "HA", "NP", "NA", "NS1", "M", "M1"]:
        if stem.startswith(p):
            return p
    return table_path.stem.split("_")[0].upper()


def add_full_subtype_from_hn(df: pd.DataFrame, out_col: str = "_full_subtype_from_hn") -> pd.DataFrame:
    df = df.copy()
    hvals = df["H_subtype"].map(normalize_h) if "H_subtype" in df.columns else pd.Series([""] * len(df), index=df.index)
    nvals = df["N_subtype"].map(normalize_n) if "N_subtype" in df.columns else pd.Series([""] * len(df), index=df.index)
    df[out_col] = [f"{h}{n}" if h and n else "" for h, n in zip(hvals.tolist(), nvals.tolist())]
    return df


def resolve_subtype_col(df: pd.DataFrame, protein: str, subtype_mode: str, subtype_col: str) -> Tuple[Optional[str], str, str]:
    protein_up = protein.upper()

    if subtype_col:
        if subtype_col not in df.columns:
            raise ValueError(f"Specified --subtype_col={subtype_col} was not found. Available columns: {list(df.columns)}")
        return subtype_col, f"{subtype_col}_norm", f"manual:{subtype_col}"

    mode = (subtype_mode or "auto").strip().lower()
    if mode == "h":
        if "H_subtype" in df.columns:
            return "H_subtype", "H_subtype_norm", "forced:H_subtype"
        if "subtype_final" in df.columns:
            return "subtype_final", "subtype_final_norm", "forced:h->subtype_final_fallback"
        if "subtype" in df.columns:
            return "subtype", "subtype_norm", "forced:h->subtype_fallback"
        raise ValueError("H-subtype grouping was requested (--subtype_mode h), but no H_subtype, subtype_final, or subtype column was found.")
    if mode == "n":
        if "N_subtype" in df.columns:
            return "N_subtype", "N_subtype_norm", "forced:N_subtype"
        if "subtype_final" in df.columns:
            return "subtype_final", "subtype_final_norm", "forced:n->subtype_final_fallback"
        if "subtype" in df.columns:
            return "subtype", "subtype_norm", "forced:n->subtype_fallback"
        raise ValueError("N-subtype grouping was requested (--subtype_mode n), but no N_subtype, subtype_final, or subtype column was found.")
    if mode == "full":
        if "subtype_final" in df.columns:
            return "subtype_final", "subtype_final_norm", "forced:subtype_final"
        if "subtype" in df.columns:
            return "subtype", "subtype_norm", "forced:subtype"
        if "H_subtype" in df.columns and "N_subtype" in df.columns:
            return "_full_subtype_from_hn", "_full_subtype_from_hn_norm", "forced:H_subtype+N_subtype"
        return None, "subtype_norm", "forced:none"
    if mode != "auto":
        raise ValueError(f"Unsupported --subtype_mode: {subtype_mode}")

    if protein_up == "HA":
        if "H_subtype" in df.columns:
            return "H_subtype", "H_subtype_norm", "auto:HA->H_subtype"
        return None, "subtype_norm", "auto:HA->none"

    if protein_up == "NA":
        if "N_subtype" in df.columns:
            return "N_subtype", "N_subtype_norm", "auto:NA->N_subtype"
        if "subtype_final" in df.columns:
            return "subtype_final", "subtype_final_norm", "auto:NA->subtype_final"
        if "subtype" in df.columns:
            return "subtype", "subtype_norm", "auto:NA->subtype"
        return None, "subtype_norm", "auto:NA->none"

    if protein_up in {"PB2", "NP", "NS1", "PB1", "PA", "M", "M1"}:
        if "subtype_final" in df.columns:
            return "subtype_final", "subtype_final_norm", f"auto:{protein_up}->subtype_final"
        if "subtype" in df.columns:
            return "subtype", "subtype_norm", f"auto:{protein_up}->subtype"
        if "H_subtype" in df.columns and "N_subtype" in df.columns:
            return "_full_subtype_from_hn", "_full_subtype_from_hn_norm", f"auto:{protein_up}->H_subtype+N_subtype"
        if "H_subtype" in df.columns:
            return "H_subtype", "H_subtype_norm", f"auto:{protein_up}->H_subtype_fallback"
        if "N_subtype" in df.columns:
            return "N_subtype", "N_subtype_norm", f"auto:{protein_up}->N_subtype_fallback"
        return None, "subtype_norm", f"auto:{protein_up}->none"

    if "subtype_final" in df.columns:
        return "subtype_final", "subtype_final_norm", f"auto:{protein_up}->subtype_final"
    if "subtype" in df.columns:
        return "subtype", "subtype_norm", f"auto:{protein_up}->subtype"
    if "H_subtype" in df.columns:
        return "H_subtype", "H_subtype_norm", f"auto:{protein_up}->H_subtype"
    if "N_subtype" in df.columns:
        return "N_subtype", "N_subtype_norm", f"auto:{protein_up}->N_subtype"
    return None, "subtype_norm", f"auto:{protein_up}->none"


def normalize_subtype(df: pd.DataFrame, subtype_col: Optional[str], subtype_norm_col: str) -> pd.DataFrame:
    df = df.copy()
    if subtype_col == "_full_subtype_from_hn":
        df = add_full_subtype_from_hn(df, out_col=subtype_col)

    if not subtype_col:
        df[subtype_norm_col] = ""
        return df

    if subtype_col not in df.columns:
        raise ValueError(f"Grouping column {subtype_col} is not present in the input table. Available columns: {list(df.columns)}")

    if subtype_col == "H_subtype":
        df[subtype_norm_col] = df[subtype_col].map(normalize_h)
    elif subtype_col == "N_subtype":
        df[subtype_norm_col] = df[subtype_col].map(normalize_n)
    elif subtype_col in {"subtype", "subtype_final", "_full_subtype_from_hn"}:
        df[subtype_norm_col] = df[subtype_col].map(normalize_full_subtype)
        bad = df[subtype_norm_col].astype(str).eq("")
        if bad.any():
            df.loc[bad, subtype_norm_col] = df.loc[bad, subtype_col].astype(str).str.strip()
    else:
        df[subtype_norm_col] = df[subtype_col].astype(str).str.strip()
    return df


def choose_metadata_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "accession", "title", "title_aln", "host", "host_norm", "collection_date", "year", "month",
        "country", "protein", "subtype", "subtype_final", "H_subtype", "N_subtype",
        "Prob_AR", "Prob_PR", "Prob_AV",
    ]
    return [c for c in candidates if c in df.columns]


def fit_pca(features: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be two-dimensional; current shape={features.shape}")
    if features.shape[0] < 2:
        raise ValueError("At least two samples are required for PCA.")
    n_comp = min(n_components, features.shape[0], features.shape[1])
    if n_comp < 2:
        raise ValueError(f"Cannot compute at least two PCA components for shape={features.shape}")
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_
    singular = pca.singular_values_
    return pcs, explained, singular


def build_pca_df(df_meta: pd.DataFrame, features: np.ndarray, feature_rows: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pcs, explained, singular = fit_pca(features, n_components=3)
    out = df_meta.reset_index(drop=True).copy()
    if feature_rows is None:
        feature_rows = np.arange(len(out), dtype=int)
    out["feature_row"] = feature_rows.astype(int)
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out["PC3"] = pcs[:, 2] if pcs.shape[1] >= 3 else 0.0
    exp_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(explained))],
        "explained_variance_ratio": explained,
        "singular_value": singular,
    })
    return out, exp_df


def _scatter3d_by_group(ax, df: pd.DataFrame, color_col: str, title: str, subtitle: str = "") -> None:
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
        ax.scatter(tmp["PC1"], tmp["PC2"], tmp["PC3"], s=8, alpha=0.45, label=f"{label} (n={len(tmp)})", depthshade=False)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}")
    ax.legend(loc="best", fontsize=8)


def save_3d_plot(df: pd.DataFrame, color_col: str, out_png: Path, title: str, subtitle: str = "") -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _scatter3d_by_group(ax, df, color_col=color_col, title=title, subtitle=subtitle)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered_groups(df: pd.DataFrame, color_col: str) -> List[str]:
    if color_col == "host_norm":
        groups = [g for g in HOST_ORDER if g in set(df[color_col].astype(str))]
        extras = [g for g in df[color_col].astype(str).dropna().unique().tolist() if g not in groups]
        groups.extend(sorted(extras))
    else:
        groups = sorted([g for g in df[color_col].astype(str).dropna().unique().tolist() if g and g != "nan"])
    return groups


def _group_color_map(groups: List[str], color_col: str) -> Dict[str, str]:
    cmap: Dict[str, str] = {}
    extra_i = 0
    for g in groups:
        if color_col == "host_norm" and g in HOST_COLORS:
            cmap[g] = HOST_COLORS[g]
        else:
            cmap[g] = EXTRA_COLORS[extra_i % len(EXTRA_COLORS)]
            extra_i += 1
    return cmap


def save_3d_plot_html(df: pd.DataFrame, color_col: str, out_html: Path, title: str, subtitle: str = "") -> None:
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is not installed; interactive HTML 3D PCA output is unavailable.")

    groups = _ordered_groups(df, color_col)
    color_map = _group_color_map(groups, color_col)
    traces = []
    hover_cols = [c for c in ["accession", "title_aln", "title", "host", "host_norm", "subtype", "subtype_final", "H_subtype", "N_subtype", "feature_row"] if c in df.columns]
    customdata = df[hover_cols].astype(str).to_numpy() if hover_cols else None

    for g in groups:
        mask = df[color_col].astype(str) == str(g)
        tmp = df.loc[mask].copy()
        if len(tmp) == 0:
            continue
        label = HOST_LABELS.get(g, g) if color_col == "host_norm" else g
        cd = tmp[hover_cols].astype(str).to_numpy() if hover_cols else None
        hovertemplate = (
            "<b>%s</b><br>PC1=%%{x:.4f}<br>PC2=%%{y:.4f}<br>PC3=%%{z:.4f}" % label
        )
        for i, c in enumerate(hover_cols):
            hovertemplate += "<br>%s=%%{customdata[%d]}" % (c, i)
        hovertemplate += "<extra></extra>"
        traces.append(
            go.Scatter3d(
                x=tmp["PC1"],
                y=tmp["PC2"],
                z=tmp["PC3"],
                mode="markers",
                name=f"{label} (n={len(tmp)})",
                marker=dict(size=3, opacity=0.6, color=color_map[g]),
                customdata=cd,
                hovertemplate=hovertemplate,
            )
        )

    full_title = title if not subtitle else f"{title}<br><sup>{subtitle}</sup>"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=full_title,
        template="plotly_white",
        width=1000,
        height=800,
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    pyo.plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def make_summary_json(protein: str, subtype_col: Optional[str], subtype_strategy: str, n_rows: int, feature_dim: int,
                      host_counts: Dict[str, int], subtype_counts: Dict[str, int], explained_variance: List[float],
                      alignment_info: Dict[str, object], interactive_html: bool) -> Dict[str, object]:
    return {
        "protein": protein,
        "subtype_col": subtype_col,
        "subtype_strategy": subtype_strategy,
        "n_rows": int(n_rows),
        "feature_dim": int(feature_dim),
        "host_counts": host_counts,
        "subtype_counts": subtype_counts,
        "explained_variance_ratio": explained_variance,
        "alignment_info": alignment_info,
        "interactive_html": bool(interactive_html),
    }


def align_by_row_id(df: pd.DataFrame, features: np.ndarray) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, object]]]:
    for col in ROW_ID_CANDIDATES:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            continue
        rows = vals.astype(int).to_numpy()
        if (rows < 0).any() or (rows >= features.shape[0]).any():
            continue
        if len(set(rows.tolist())) != len(rows):
            continue
        return (
            df.reset_index(drop=True).copy(),
            features[rows],
            rows,
            {
                "mode": "row_id",
                "row_id_col": col,
                "matched_rows": int(len(rows)),
                "feature_rows_min": int(rows.min()) if len(rows) else None,
                "feature_rows_max": int(rows.max()) if len(rows) else None,
            },
        )
    return None


def align_by_reference_table(df: pd.DataFrame, ref_df: pd.DataFrame, features: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(ref_df) != features.shape[0]:
        raise ValueError(f"Reference-table row count does not match the feature matrix: ref_table={len(ref_df)}, features={features.shape[0]}.")

    ref_df = ref_df.reset_index(drop=True).copy()
    ref_df["_feature_row"] = np.arange(len(ref_df), dtype=int)

    for cols in KEY_CANDIDATE_GROUPS:
        if not all(c in df.columns for c in cols):
            continue
        if not all(c in ref_df.columns for c in cols):
            continue

        left_key = make_join_key(df, cols)
        right_key = make_join_key(ref_df, cols)

        left_nonempty = left_key.astype(str) != ""
        right_nonempty = right_key.astype(str) != ""
        if left_nonempty.sum() == 0 or right_nonempty.sum() == 0:
            continue

        left_counts = left_key[left_nonempty].value_counts()
        right_counts = right_key[right_nonempty].value_counts()
        valid_keys = set(left_counts[left_counts == 1].index) & set(right_counts[right_counts == 1].index)
        if not valid_keys:
            continue

        left_tmp = df.copy()
        right_tmp = ref_df.copy()
        left_tmp["_join_key"] = left_key
        right_tmp["_join_key"] = right_key

        left_tmp = left_tmp[left_tmp["_join_key"].isin(valid_keys)].copy()
        if len(left_tmp) != len(df):
            continue

        right_tmp = right_tmp[right_tmp["_join_key"].isin(valid_keys)][["_join_key", "_feature_row"]].copy()
        merged = left_tmp.merge(right_tmp, on="_join_key", how="left", sort=False)
        merged = merged.dropna(subset=["_feature_row"]).copy()

        if len(merged) != len(df):
            continue

        rows = merged["_feature_row"].astype(int).to_numpy()
        if len(set(rows.tolist())) != len(rows):
            continue

        aligned_df = merged.drop(columns=["_join_key", "_feature_row"]).reset_index(drop=True)
        return (
            aligned_df,
            features[rows],
            rows,
            {
                "mode": "reference_table",
                "match_cols": cols,
                "matched_rows": int(len(rows)),
                "feature_rows_min": int(rows.min()) if len(rows) else None,
                "feature_rows_max": int(rows.max()) if len(rows) else None,
            },
        )

    raise ValueError(
        "Unable to map the current master_table back to feature-matrix rows."
        "Confirm that --feature_ref_tables contains the full master table used to generate the .npy file,"
        "and that the two tables share at least one unique key such as accession, title_aln, or title."
    )


def align_df_and_features(df: pd.DataFrame, features: np.ndarray, feature_ref_table: Optional[Path]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(df) == features.shape[0]:
        rows = np.arange(len(df), dtype=int)
        return df.reset_index(drop=True).copy(), features, rows, {"mode": "direct", "matched_rows": int(len(rows))}

    row_id_result = align_by_row_id(df, features)
    if row_id_result is not None:
        return row_id_result

    if feature_ref_table is None:
        raise ValueError(
            f"Row-count mismatch: master_table has {len(df)} rows, but the feature matrix has {features.shape[0]} rows."
            "The current table also lacks original-row columns such as feature_row or orig_row, so automatic alignment is not possible."
            "Provide --feature_ref_tables containing the full master table corresponding to the .npy file."
        )

    ref_df = pd.read_csv(feature_ref_table, dtype=str).fillna("")
    return align_by_reference_table(df, ref_df, features)


def run_one(master_table: Path, attn_npy: Path, out_root: Path, subtype_mode: str, subtype_col: str,
            min_subtype_samples: int, min_host_samples: int, protein_override: str,
            save_unknown_hosts: bool, feature_ref_table: Optional[Path], interactive_html: bool) -> None:
    df = pd.read_csv(master_table, dtype=str).fillna("")
    features = np.load(attn_npy)
    if features.ndim != 2:
        raise ValueError(f"{attn_npy} is not a two-dimensional feature matrix; shape={features.shape}")

    df, features_aligned, feature_rows, align_info = align_df_and_features(df, features, feature_ref_table)

    protein = (protein_override or "").strip().upper() or infer_protein(df, master_table)
    df["host_norm"] = df["host"].map(normalize_host) if "host" in df.columns else ""

    subtype_col_resolved, subtype_norm_col, subtype_strategy = resolve_subtype_col(
        df=df,
        protein=protein,
        subtype_mode=subtype_mode,
        subtype_col=subtype_col,
    )
    df = normalize_subtype(df, subtype_col_resolved, subtype_norm_col)

    meta_cols = choose_metadata_cols(df)
    extra_cols = [c for c in ["host_norm", subtype_norm_col] if c not in meta_cols]
    df_meta = df[meta_cols + extra_cols].copy()

    protein_dir = out_root / protein
    safe_mkdir(protein_dir)

    pca_df, exp_df = build_pca_df(df_meta, features_aligned, feature_rows=feature_rows)
    pca_df.to_csv(protein_dir / f"{protein}_global_pca_coordinates.csv", index=False)
    exp_df.to_csv(protein_dir / f"{protein}_global_pca_explained_variance.csv", index=False)

    subtitle = "; ".join([f"{row['component']}={row['explained_variance_ratio']:.4f}" for _, row in exp_df.iterrows()])
    save_3d_plot(
        pca_df,
        color_col="host_norm",
        out_png=protein_dir / f"{protein}_global_pca_3d_by_host.png",
        title=f"{protein} global 3D PCA by host",
        subtitle=subtitle,
    )
    if interactive_html:
        save_3d_plot_html(
            pca_df,
            color_col="host_norm",
            out_html=protein_dir / f"{protein}_global_pca_3d_by_host_interactive.html",
            title=f"{protein} global 3D PCA by host",
            subtitle=subtitle,
        )

    if subtype_col_resolved:
        subtype_values = [x for x in pca_df[subtype_norm_col].astype(str).dropna().unique().tolist() if x and x != "nan"]
        subtype_values = sorted(subtype_values)
    else:
        subtype_values = []

    subtype_summary_rows = []
    subtype_out_dir = protein_dir / "subtypes"
    safe_mkdir(subtype_out_dir)

    for st in subtype_values:
        tmp = pca_df[pca_df[subtype_norm_col].astype(str) == str(st)].copy().reset_index(drop=True)
        if len(tmp) < min_subtype_samples:
            subtype_summary_rows.append({"subtype": st, "n_rows": len(tmp), "status": f"skip_n_lt_{min_subtype_samples}"})
            continue

        host_counts = tmp["host_norm"].astype(str).value_counts().to_dict()
        host_ok = sum(int(v) >= min_host_samples for v in host_counts.values())
        if host_ok < 2:
            subtype_summary_rows.append({
                "subtype": st,
                "n_rows": len(tmp),
                "status": f"skip_host_groups_lt2_with_min_{min_host_samples}",
            })
            continue

        st_dir = subtype_out_dir / st
        safe_mkdir(st_dir)
        tmp.to_csv(st_dir / f"{protein}_{st}_pca_coordinates.csv", index=False)
        save_3d_plot(
            tmp,
            color_col="host_norm",
            out_png=st_dir / f"{protein}_{st}_pca_3d_by_host.png",
            title=f"{protein} {st} 3D PCA by host",
            subtitle=f"n={len(tmp)}",
        )
        if interactive_html:
            save_3d_plot_html(
                tmp,
                color_col="host_norm",
                out_html=st_dir / f"{protein}_{st}_pca_3d_by_host_interactive.html",
                title=f"{protein} {st} 3D PCA by host",
                subtitle=f"n={len(tmp)}",
            )
        subtype_summary_rows.append({
            "subtype": st,
            "n_rows": len(tmp),
            "status": "ok",
            **{f"host_{k}": v for k, v in host_counts.items()},
        })

    pd.DataFrame(subtype_summary_rows).to_csv(protein_dir / f"{protein}_subtype_summary.csv", index=False)

    host_counts_global = pca_df["host_norm"].astype(str).value_counts().to_dict() if "host_norm" in pca_df.columns else {}
    subtype_counts_global = pca_df[subtype_norm_col].astype(str).value_counts().to_dict() if subtype_norm_col in pca_df.columns else {}
    summary = make_summary_json(
        protein=protein,
        subtype_col=subtype_col_resolved,
        subtype_strategy=subtype_strategy,
        n_rows=len(pca_df),
        feature_dim=features_aligned.shape[1],
        host_counts=host_counts_global,
        subtype_counts=subtype_counts_global,
        explained_variance=exp_df["explained_variance_ratio"].tolist(),
        alignment_info=align_info,
        interactive_html=interactive_html,
    )
    with open(protein_dir / f"{protein}_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_unknown_hosts:
        known = set(HOST_ORDER)
        unk = pca_df.loc[~pca_df["host_norm"].astype(str).isin(known), :].copy()
        if len(unk) > 0:
            unk.to_csv(protein_dir / f"{protein}_unknown_host_rows.csv", index=False)

    print(f"[OK] {protein}: rows={len(pca_df)}, feature_dim={features_aligned.shape[1]}, subtype_col={subtype_col_resolved}, strategy={subtype_strategy}, align={align_info}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequence-level 3D PCA with optional subset-to-full feature-row mapping.")
    p.add_argument("--master_tables", nargs="+", required=True, help="One or more master_table CSV files")
    p.add_argument("--attn_npys", nargs="+", required=True, help="Feature .npy files corresponding one-to-one with --master_tables")
    p.add_argument("--feature_ref_tables", nargs="*", default=[], help="Optional full master tables used to generate the .npy files when the input master tables are filtered subsets.")
    p.add_argument("--out_root", required=True, help="Output directory")
    p.add_argument("--subtype_mode", default="auto", choices=["auto", "h", "n", "full"], help="auto/h/n/full")
    p.add_argument("--subtype_col", default="", help="Manually specify the grouping column; overrides --subtype_mode")
    p.add_argument("--min_subtype_samples", type=int, default=DEFAULT_MIN_SUBTYPE_SAMPLES, help=f"Minimum samples required for a subtype panel; default {DEFAULT_MIN_SUBTYPE_SAMPLES}")
    p.add_argument("--min_host_samples", type=int, default=DEFAULT_MIN_HOST_SAMPLES, help=f"Minimum samples per host group within a subtype panel; default {DEFAULT_MIN_HOST_SAMPLES}")
    p.add_argument("--protein_names", nargs="*", default=[], help="Optional protein names; the number must match the number of input tables.")
    p.add_argument("--save_unknown_hosts", action="store_true", help="Save rows with unmapped host labels")
    p.add_argument("--no_interactive_html", action="store_true", help="Disable interactive HTML 3D PCA output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.master_tables) != len(args.attn_npys):
        raise ValueError("--master_tables and --attn_npys must contain the same number of files.")
    if args.protein_names and len(args.protein_names) != len(args.master_tables):
        raise ValueError("If --protein_names is provided, its length must match the number of input tables.")
    if args.feature_ref_tables and len(args.feature_ref_tables) != len(args.master_tables):
        raise ValueError("If --feature_ref_tables is provided, its length must match the number of input tables.")

    out_root = Path(args.out_root)
    safe_mkdir(out_root)

    if not args.no_interactive_html and not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Install plotly or use --no_interactive_html to generate static PNG output only.")

    for i, (mt, npy) in enumerate(zip(args.master_tables, args.attn_npys)):
        protein_override = args.protein_names[i] if args.protein_names else ""
        feature_ref_table = Path(args.feature_ref_tables[i]) if args.feature_ref_tables else None
        run_one(
            master_table=Path(mt),
            attn_npy=Path(npy),
            out_root=out_root,
            subtype_mode=args.subtype_mode,
            subtype_col=args.subtype_col,
            min_subtype_samples=args.min_subtype_samples,
            min_host_samples=args.min_host_samples,
            protein_override=protein_override,
            save_unknown_hosts=args.save_unknown_hosts,
            feature_ref_table=feature_ref_table,
            interactive_html=not args.no_interactive_html,
        )

    print("Done.")


if __name__ == "__main__":
    main()
