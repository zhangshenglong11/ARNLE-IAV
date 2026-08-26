#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transition-band support analysis for Figure 3.

Generate the following support outputs:
1. merged_pairwise_master_inner.csv
2. merged_pairwise_master_inner_with_projection.csv
3. permutation_negative_control_summary.csv
4. permutation_null_distribution.csv
5. transition_band_sensitivity_summary.csv

Implementation notes:
- Automatically handles host_pca / host_master / host_norm after pandas merge.
- Creates canonical columns: host, accession, title, year, country, protein, subtype, Prob_AR/PR/AV.
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd


ID_CANDIDATES = [
    "accession", "accession_pca", "accession_master",
    "protein_accession", "Protein Accession",
    "id", "ID", "seq_id", "sequence_id",
    "title", "title_pca", "title_master", "Title",
    "name", "merge_key"
]

HOST_CANDIDATES = [
    "host", "host_pca", "host_master", "host_norm",
    "Host", "Host_pca", "Host_master",
    "host_group", "host_label", "true_host", "true_label", "label"
]

PC_CANDIDATES = [
    ("PC1", "PC2", "PC3"),
    ("pc1", "pc2", "pc3"),
    ("PCA1", "PCA2", "PCA3"),
    ("pca1", "pca2", "pca3"),
]


def normalize_key(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if "|" in x:
        x = x.split("|")[0].strip()
    if x.startswith(">"):
        x = x[1:].strip()
    parts = x.split()
    if len(parts) > 0 and re.match(r"^[A-Z]{2,5}\d+", parts[0]):
        x = parts[0]
    x = re.sub(r"\s+", "", x)
    return x


def normalize_host_value(x):
    if pd.isna(x):
        return np.nan
    low = str(x).strip().lower()
    host_map = {
        "artiodactyla": "artiodactyla",
        "artiodactyl": "artiodactyla",
        "swine": "artiodactyla",
        "pig": "artiodactyla",
        "porcine": "artiodactyla",
        "sus scrofa": "artiodactyla",
        "cow": "artiodactyla",
        "bovine": "artiodactyla",
        "cattle": "artiodactyla",
        "primates": "primates",
        "primate": "primates",
        "human": "primates",
        "homo sapiens": "primates",
        "non-human primate": "primates",
        "nonhuman primate": "primates",
        "monkey": "primates",
        "aves": "aves",
        "avian": "aves",
        "bird": "aves",
        "chicken": "aves",
        "duck": "aves",
        "goose": "aves",
        "wild bird": "aves",
        "domestic bird": "aves",
    }
    return host_map.get(low, low)


def find_column(df, candidates, required=True, label="column"):
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]
    if required:
        raise ValueError(
            f"Cannot find {label}. Tried candidates: {candidates}. "
            f"Available columns: {cols}"
        )
    return None


def find_pc_columns(df):
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for trio in PC_CANDIDATES:
        found = []
        ok = True
        for c in trio:
            if c in cols:
                found.append(c)
            elif str(c).lower() in lower_map:
                found.append(lower_map[str(c).lower()])
            else:
                ok = False
                break
        if ok:
            return found
    raise ValueError(f"Cannot find PC1/PC2/PC3 columns. Available columns: {cols}")


def safe_read_csv(path, label="input"):
    if path is None:
        raise FileNotFoundError(f"{label} path is None.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def add_canonical_col(df, new_col, candidate_cols):
    if new_col in df.columns:
        return df
    for c in candidate_cols:
        if c in df.columns:
            df[new_col] = df[c]
            return df
    return df


def add_canonical_columns_after_merge(merged):
    merged = add_canonical_col(
        merged, "host",
        ["host_pca", "host_master", "host_norm", "Host_pca", "Host_master", "Host"]
    )
    merged = add_canonical_col(
        merged, "accession",
        ["accession_pca", "accession_master", "protein_accession_pca",
         "protein_accession_master", "id_pca", "id_master", "merge_key"]
    )
    merged = add_canonical_col(
        merged, "title",
        ["title_pca", "title_master", "Title_pca", "Title_master"]
    )
    merged = add_canonical_col(
        merged, "collection_date",
        ["collection_date_pca", "collection_date_master", "date_pca", "date_master"]
    )
    merged = add_canonical_col(merged, "year", ["year_pca", "year_master"])
    merged = add_canonical_col(merged, "month", ["month_pca", "month_master"])
    merged = add_canonical_col(merged, "country", ["country_pca", "country_master"])
    merged = add_canonical_col(merged, "protein", ["protein_pca", "protein_master"])
    merged = add_canonical_col(
        merged, "subtype",
        ["subtype_final", "subtype_pca", "subtype_master", "subtype_norm",
         "H_subtype_pca", "H_subtype_master", "fasta_subtype", "gp_subtype"]
    )
    merged = add_canonical_col(merged, "H_subtype", ["H_subtype_pca", "H_subtype_master"])
    merged = add_canonical_col(merged, "N_subtype", ["N_subtype_pca", "N_subtype_master", "N_subtype"])
    merged = add_canonical_col(merged, "Prob_AR", ["Prob_AR", "Prob_AR_pca", "Prob_AR_master"])
    merged = add_canonical_col(merged, "Prob_PR", ["Prob_PR", "Prob_PR_pca", "Prob_PR_master"])
    merged = add_canonical_col(merged, "Prob_AV", ["Prob_AV", "Prob_AV_pca", "Prob_AV_master"])

    if "host" not in merged.columns:
        raise ValueError(
            "No usable host column found after merge. Expected one of: "
            "host, host_pca, host_master, host_norm, Host."
        )
    merged["host"] = merged["host"].apply(normalize_host_value)
    return merged


def prepare_merge_key(df, preferred_col=None, table_name="table"):
    out = df.copy()
    if preferred_col is not None and preferred_col in out.columns:
        id_col = preferred_col
    elif "merge_key" in out.columns:
        id_col = "merge_key"
    else:
        id_col = find_column(out, ID_CANDIDATES, label=f"{table_name} ID column")
    out["merge_key"] = out[id_col].apply(normalize_key)
    if out["merge_key"].isna().all():
        raise ValueError(f"All merge_key values are NA in {table_name}. ID column used: {id_col}")
    return out, id_col


def compute_projection(df, pc_cols, host_col, source_host, target_host):
    data = df.copy()
    data[host_col] = data[host_col].apply(normalize_host_value)
    source_host = normalize_host_value(source_host)
    target_host = normalize_host_value(target_host)

    source_df = data[data[host_col].astype(str) == source_host]
    target_df = data[data[host_col].astype(str) == target_host]

    if len(source_df) == 0:
        available = data[host_col].value_counts(dropna=False).to_dict()
        raise ValueError(f"No source host samples found: {source_host}. Available host counts: {available}")
    if len(target_df) == 0:
        available = data[host_col].value_counts(dropna=False).to_dict()
        raise ValueError(f"No target host samples found: {target_host}. Available host counts: {available}")

    X = data[pc_cols].astype(float).values
    Xs = source_df[pc_cols].astype(float).values
    Xt = target_df[pc_cols].astype(float).values

    Cs = Xs.mean(axis=0)
    Ct = Xt.mean(axis=0)
    axis_vec = Ct - Cs
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0:
        raise ValueError("Source and target centroids are identical; cannot define PCA axis.")

    u = axis_vec / axis_len
    centered = X - Cs
    dot = centered @ u
    projection_score = dot / axis_len
    orthogonal_vec = centered - np.outer(dot, u)
    orthogonal_distance = np.linalg.norm(orthogonal_vec, axis=1)

    data["projection_score"] = projection_score
    data["orthogonal_distance"] = orthogonal_distance
    data["centroid_distance"] = axis_len
    data["axis_distance_recomputed"] = orthogonal_distance
    return data, axis_len


def extract_transition_band(df, q=0.15, trim=0.10, min_group_size=30):
    data = df.copy()
    for c in ["projection_score", "orthogonal_distance"]:
        if c not in data.columns:
            raise ValueError(f"Missing required column for transition-band extraction: {c}")

    in_axis = data[(data["projection_score"] > 0) & (data["projection_score"] < 1)].copy()
    if len(in_axis) == 0:
        empty = data.iloc[0:0].copy()
        return empty, {
            "n_transition_band": 0,
            "n_source_near": 0,
            "n_center": 0,
            "n_target_near": 0,
            "mean_orthogonal_distance": np.nan,
            "balance_score": 0.0,
            "band_score": 0.0,
            "orthogonal_threshold": np.nan,
            "passed_min_group_size": False,
        }

    orth_threshold = in_axis["orthogonal_distance"].quantile(q)
    band = in_axis[
        (in_axis["orthogonal_distance"] <= orth_threshold) &
        (in_axis["projection_score"] >= trim) &
        (in_axis["projection_score"] <= 1 - trim)
    ].copy()

    def assign_layer(t):
        if t < 1 / 3:
            return "source_near"
        elif t < 2 / 3:
            return "center"
        return "target_near"

    if len(band) > 0:
        band["transition_layer"] = band["projection_score"].apply(assign_layer)
    else:
        band["transition_layer"] = pd.Series(dtype=str)

    counts = band["transition_layer"].value_counts().to_dict()
    n_source_near = int(counts.get("source_near", 0))
    n_center = int(counts.get("center", 0))
    n_target_near = int(counts.get("target_near", 0))
    n_list = [n_source_near, n_center, n_target_near]
    balance_score = min(n_list) / max(n_list) if max(n_list) > 0 else 0.0
    passed_min_group_size = all(n >= min_group_size for n in n_list)
    mean_orth = float(band["orthogonal_distance"].mean()) if len(band) else np.nan

    if len(band) > 0 and pd.notna(orth_threshold) and orth_threshold > 0:
        compactness = max(0.0, 1.0 - mean_orth / orth_threshold)
    else:
        compactness = 0.0
    band_score = balance_score * compactness * np.log1p(len(band))

    metrics = {
        "n_transition_band": int(len(band)),
        "n_source_near": n_source_near,
        "n_center": n_center,
        "n_target_near": n_target_near,
        "mean_orthogonal_distance": mean_orth,
        "balance_score": float(balance_score),
        "band_score": float(band_score),
        "orthogonal_threshold": float(orth_threshold) if pd.notna(orth_threshold) else np.nan,
        "passed_min_group_size": bool(passed_min_group_size),
    }
    return band, metrics


def jaccard_index(a, b):
    a = set(a)
    b = set(b)
    if len(a) == 0 and len(b) == 0:
        return np.nan
    union = a | b
    if len(union) == 0:
        return np.nan
    return len(a & b) / len(union)


def load_reference_transition_keys(transition_path):
    if transition_path is None or not os.path.exists(transition_path):
        return set()
    try:
        tdf = pd.read_csv(transition_path, low_memory=False)
        tdf, _ = prepare_merge_key(tdf, table_name="transition table")
        return set(tdf["merge_key"].dropna().astype(str))
    except Exception:
        return set()


def build_merged_table(args):
    pca = safe_read_csv(args.pca, label="PCA")
    master = safe_read_csv(args.master, label="master")
    pca, pca_id_col = prepare_merge_key(pca, table_name="PCA table")
    master, master_id_col = prepare_merge_key(master, table_name="master table")

    merged = pd.merge(
        pca,
        master,
        on="merge_key",
        how="inner",
        suffixes=("_pca", "_master")
    )
    if len(merged) == 0:
        raise ValueError(
            "Inner merge produced 0 rows. Check accession/title formats in PCA and master tables. "
            f"PCA ID column: {pca_id_col}; master ID column: {master_id_col}"
        )

    merged = add_canonical_columns_after_merge(merged)
    merged["protein_for_analysis"] = args.protein
    merged["subtype_for_analysis"] = args.subtype
    merged["direction_for_analysis"] = args.direction
    merged["source_host_for_analysis"] = normalize_host_value(args.source_host)
    merged["target_host_for_analysis"] = normalize_host_value(args.target_host)

    out_path = os.path.join(args.outdir, "merged_pairwise_master_inner.csv")
    merged.to_csv(out_path, index=False)
    return merged, out_path


def run_permutation_control(args, merged):
    host_col = "host" if "host" in merged.columns else find_column(merged, HOST_CANDIDATES, label="host column")
    pc_cols = find_pc_columns(merged)

    observed_df, centroid_distance = compute_projection(
        merged,
        pc_cols=pc_cols,
        host_col=host_col,
        source_host=args.source_host,
        target_host=args.target_host
    )
    observed_band, observed_metrics = extract_transition_band(
        observed_df,
        q=args.default_q,
        trim=args.default_trim,
        min_group_size=args.default_min_group_size
    )

    rng = np.random.default_rng(args.seed)
    null_rows = []
    host_values = observed_df[host_col].apply(normalize_host_value).astype(str).values.copy()

    for i in range(args.n_perm):
        perm_df = observed_df.copy()
        permuted_hosts = host_values.copy()
        rng.shuffle(permuted_hosts)
        perm_df["_permuted_host"] = permuted_hosts
        try:
            perm_projected, perm_centroid_distance = compute_projection(
                perm_df,
                pc_cols=pc_cols,
                host_col="_permuted_host",
                source_host=args.source_host,
                target_host=args.target_host
            )
            _, perm_metrics = extract_transition_band(
                perm_projected,
                q=args.default_q,
                trim=args.default_trim,
                min_group_size=args.default_min_group_size
            )
            row = {
                "perm_id": i + 1,
                "centroid_distance": perm_centroid_distance,
                **perm_metrics,
                "error": ""
            }
        except Exception as e:
            row = {
                "perm_id": i + 1,
                "centroid_distance": np.nan,
                "n_transition_band": np.nan,
                "n_source_near": np.nan,
                "n_center": np.nan,
                "n_target_near": np.nan,
                "mean_orthogonal_distance": np.nan,
                "balance_score": np.nan,
                "band_score": np.nan,
                "orthogonal_threshold": np.nan,
                "passed_min_group_size": False,
                "error": str(e)
            }
        null_rows.append(row)

    null_df = pd.DataFrame(null_rows)
    null_path = os.path.join(args.outdir, "permutation_null_distribution.csv")
    null_df.to_csv(null_path, index=False)

    valid_scores = null_df["band_score"].dropna().values
    observed_score = observed_metrics["band_score"]
    if len(valid_scores) > 0:
        empirical_p = (1 + np.sum(valid_scores >= observed_score)) / (len(valid_scores) + 1)
        null_mean = float(np.mean(valid_scores))
        null_sd = float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else np.nan
        null_median = float(np.median(valid_scores))
        null_q95 = float(np.quantile(valid_scores, 0.95))
    else:
        empirical_p = np.nan
        null_mean = np.nan
        null_sd = np.nan
        null_median = np.nan
        null_q95 = np.nan

    source_norm = normalize_host_value(args.source_host)
    target_norm = normalize_host_value(args.target_host)
    summary = pd.DataFrame([{
        "protein": args.protein,
        "subtype": args.subtype,
        "direction": args.direction,
        "source_host": source_norm,
        "target_host": target_norm,
        "n_total": int(len(observed_df)),
        "n_source": int((observed_df[host_col].apply(normalize_host_value).astype(str) == source_norm).sum()),
        "n_target": int((observed_df[host_col].apply(normalize_host_value).astype(str) == target_norm).sum()),
        "n_perm": int(args.n_perm),
        "pc_columns": ",".join(pc_cols),
        "host_column_used": host_col,
        "observed_centroid_distance": float(centroid_distance),
        "observed_n_transition_band": observed_metrics["n_transition_band"],
        "observed_n_source_near": observed_metrics["n_source_near"],
        "observed_n_center": observed_metrics["n_center"],
        "observed_n_target_near": observed_metrics["n_target_near"],
        "observed_balance_score": observed_metrics["balance_score"],
        "observed_mean_orthogonal_distance": observed_metrics["mean_orthogonal_distance"],
        "observed_band_score": observed_score,
        "null_mean_band_score": null_mean,
        "null_median_band_score": null_median,
        "null_sd_band_score": null_sd,
        "null_95pct_band_score": null_q95,
        "empirical_p_value": empirical_p,
        "default_q": args.default_q,
        "default_trim": args.default_trim,
        "default_min_group_size": args.default_min_group_size,
    }])
    summary_path = os.path.join(args.outdir, "permutation_negative_control_summary.csv")
    summary.to_csv(summary_path, index=False)
    return observed_df, observed_band, summary_path, null_path


def run_sensitivity(args, projected_df, observed_band):
    rows = []
    default_keys = set(observed_band["merge_key"].dropna().astype(str)) if "merge_key" in observed_band.columns else set()
    reference_transition_keys = load_reference_transition_keys(args.transition)

    q_values = [0.10, 0.15, 0.20, 0.25]
    trim_values = [0.05, 0.10, 0.15]
    min_group_values = [20, 30, 50]

    for q in q_values:
        for trim in trim_values:
            for min_group in min_group_values:
                band, metrics = extract_transition_band(
                    projected_df,
                    q=q,
                    trim=trim,
                    min_group_size=min_group
                )
                current_keys = set(band["merge_key"].dropna().astype(str)) if "merge_key" in band.columns else set()
                jac_default = jaccard_index(default_keys, current_keys)
                jac_reference = jaccard_index(reference_transition_keys, current_keys) if len(reference_transition_keys) > 0 else np.nan

                rows.append({
                    "protein": args.protein,
                    "subtype": args.subtype,
                    "direction": args.direction,
                    "source_host": normalize_host_value(args.source_host),
                    "target_host": normalize_host_value(args.target_host),
                    "orthogonal_distance_quantile": q,
                    "source_target_trim": trim,
                    "min_group_size": min_group,
                    "n_transition_band": metrics["n_transition_band"],
                    "n_source_near": metrics["n_source_near"],
                    "n_center": metrics["n_center"],
                    "n_target_near": metrics["n_target_near"],
                    "mean_orthogonal_distance": metrics["mean_orthogonal_distance"],
                    "orthogonal_threshold": metrics["orthogonal_threshold"],
                    "balance_score": metrics["balance_score"],
                    "band_score": metrics["band_score"],
                    "passed_min_group_size": metrics["passed_min_group_size"],
                    "candidate_sample_Jaccard_index_vs_default": jac_default,
                    "candidate_sample_Jaccard_index_vs_existing_transition_file": jac_reference,
                    "top_site_overlap_with_default": np.nan,
                    "trajectory_score_correlation_with_default": np.nan,
                    "note": "Top-site stability requires rerunning site trajectory for each threshold or using a site-state matrix."
                })

    sens_df = pd.DataFrame(rows)
    sens_path = os.path.join(args.outdir, "transition_band_sensitivity_summary.csv")
    sens_df.to_csv(sens_path, index=False)
    return sens_path


def print_basic_checks(merged):
    print("\n[CHECK] Canonical columns")
    for c in ["accession", "host", "protein", "subtype", "collection_date", "year", "country", "PC1", "PC2", "PC3"]:
        print(f"  {c}: {'OK' if c in merged.columns else 'missing'}")
    if "host" in merged.columns:
        print("\n[CHECK] host counts")
        print(merged["host"].value_counts(dropna=False).to_string())
    print("")



def parse_combo_dir_name(combo_name):
    """Parse names like HA_H1_artiodactyla_to_primates."""
    if "_to_" not in combo_name:
        raise ValueError(
            f"Directory name does not contain '_to_': {combo_name}. "
            "Expected format: Protein_Subtype_source_to_target, e.g. NP_H1N1_artiodactyla_to_primates."
        )
    left, target_host = combo_name.split("_to_", 1)
    parts = left.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse protein/subtype/source_host from directory name: {combo_name}. "
            "Expected format: Protein_Subtype_source_to_target."
        )
    protein = parts[0]
    subtype = parts[1]
    source_host = "_".join(parts[2:])
    direction = f"{source_host}_to_{target_host}"
    return protein, subtype, source_host, target_host, direction


def list_csv_files(combo_dir):
    csv_files = []
    for root, dirs, files in os.walk(combo_dir):
        # Avoid reading previously generated output directories if a user accidentally nests them.
        dirs[:] = [d for d in dirs if not d.startswith("figure3_support") and d not in {"__pycache__"}]
        for fn in files:
            if fn.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, fn))
    return csv_files


def choose_file_by_patterns(csv_files, exact_names, include_tokens=None, exclude_tokens=None, label="file"):
    """Choose one CSV path by exact-name priority, then token matching."""
    include_tokens = include_tokens or []
    exclude_tokens = exclude_tokens or []

    # 1) Exact basename priority.
    basename_to_paths = {}
    for p in csv_files:
        basename_to_paths.setdefault(os.path.basename(p), []).append(p)
    for name in exact_names:
        if name in basename_to_paths:
            # Prefer the shallowest path if duplicates exist.
            return sorted(basename_to_paths[name], key=lambda x: (x.count(os.sep), x))[0]

    # 2) Token matching.
    candidates = []
    for p in csv_files:
        low = os.path.basename(p).lower()
        if all(tok.lower() in low for tok in include_tokens) and not any(tok.lower() in low for tok in exclude_tokens):
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=lambda x: (x.count(os.sep), len(os.path.basename(x)), x))[0]

    return None


def autodetect_combo_files(combo_dir):
    csv_files = list_csv_files(combo_dir)

    pca_path = choose_file_by_patterns(
        csv_files,
        exact_names=[
            "pca_coordinates.csv",
            "pairwise_pca_coordinates.csv",
            "pca_coords.csv",
        ],
        include_tokens=["pca"],
        exclude_tokens=["permutation", "sensitivity", "summary", "null", "merged"],
        label="PCA"
    )

    master_path = choose_file_by_patterns(
        csv_files,
        exact_names=[
            "master_table.csv",
            "val_master_table.csv",
            "merged_pairwise_master_inner.csv",
            "pairwise_master.csv",
            "merged_master.csv",
        ],
        include_tokens=["master"],
        exclude_tokens=["with_projection", "recomputed", "permutation", "sensitivity", "summary", "null"],
        label="master"
    )

    transition_path = choose_file_by_patterns(
        csv_files,
        exact_names=[
            "transition_band_samples.csv",
            "bridge_candidates.csv",
        ],
        include_tokens=["transition"],
        exclude_tokens=["recomputed", "sensitivity", "summary", "null"],
        label="transition"
    )

    return pca_path, master_path, transition_path, csv_files


def process_one_combo(args, combo_label="single"):
    """Run one protein-subtype-direction combination and return a status dictionary."""
    os.makedirs(args.outdir, exist_ok=True)
    status = {
        "combo": combo_label,
        "protein": getattr(args, "protein", ""),
        "subtype": getattr(args, "subtype", ""),
        "direction": getattr(args, "direction", ""),
        "source_host": getattr(args, "source_host", ""),
        "target_host": getattr(args, "target_host", ""),
        "pca": getattr(args, "pca", ""),
        "master": getattr(args, "master", ""),
        "transition": getattr(args, "transition", ""),
        "outdir": getattr(args, "outdir", ""),
        "status": "failed",
        "message": "",
        "merged_rows": np.nan,
        "observed_transition_band_rows": np.nan,
    }
    try:
        merged, merged_path = build_merged_table(args)
        print_basic_checks(merged)

        projected_df, observed_band, perm_summary_path, null_path = run_permutation_control(args, merged)

        projected_path = os.path.join(args.outdir, "merged_pairwise_master_inner_with_projection.csv")
        projected_df.to_csv(projected_path, index=False)

        recomputed_band_path = os.path.join(args.outdir, "transition_band_samples_recomputed_default.csv")
        observed_band.to_csv(recomputed_band_path, index=False)

        sensitivity_path = run_sensitivity(args, projected_df, observed_band)

        print("[OK] Files generated:")
        print(f"  merged table: {merged_path}")
        print(f"  merged table with projection: {projected_path}")
        print(f"  recomputed default transition band: {recomputed_band_path}")
        print(f"  permutation summary: {perm_summary_path}")
        print(f"  permutation null distribution: {null_path}")
        print(f"  transition-band sensitivity summary: {sensitivity_path}")
        print(f"[INFO] merged rows: {len(merged)}")
        print(f"[INFO] observed default transition-band rows: {len(observed_band)}")

        status.update({
            "status": "ok",
            "message": "",
            "merged_rows": int(len(merged)),
            "observed_transition_band_rows": int(len(observed_band)),
        })
    except Exception as e:
        status["message"] = str(e)
        print(f"[ERROR] {combo_label}: {e}", file=sys.stderr)
        if getattr(args, "raise_on_error", False):
            raise
    return status


def build_args_for_combo(parent_args, combo_dir, combo_name, pca_path, master_path, transition_path):
    protein, subtype, source_host, target_host, direction = parse_combo_dir_name(combo_name)
    outdir = os.path.join(parent_args.batch_out_root, combo_name)
    values = vars(parent_args).copy()
    values.update({
        "combo_dir": combo_dir,
        "pca": pca_path,
        "master": master_path,
        "transition": transition_path,
        "outdir": outdir,
        "source_host": source_host,
        "target_host": target_host,
        "protein": protein,
        "subtype": subtype,
        "direction": direction,
    })
    return argparse.Namespace(**values)


def run_batch(args):
    batch_root = os.path.abspath(args.batch_root)
    batch_out_root = os.path.abspath(args.batch_out_root)
    os.makedirs(batch_out_root, exist_ok=True)

    if not os.path.isdir(batch_root):
        raise FileNotFoundError(f"Batch root directory not found: {batch_root}")

    combo_dirs = []
    for name in sorted(os.listdir(batch_root)):
        path = os.path.join(batch_root, name)
        if not os.path.isdir(path):
            continue
        if name.startswith(".") or name in {"__pycache__"}:
            continue
        try:
            parse_combo_dir_name(name)
        except Exception:
            if args.verbose_skip:
                print(f"[SKIP] {name}: directory name does not match Protein_Subtype_source_to_target")
            continue
        combo_dirs.append((name, path))

    if len(combo_dirs) == 0:
        raise ValueError(f"No valid combo directories found under: {batch_root}")

    print(f"[BATCH] root: {batch_root}")
    print(f"[BATCH] output root: {batch_out_root}")
    print(f"[BATCH] valid combo directories: {len(combo_dirs)}")

    status_rows = []
    for idx, (name, combo_dir) in enumerate(combo_dirs, 1):
        print("\n" + "=" * 90)
        print(f"[BATCH] {idx}/{len(combo_dirs)}  {name}")
        print("=" * 90)

        try:
            pca_path, master_path, transition_path, csv_files = autodetect_combo_files(combo_dir)
            protein, subtype, source_host, target_host, direction = parse_combo_dir_name(name)
            outdir = os.path.join(batch_out_root, name)

            detect_status = {
                "combo": name,
                "protein": protein,
                "subtype": subtype,
                "direction": direction,
                "source_host": source_host,
                "target_host": target_host,
                "pca": pca_path or "",
                "master": master_path or "",
                "transition": transition_path or "",
                "outdir": outdir,
                "status": "detected",
                "message": "",
                "merged_rows": np.nan,
                "observed_transition_band_rows": np.nan,
                "n_csv_files_seen": len(csv_files),
            }

            if pca_path is None or master_path is None:
                missing = []
                if pca_path is None:
                    missing.append("PCA CSV")
                if master_path is None:
                    missing.append("master CSV")
                detect_status["status"] = "skipped"
                detect_status["message"] = "Missing required file(s): " + ", ".join(missing)
                status_rows.append(detect_status)
                print(f"[SKIP] {name}: {detect_status['message']}")
                print(f"[INFO] CSV files seen: {len(csv_files)}")
                continue

            print(f"[DETECT] PCA:        {pca_path}")
            print(f"[DETECT] master:     {master_path}")
            print(f"[DETECT] transition: {transition_path if transition_path else 'not found; sensitivity Jaccard vs existing transition file will be NA'}")
            print(f"[DETECT] output:     {outdir}")
            print(f"[DETECT] parsed:     protein={protein}, subtype={subtype}, source={source_host}, target={target_host}, direction={direction}")

            if args.dry_run:
                detect_status["status"] = "dry_run_detected"
                detect_status["message"] = "Dry run only; no files generated."
                status_rows.append(detect_status)
                continue

            combo_args = build_args_for_combo(args, combo_dir, name, pca_path, master_path, transition_path)
            one_status = process_one_combo(combo_args, combo_label=name)
            one_status["n_csv_files_seen"] = len(csv_files)
            status_rows.append(one_status)

        except Exception as e:
            row = {
                "combo": name,
                "protein": "",
                "subtype": "",
                "direction": "",
                "source_host": "",
                "target_host": "",
                "pca": "",
                "master": "",
                "transition": "",
                "outdir": os.path.join(batch_out_root, name),
                "status": "failed",
                "message": str(e),
                "merged_rows": np.nan,
                "observed_transition_band_rows": np.nan,
                "n_csv_files_seen": np.nan,
            }
            status_rows.append(row)
            print(f"[ERROR] {name}: {e}", file=sys.stderr)
            if args.raise_on_error:
                raise

    summary_df = pd.DataFrame(status_rows)
    summary_path = os.path.join(batch_out_root, "batch_run_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 90)
    print(f"[BATCH OK] Summary written to: {summary_path}")
    print(summary_df["status"].value_counts(dropna=False).to_string())
    print("=" * 90)
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build Figure 3 support tables. Supports both single-combo mode and batch mode. "
            "Batch mode scans directories named Protein_Subtype_source_to_target and writes outputs to "
            "batch_out_root/same_directory_name."
        )
    )

    # Batch mode options.
    parser.add_argument(
        "--batch-root",
        default=None,
        help="Directory containing combo subdirectories. Specify the input directory explicitly."
    )
    parser.add_argument(
        "--batch-out-root",
        default=None,
        help="Output root for batch mode. Each combo writes to batch_out_root/combo_dir_name. Specify explicitly."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Batch mode only: detect directories/files and write no output."
    )
    parser.add_argument(
        "--verbose-skip",
        action="store_true",
        help="Batch mode only: print skipped non-combo directories."
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Stop immediately on the first error instead of continuing batch mode."
    )

    # Single-combo mode options. Kept compatible with v2.
    parser.add_argument("--combo-dir", default=None, help="Optional combo directory; kept for compatibility.")
    parser.add_argument("--pca", default=None, help="Path to pca_coordinates.csv.")
    parser.add_argument("--master", default=None, help="Path to master_table.csv or val_master_table.csv.")
    parser.add_argument("--transition", default=None, help="Optional path to transition_band_samples.csv.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument("--source-host", default=None, help="Source host, e.g. artiodactyla.")
    parser.add_argument("--target-host", default=None, help="Target host, e.g. primates.")
    parser.add_argument("--protein", default=None, help="Protein name, e.g. NP.")
    parser.add_argument("--subtype", default=None, help="Subtype name, e.g. H1N1.")
    parser.add_argument("--direction", default=None, help="Direction name, e.g. artiodactyla_to_primates.")

    # Shared analysis parameters.
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--default-q", type=float, default=0.15, help="Default orthogonal-distance quantile.")
    parser.add_argument("--default-trim", type=float, default=0.10, help="Default projection-score trimming.")
    parser.add_argument("--default-min-group-size", type=int, default=30, help="Default minimum size for each transition layer.")
    args = parser.parse_args()

    if args.batch_root is not None:
        if args.batch_out_root is None:
            parser.error("--batch-out-root is required when --batch-root is used.")
        run_batch(args)
        return

    # Single mode validation.
    required = {
        "--pca": args.pca,
        "--master": args.master,
        "--outdir": args.outdir,
        "--source-host": args.source_host,
        "--target-host": args.target_host,
        "--protein": args.protein,
        "--subtype": args.subtype,
        "--direction": args.direction,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Missing required arguments for single-combo mode: " + ", ".join(missing) +
            ". Alternatively, use --batch-root Figure3_source_data."
        )

    process_one_combo(args, combo_label=args.direction)


if __name__ == "__main__":
    main()
