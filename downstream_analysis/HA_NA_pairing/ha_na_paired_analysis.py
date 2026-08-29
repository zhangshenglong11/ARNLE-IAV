from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



META_COLS = [
    "accession", "title", "title_aln", "host", "host_norm",
    "collection_date", "year", "month", "country",
    "protein", "subtype", "subtype_final", "H_subtype", "N_subtype",
    "Prob_AR", "Prob_PR", "Prob_AV",
]
PCA_COLS = [
    "PC1", "PC2", "PC3", "feature_row", "projection_t", "approach_score",
    "orthogonal_distance", "distance_to_source", "distance_to_target",
]
# Accession is deliberately excluded.
STRICT_KEY_CANDIDATES = [
    "_auto_pair_key",
    "_isolate_key_strain", "_strain_strain", "_virus_name_strain",
    "_isolate_strain", "_strain_name_strain", "_sample_id_strain",
    "_isolate_key_norm", "_strain_norm", "_virus_name_norm",
    "_isolate_norm", "_strain_name_norm", "_sample_id_norm",
    "_title_aln_strain", "_title_strain",
]


def read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def normalize_text(value: object) -> str:
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value.strip())


def normalize_key(value: object) -> str:
    value = normalize_text(value).lower().replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def extract_strain_from_text(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    patterns = [
        r"(A/[A-Za-z0-9_.\- /]+?/\d{4})(?:\s*\([Hh]\d+[Nn]\d+\))?",
        r"(A/[A-Za-z0-9_.\- /]+?/\d{4})(?:[Hh]\d+[Nn]\d+)?",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return normalize_key(matches[-1].group(1).rstrip(" .;,:"))
    return ""


def add_pair_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["title_aln", "title"]:
        if col in df.columns:
            df[f"_{col}_strain"] = df[col].map(extract_strain_from_text)
        else:
            df[f"_{col}_strain"] = ""

    for col in [
        "isolate_key", "strain", "virus_name", "isolate",
        "strain_name", "sample_id",
    ]:
        if col in df.columns:
            df[f"_{col}_norm"] = df[col].map(normalize_key)
            df[f"_{col}_strain"] = df[col].map(extract_strain_from_text)
        else:
            df[f"_{col}_norm"] = ""
            df[f"_{col}_strain"] = ""

    auto = pd.Series("", index=df.index, dtype=object)
    priority = [
        "_isolate_key_strain", "_strain_strain", "_virus_name_strain",
        "_isolate_strain", "_strain_name_strain", "_sample_id_strain",
        "_isolate_key_norm", "_strain_norm", "_virus_name_norm",
        "_isolate_norm", "_strain_name_norm", "_sample_id_norm",
        "_title_aln_strain", "_title_strain",
    ]
    for col in priority:
        values = df[col].astype(str)
        auto = auto.mask(auto.eq("") & values.ne(""), values)
    df["_auto_pair_key"] = auto
    return df


def key_diagnostics(
    ha: pd.DataFrame,
    na: pd.DataFrame,
    candidates: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for col in candidates:
        if col not in ha.columns or col not in na.columns:
            rows.append({"key_col": col, "status": "missing"})
            continue
        h = ha[col].astype(str)
        n = na[col].astype(str)
        h_non = h[h.ne("")]
        n_non = n[n.ne("")]
        hc = h_non.value_counts()
        nc = n_non.value_counts()
        h_unique = set(hc[hc.eq(1)].index)
        n_unique = set(nc[nc.eq(1)].index)
        rows.append(
            {
                "key_col": col,
                "status": "ok",
                "ha_nonempty_rows": int(len(h_non)),
                "na_nonempty_rows": int(len(n_non)),
                "ha_distinct_keys": int(h_non.nunique()),
                "na_distinct_keys": int(n_non.nunique()),
                "ha_unique_once_keys": int(len(h_unique)),
                "na_unique_once_keys": int(len(n_unique)),
                "intersection_unique_on_both": int(len(h_unique & n_unique)),
                "intersection_any": int(len(set(h_non) & set(n_non))),
                "ha_duplicate_keys": int(hc.gt(1).sum()),
                "na_duplicate_keys": int(nc.gt(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_key(
    diagnostics: pd.DataFrame,
    requested: str,
    min_pairs: int,
) -> str:
    if requested != "auto":
        if requested not in diagnostics["key_col"].tolist():
            raise ValueError(f"Requested key not available: {requested}")
        row = diagnostics.loc[diagnostics["key_col"].eq(requested)].iloc[0]
        if row.get("status") != "ok":
            raise ValueError(f"Requested key unavailable: {requested}")
        if int(row["intersection_unique_on_both"]) < min_pairs:
            raise ValueError(
                f"{requested} has only "
                f"{int(row['intersection_unique_on_both'])} strict pairs"
            )
        return requested

    valid = diagnostics.loc[diagnostics["status"].eq("ok")].copy()
    valid = valid.sort_values(
        "intersection_unique_on_both", ascending=False
    )
    valid = valid.loc[valid["intersection_unique_on_both"].ge(min_pairs)]
    if valid.empty:
        raise ValueError(
            "No non-accession isolate-level key produced enough strict "
            "one-to-one pairs."
        )
    return str(valid.iloc[0]["key_col"])


def normalize_meta(value: object) -> str:
    return normalize_key(value)


def metadata_conflict_columns(
    paired: pd.DataFrame,
    expected_subtype: str,
) -> pd.DataFrame:
    flags = pd.DataFrame(index=paired.index)

    comparisons = {
        "conflicting_host": ("HA_host", "NA_host"),
        "conflicting_year": ("HA_year", "NA_year"),
        "conflicting_country": ("HA_country", "NA_country"),
        "conflicting_subtype_final": ("HA_subtype_final", "NA_subtype_final"),
        "conflicting_H_subtype": ("HA_H_subtype", "NA_H_subtype"),
        "conflicting_N_subtype": ("HA_N_subtype", "NA_N_subtype"),
    }

    for flag, (left, right) in comparisons.items():
        if left in paired.columns and right in paired.columns:
            l = paired[left].map(normalize_meta)
            r = paired[right].map(normalize_meta)
            flags[flag] = l.ne("") & r.ne("") & l.ne(r)
        else:
            flags[flag] = False

    flags["conflicting_expected_subtype"] = False
    expected = normalize_meta(expected_subtype)
    if expected:
        for col in ["HA_subtype", "HA_subtype_final", "NA_subtype", "NA_subtype_final"]:
            if col in paired.columns:
                values = paired[col].map(normalize_meta)
                nonempty = values.ne("")
                # Compare only complete combined-subtype fields.
                combined = values.str.match(r"^h\d+n\d+$", na=False)
                flags["conflicting_expected_subtype"] |= (
                    nonempty & combined & values.ne(expected)
                )

    flags["metadata_conflict_any"] = flags.any(axis=1)
    return flags


def infer_group_col(df: pd.DataFrame) -> Optional[str]:
    for col in [
        "transition_group",
        "transition_band_group",
        "middle_group",
        "group",
        "band_group",
    ]:
        if col in df.columns:
            return col
    return None


def attach_pca(
    base_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    key_col: str,
    prefix: str,
) -> pd.DataFrame:
    keep = [key_col] + [c for c in PCA_COLS if c in pca_df.columns]
    small = pca_df[keep].copy()
    if small[key_col].duplicated().any():
        raise ValueError(f"{prefix} PCA table has duplicated {key_col}")
    small = small.rename(
        columns={c: f"{prefix}_{c}" for c in keep if c != key_col}
    )
    return base_df.merge(small, on=key_col, how="left", validate="one_to_one")


def attach_transition(
    base_df: pd.DataFrame,
    trans_df: Optional[pd.DataFrame],
    key_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = base_df.copy()
    out[f"{prefix}_in_transition_band"] = False
    out[f"{prefix}_transition_group"] = ""
    if trans_df is None:
        return out

    trans_df = add_pair_keys(trans_df)
    group_col = infer_group_col(trans_df)
    keep = [key_col]
    for col in [
        "projection_t", "approach_score", "orthogonal_distance",
        "distance_to_source", "distance_to_target",
    ]:
        if col in trans_df.columns:
            keep.append(col)
    if group_col and group_col not in keep:
        keep.append(group_col)

    small = trans_df[keep].drop_duplicates(subset=[key_col]).copy()
    rename = {c: f"{prefix}_{c}" for c in keep if c != key_col}
    small = small.rename(columns=rename)
    out = out.merge(small, on=key_col, how="left", validate="one_to_one")

    projection = f"{prefix}_projection_t"
    if projection in out.columns:
        out[f"{prefix}_in_transition_band"] = out[projection].ne("")
    else:
        out[f"{prefix}_in_transition_band"] = out[key_col].isin(
            set(small[key_col])
        )
    if group_col:
        out[f"{prefix}_transition_group"] = out[
            f"{prefix}_{group_col}"
        ].fillna("")
    return out


def top_sites(
    trajectory: Optional[pd.DataFrame],
    top_k: int,
) -> List[Tuple[str, str]]:
    if trajectory is None or trajectory.empty:
        return []
    df = trajectory.copy()
    site_col = next(
        (c for c in ["position", "site", "msa_col", "alignment_col", "col"]
         if c in df.columns),
        None,
    )
    aa_col = next(
        (c for c in ["target_major_aa", "target_major", "target_aa",
                     "primates_major_aa"] if c in df.columns),
        None,
    )
    if site_col is None or aa_col is None:
        raise ValueError("Trajectory table lacks site/target-major-AA columns")

    if "trajectory_score" in df.columns:
        df["_sort"] = pd.to_numeric(
            df["trajectory_score"], errors="coerce"
        ).fillna(-np.inf)
        df = df.sort_values("_sort", ascending=False)
    elif "rank" in df.columns:
        df["_sort"] = pd.to_numeric(
            df["rank"], errors="coerce"
        ).fillna(np.inf)
        df = df.sort_values("_sort", ascending=True)

    result = []
    for _, row in df.head(top_k).iterrows():
        site = str(row[site_col]).strip()
        aa = str(row[aa_col]).strip()
        if site and aa:
            result.append((site, aa))
    return result


def add_target_major_fraction(
    base_df: pd.DataFrame,
    master_df: pd.DataFrame,
    key_col: str,
    prefix: str,
    sites: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    available = [(site, aa) for site, aa in sites if site in master_df.columns]
    if not available:
        out = base_df.copy()
        out[f"{prefix}_target_major_count"] = np.nan
        out[f"{prefix}_target_major_site_n"] = 0
        out[f"{prefix}_target_major_fraction"] = np.nan
        return out

    tmp = master_df[[key_col] + [s for s, _ in available]].copy()
    if tmp[key_col].duplicated().any():
        raise ValueError(
            f"{prefix} strict master subset unexpectedly contains duplicate keys"
        )

    count = pd.Series(0.0, index=tmp.index)
    valid = pd.Series(0.0, index=tmp.index)
    for site, target_aa in available:
        values = tmp[site].astype(str).str.strip()
        ok = values.ne("") & values.ne("-") & values.ne("nan")
        valid += ok.astype(float)
        count += (values == target_aa).astype(float)

    tmp[f"{prefix}_target_major_count"] = count
    tmp[f"{prefix}_target_major_site_n"] = valid
    tmp[f"{prefix}_target_major_fraction"] = np.where(
        valid.gt(0), count / valid, np.nan
    )
    keep = [
        key_col,
        f"{prefix}_target_major_count",
        f"{prefix}_target_major_site_n",
        f"{prefix}_target_major_fraction",
    ]
    return base_df.merge(tmp[keep], on=key_col, how="left", validate="one_to_one")


def prefix_except_key(
    df: pd.DataFrame,
    prefix: str,
    key_col: str,
) -> pd.DataFrame:
    return df.rename(
        columns={
            c: f"{prefix}_{c}"
            for c in df.columns
            if c != key_col and not c.startswith(f"{prefix}_")
        }
    )


def classify_joint_state(
    row: pd.Series,
    threshold: float,
) -> str:
    ha = pd.to_numeric(row.get("HA_target_major_fraction"), errors="coerce")
    na = pd.to_numeric(row.get("NA_target_major_fraction"), errors="coerce")
    ha_high = pd.notna(ha) and float(ha) >= threshold
    na_high = pd.notna(na) and float(na) >= threshold

    if ha_high and na_high:
        return "Both target-like"
    if ha_high and not na_high:
        return "HA only"
    if na_high and not ha_high:
        return "NA only"
    if bool(row.get("HA_in_transition_band", False)) or bool(
        row.get("NA_in_transition_band", False)
    ):
        return "Intermediate / low"
    return "No clear joint state"


def pearson_permutation_p(
    ha: pd.Series,
    na: pd.Series,
    n_perm: int = 1000,
    seed: int = 2026,
) -> float:
    """Random-label permutation P for the HA-NA target-major-fraction Pearson r."""
    ha_arr = ha.to_numpy(dtype=float)
    na_arr = na.to_numpy(dtype=float)
    if len(ha_arr) < 3:
        return np.nan
    obs = float(np.corrcoef(ha_arr, na_arr)[0, 1])
    rng = np.random.default_rng(seed)
    count = 0
    n_valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(na_arr)
        r = np.corrcoef(ha_arr, perm)[0, 1]
        if np.isfinite(r):
            n_valid += 1
            if r >= obs:
                count += 1
    if n_valid == 0:
        return np.nan
    return float((count + 1) / (n_valid + 1))


def pearson_bootstrap_ci(
    ha: pd.Series,
    na: pd.Series,
    n_iter: int = 1000,
    seed: int = 2027,
    ci: float = 0.95,
):
    """Bootstrap 95% CI for the HA-NA target-major-fraction Pearson r."""
    ha_arr = ha.to_numpy(dtype=float)
    na_arr = na.to_numpy(dtype=float)
    n = len(ha_arr)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        r = np.corrcoef(ha_arr[idx], na_arr[idx])[0, 1]
        if np.isfinite(r):
            rs.append(float(r))
    if len(rs) < 200:
        return np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(rs, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def joint_state_threshold_sensitivity(paired: pd.DataFrame) -> dict:
    """Joint-state composition under the 0.5/0.6/0.7/0.8 thresholds."""
    out = {}
    for thr in (0.5, 0.6, 0.7, 0.8):
        js = paired.apply(classify_joint_state, axis=1, threshold=thr)
        out[str(thr)] = {
            str(k): float(v)
            for k, v in js.value_counts(normalize=True).to_dict().items()
        }
    return out


def save_paired_scatter(both_nonmissing: pd.DataFrame, out_dir: Path) -> None:
    """Write the HA-NA paired target-major-fraction scatter plot."""
    ha = pd.to_numeric(
        both_nonmissing["HA_target_major_fraction"], errors="coerce"
    )
    na = pd.to_numeric(
        both_nonmissing["NA_target_major_fraction"], errors="coerce"
    )
    valid = ha.notna() & na.notna()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(na[valid], ha[valid], s=8, alpha=0.5, edgecolors="none")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("NA target-major fraction")
    ax.set_ylabel("HA target-major fraction")
    ax.set_title("HA-NA paired isolate scatter")
    fig.tight_layout()
    fig.savefig(out_dir / "paired_scatter.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ha-master", required=True, type=Path)
    parser.add_argument("--na-master", required=True, type=Path)
    parser.add_argument("--ha-pca", required=True, type=Path)
    parser.add_argument("--na-pca", required=True, type=Path)
    parser.add_argument("--ha-transition", type=Path)
    parser.add_argument("--na-transition", type=Path)
    parser.add_argument("--ha-trajectory", type=Path)
    parser.add_argument("--na-trajectory", type=Path)
    parser.add_argument("--pair-key", default="auto")
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--top-k-sites", type=int, default=40)
    parser.add_argument("--expected-subtype", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--joint-threshold", type=float, default=0.60)
    parser.add_argument("--strict-high", type=float, default=0.80)
    parser.add_argument("--strict-low", type=float, default=0.50)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ha_master = add_pair_keys(read_csv(args.ha_master))
    na_master = add_pair_keys(read_csv(args.na_master))
    ha_pca = add_pair_keys(read_csv(args.ha_pca))
    na_pca = add_pair_keys(read_csv(args.na_pca))
    ha_transition = read_csv(args.ha_transition)
    na_transition = read_csv(args.na_transition)
    ha_trajectory = read_csv(args.ha_trajectory)
    na_trajectory = read_csv(args.na_trajectory)

    diagnostics = key_diagnostics(
        ha_master, na_master, STRICT_KEY_CANDIDATES
    )
    diagnostics.to_csv(args.out_dir / "key_diagnostics.csv", index=False)
    key_col = choose_key(diagnostics, args.pair_key, args.min_pairs)

    ha_counts = ha_master[key_col].value_counts()
    na_counts = na_master[key_col].value_counts()
    ha_unique = set(ha_counts[ha_counts.eq(1)].index) - {""}
    na_unique = set(na_counts[na_counts.eq(1)].index) - {""}
    valid_keys = ha_unique & na_unique

    any_intersection = (
        (set(ha_counts.index) - {""}) & (set(na_counts.index) - {""})
    )
    duplicate_intersection = any_intersection - valid_keys

    ha_duplicate_keys = ha_counts[ha_counts.gt(1)].rename("HA_count")
    na_duplicate_keys = na_counts[na_counts.gt(1)].rename("NA_count")
    ha_duplicate_keys.to_csv(args.out_dir / "duplicate_HA_keys.csv")
    na_duplicate_keys.to_csv(args.out_dir / "duplicate_NA_keys.csv")
    pd.DataFrame({key_col: sorted(duplicate_intersection)}).to_csv(
        args.out_dir / "duplicate_or_many_to_many_intersection_keys.csv",
        index=False,
    )

    # CRITICAL FIX: filter to valid_keys BEFORE drop_duplicates.
    ha_strict = ha_master.loc[ha_master[key_col].isin(valid_keys)].copy()
    na_strict = na_master.loc[na_master[key_col].isin(valid_keys)].copy()

    if len(ha_strict) != len(valid_keys) or len(na_strict) != len(valid_keys):
        raise AssertionError("Strict subsets are not one row per valid key")
    if ha_strict[key_col].duplicated().any() or na_strict[key_col].duplicated().any():
        raise AssertionError("Duplicated key remained after strict filtering")

    ha_unmatched = ha_master.loc[~ha_master[key_col].isin(valid_keys)].copy()
    na_unmatched = na_master.loc[~na_master[key_col].isin(valid_keys)].copy()
    ha_unmatched.to_csv(args.out_dir / "unmatched_or_duplicate_HA.csv", index=False)
    na_unmatched.to_csv(args.out_dir / "unmatched_or_duplicate_NA.csv", index=False)

    ha_cols = [key_col] + [c for c in META_COLS if c in ha_strict.columns]
    na_cols = [key_col] + [c for c in META_COLS if c in na_strict.columns]
    ha_base = ha_strict[ha_cols].copy()
    na_base = na_strict[na_cols].copy()

    ha_base = attach_pca(ha_base, ha_pca, key_col, "HA")
    na_base = attach_pca(na_base, na_pca, key_col, "NA")
    ha_base = attach_transition(
        ha_base, ha_transition, key_col, "HA"
    )
    na_base = attach_transition(
        na_base, na_transition, key_col, "NA"
    )

    ha_sites = top_sites(ha_trajectory, args.top_k_sites)
    na_sites = top_sites(na_trajectory, args.top_k_sites)
    ha_base = add_target_major_fraction(
        ha_base, ha_strict, key_col, "HA", ha_sites
    )
    na_base = add_target_major_fraction(
        na_base, na_strict, key_col, "NA", na_sites
    )

    paired = prefix_except_key(ha_base, "HA", key_col).merge(
        prefix_except_key(na_base, "NA", key_col),
        on=key_col,
        how="inner",
        validate="one_to_one",
    )

    conflict_flags = metadata_conflict_columns(
        paired, args.expected_subtype
    )
    paired = pd.concat([paired, conflict_flags], axis=1)
    conflicting = paired.loc[paired["metadata_conflict_any"]].copy()
    paired = paired.loc[~paired["metadata_conflict_any"]].copy()

    conflicting.to_csv(args.out_dir / "conflicting_pairs_excluded.csv", index=False)

    paired["subtype"] = args.expected_subtype
    paired["source_host"] = args.source_host
    paired["target_host"] = args.target_host
    paired["direction"] = f"{args.source_host} to {args.target_host}"
    paired["joint_state"] = paired.apply(
        classify_joint_state,
        axis=1,
        threshold=args.joint_threshold,
    )

    both_nonmissing = paired[
        ["HA_target_major_fraction", "NA_target_major_fraction"]
    ].dropna()
    correlation = both_nonmissing["HA_target_major_fraction"].corr(
        both_nonmissing["NA_target_major_fraction"], method="pearson"
    )
    ha_high = both_nonmissing["HA_target_major_fraction"].ge(args.strict_high)
    na_high = both_nonmissing["NA_target_major_fraction"].ge(args.strict_high)
    ha_low = both_nonmissing["HA_target_major_fraction"].lt(args.strict_low)
    na_low = both_nonmissing["NA_target_major_fraction"].lt(args.strict_low)
    strict_discordant = (ha_high & na_low) | (na_high & ha_low)

    ha_frac = pd.to_numeric(
        both_nonmissing["HA_target_major_fraction"], errors="coerce"
    )
    na_frac = pd.to_numeric(
        both_nonmissing["NA_target_major_fraction"], errors="coerce"
    )
    corr_perm_p = pearson_permutation_p(ha_frac, na_frac)
    corr_ci_low, corr_ci_high = pearson_bootstrap_ci(ha_frac, na_frac)
    save_paired_scatter(both_nonmissing, args.out_dir)
    thr_sensitivity = joint_state_threshold_sensitivity(paired)
    pd.DataFrame(
        [
            {
                "threshold": k,
                "joint_state": str(state),
                "fraction": float(frac),
            }
            for k, states in thr_sensitivity.items()
            for state, frac in states.items()
        ]
    ).to_csv(
        args.out_dir / "joint_state_threshold_sensitivity.csv",
        index=False,
    )

    paired.to_csv(args.out_dir / "paired_table.csv", index=False)

    clean_intersection = len(valid_keys) - len(conflicting)
    paired_unique = paired[key_col].nunique()
    if not (
        clean_intersection == len(paired) == paired_unique
    ):
        raise AssertionError(
            "Pair-count invariant failed: clean intersection, row count "
            "and paired-key count must be identical"
        )

    summary = {
        "pipeline_version": "HA-NA-strict-public-v2.0",
        "pair_key_col": key_col,
        "protein_accession_used_as_pair_key": False,
        "n_HA_input_rows": int(len(ha_master)),
        "n_NA_input_rows": int(len(na_master)),
        "n_HA_unique_once_keys": int(len(ha_unique)),
        "n_NA_unique_once_keys": int(len(na_unique)),
        "n_intersection_any": int(len(any_intersection)),
        "n_duplicate_or_many_to_many_intersection_keys": int(
            len(duplicate_intersection)
        ),
        "n_unique_key_intersection_before_conflict_filter": int(
            len(valid_keys)
        ),
        "n_conflicting_keys_excluded": int(len(conflicting)),
        "n_unique_key_intersection": int(clean_intersection),
        "n_paired_rows": int(len(paired)),
        "paired_key_nunique": int(paired_unique),
        "pair_count_invariant_passed": True,
        "expected_subtype": args.expected_subtype,
        "source_host": args.source_host,
        "target_host": args.target_host,
        "direction": f"{args.source_host} to {args.target_host}",
        "top_k_sites_requested": int(args.top_k_sites),
        "HA_sites_used": ha_sites,
        "NA_sites_used": na_sites,
        "joint_threshold": float(args.joint_threshold),
        "joint_state_counts": paired["joint_state"].value_counts().to_dict(),
        "correlation_method": "Pearson",
        "fraction_correlation": None if pd.isna(correlation) else float(correlation),
        "correlation_n": int(len(both_nonmissing)),
        "correlation_permutation_p": (
            None if pd.isna(corr_perm_p) else float(corr_perm_p)
        ),
        "correlation_bootstrap_ci_low": (
            None if pd.isna(corr_ci_low) else float(corr_ci_low)
        ),
        "correlation_bootstrap_ci_high": (
            None if pd.isna(corr_ci_high) else float(corr_ci_high)
        ),
        "joint_state_threshold_sensitivity": thr_sensitivity,
        "strict_high_threshold_inclusive": float(args.strict_high),
        "strict_low_threshold_exclusive": float(args.strict_low),
        "strict_discordant_total": int(strict_discordant.sum()),
        "strict_discordant_fraction": float(strict_discordant.mean()),
        "strict_concordance_fraction": float(1.0 - strict_discordant.mean()),
    }
    (args.out_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
