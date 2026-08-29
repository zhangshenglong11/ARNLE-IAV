from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: statsmodels is required. Install with: pip install statsmodels"
    ) from exc

try:
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: statsmodels.stats.multitest.multipletests is required."
    ) from exc


LAYER_CODE = {
    "source-near": 0,
    "source_near": 0,
    "source near": 0,
    "source": 0,
    "near_source": 0,
    "source-near layer": 0,

    "center": 1,
    "centre": 1,
    "middle": 1,
    "mid": 1,
    "central": 1,

    "target-near": 2,
    "target_near": 2,
    "target near": 2,
    "target": 2,
    "near_target": 2,
    "target-near layer": 2,
}

ORDERED_LAYER_NAMES = {
    0: "source-near",
    1: "center",
    2: "target-near",
}


SITE_METADATA_CANDIDATES = [
    "protein",
    "subtype",
    "direction",
    "analysis_label",
    "position",
    "site_change",
    "source_major_aa",
    "target_major_aa",
    "reference_position",
    "msa_column",
]


@dataclass
class LogisticResult:
    beta: float
    pvalue: float
    status: str
    ci_low: float = np.nan
    ci_high: float = np.nan
    or_value: float = np.nan
    or_ci_low: float = np.nan
    or_ci_high: float = np.nan


def read_csv_keep_na(path: str | Path) -> pd.DataFrame:
    """Read CSV while preserving literal 'NA' as influenza neuraminidase."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_position(value) -> str:
    """
    Normalize position values to string keys.

    Position IDs are kept as strings to avoid losing suffixes such as 14A.
    Numeric values like 14.0 are converted to "14".
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_aa(value) -> str:
    """Normalize amino acid symbols, preserving gaps as '-'."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def layer_to_code(value) -> Optional[int]:
    """Map a transition-layer label or code to 0/1/2."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1, 2) else None

    if isinstance(value, float) and value.is_integer():
        return int(value) if int(value) in (0, 1, 2) else None

    text = str(value).strip().lower()
    if text in LAYER_CODE:
        return LAYER_CODE[text]

    # Common patterns in older tables.
    text = text.replace("_", "-")
    if text in LAYER_CODE:
        return LAYER_CODE[text]

    return None


def detect_site_columns(df: pd.DataFrame, fixed_columns: set[str]) -> list[str]:
    """
    Detect site-state columns for wide-format input.

    Accepted forms:
        site_14
        site14
        pos_14
        14
        14A
    """
    site_cols = []
    for col in df.columns:
        if col in fixed_columns:
            continue
        text = str(col)
        if re.fullmatch(r"(site|pos|position)[_\-]?[A-Za-z0-9\.]+", text, re.IGNORECASE):
            site_cols.append(col)
        elif re.fullmatch(r"\d+[A-Za-z]?", text):
            site_cols.append(col)
        elif re.fullmatch(r"\d+\.0", text):
            site_cols.append(col)
    return site_cols


def site_column_to_position(col: str) -> str:
    text = str(col).strip()
    text = re.sub(r"^(site|pos|position)[_\-]?", "", text, flags=re.IGNORECASE)
    return normalize_position(text)


def long_from_wide(site_state: pd.DataFrame, candidate_sites: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sequence-site-state matrix into long table."""
    required_state = ["analysis_label", "sequence_id", "transition_layer"]
    required_candidates = ["analysis_label", "position", "target_major_aa"]
    require_columns(site_state, required_state, "site-state wide table")
    require_columns(candidate_sites, required_candidates, "candidate-site table")

    fixed = set(required_state)
    site_cols = detect_site_columns(site_state, fixed)
    if not site_cols:
        raise ValueError(
            "No site columns were detected in wide input. Expected columns like site_14 or 14."
        )

    rename_map = {col: site_column_to_position(col) for col in site_cols}
    melted = site_state.melt(
        id_vars=required_state,
        value_vars=site_cols,
        var_name="site_column",
        value_name="aa",
    )
    melted["position"] = melted["site_column"].map(rename_map)
    melted = melted.drop(columns=["site_column"])

    candidates = candidate_sites.copy()
    candidates["position"] = candidates["position"].map(normalize_position)

    # Keep one target major amino acid and metadata row per analysis_label-position.
    metadata_cols = [c for c in SITE_METADATA_CANDIDATES if c in candidates.columns]
    if "analysis_label" not in metadata_cols:
        metadata_cols.append("analysis_label")
    if "position" not in metadata_cols:
        metadata_cols.append("position")
    if "target_major_aa" not in metadata_cols:
        metadata_cols.append("target_major_aa")

    candidates = (
        candidates[metadata_cols]
        .drop_duplicates(subset=["analysis_label", "position"], keep="first")
    )

    out = melted.merge(
        candidates,
        on=["analysis_label", "position"],
        how="inner",
        validate="many_to_one",
    )

    if out.empty:
        raise ValueError(
            "After merging wide site-state data with candidate-sites, no rows remained. "
            "Check analysis_label and position naming."
        )
    return out


def clean_long_input(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "analysis_label",
        "sequence_id",
        "transition_layer",
        "position",
        "aa",
        "target_major_aa",
    ]
    require_columns(df, required, "site-state long table")

    out = df.copy()
    out["analysis_label"] = out["analysis_label"].astype(str).str.strip()
    out["sequence_id"] = out["sequence_id"].astype(str).str.strip()
    out["position"] = out["position"].map(normalize_position)
    out["aa"] = out["aa"].map(normalize_aa)
    out["target_major_aa"] = out["target_major_aa"].map(normalize_aa)
    out["layer_code"] = out["transition_layer"].map(layer_to_code)

    bad_layers = out["layer_code"].isna().sum()
    if bad_layers:
        bad_examples = (
            out.loc[out["layer_code"].isna(), "transition_layer"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{bad_layers} rows have unrecognized transition_layer values. "
            f"Examples: {bad_examples}"
        )

    out["layer_code"] = out["layer_code"].astype(int)
    out = out[out["analysis_label"] != ""]
    out = out[out["sequence_id"] != ""]
    out = out[out["position"] != ""]
    out = out[out["aa"] != ""]
    out = out[out["target_major_aa"] != ""]

    if out.empty:
        raise ValueError("No valid rows remained after cleaning long input.")
    return out


def fit_logistic_trend(y: pd.Series, x: pd.Series, min_total: int, min_per_layer: int) -> LogisticResult:
    """Fit y ~ ordered layer by statsmodels Logit."""
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)

    valid = y.notna() & x.notna()
    y = y[valid]
    x = x[valid]

    if len(y) < min_total:
        return LogisticResult(np.nan, np.nan, "too_few_total_sequences")

    layer_counts = x.value_counts()
    for layer in (0, 1, 2):
        if int(layer_counts.get(layer, 0)) < min_per_layer:
            return LogisticResult(np.nan, np.nan, "too_few_sequences_in_one_or_more_layers")

    if y.nunique(dropna=True) < 2:
        return LogisticResult(np.nan, np.nan, "no_variation")

    if x.nunique(dropna=True) < 2:
        return LogisticResult(np.nan, np.nan, "no_variation")

    X = sm.add_constant(x.to_numpy(dtype=float), has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = sm.Logit(y.to_numpy(dtype=float), X)
            res = model.fit(disp=False, maxiter=200)
        beta = float(res.params[1])
        pvalue = float(res.pvalues[1])
        if not math.isfinite(beta) or not math.isfinite(pvalue):
            return LogisticResult(np.nan, np.nan, "non_finite_fit")
        conf = res.conf_int()
        ci_low = float(conf[1][0])
        ci_high = float(conf[1][1])
        or_value = math.exp(beta)
        or_ci_low = math.exp(ci_low)
        or_ci_high = math.exp(ci_high)
        if (
            not math.isfinite(ci_low)
            or not math.isfinite(ci_high)
            or (ci_high - ci_low) > 10.0
        ):
            return LogisticResult(
                beta, pvalue, "unstable_wide_ci_likely_quasi_separation",
                ci_low, ci_high, or_value, or_ci_low, or_ci_high,
            )
        return LogisticResult(
            beta, pvalue, "ok",
            ci_low, ci_high, or_value, or_ci_low, or_ci_high,
        )
    except PerfectSeparationError:
        return LogisticResult(np.nan, np.nan, "perfect_separation")
    except np.linalg.LinAlgError:
        return LogisticResult(np.nan, np.nan, "singular_matrix")
    except Exception as exc:
        return LogisticResult(np.nan, np.nan, f"fit_failed:{type(exc).__name__}")


def bootstrap_effect_ci(
    y: np.ndarray,
    layer: np.ndarray,
    n_iter: int = 1000,
    seed: int = 12345,
    ci: float = 0.95,
):
    """Bootstrap CI for the effect size = target-near freq - source-near freq."""
    rng = np.random.default_rng(seed)
    n = int(len(y))
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        tb = y[idx]
        ly = layer[idx]
        src = tb[ly == 0]
        tgt = tb[ly == 2]
        if len(src) == 0 or len(tgt) == 0:
            continue
        deltas.append(float(tgt.mean()) - float(src.mean()))
    if len(deltas) < 200:
        return np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def summarize_site(group: pd.DataFrame, min_total: int, min_per_layer: int) -> dict:
    """Calculate frequency summary and logistic trend result for one candidate site."""
    g = group.copy()

    g["target_major_binary"] = (
        g["aa"].map(normalize_aa) == g["target_major_aa"].map(normalize_aa)
    ).astype(int)

    row = {}

    # Preserve common site-level metadata.
    for col in SITE_METADATA_CANDIDATES:
        if col in g.columns:
            values = g[col].dropna().astype(str).unique()
            row[col] = values[0] if len(values) else ""

    row["analysis_label"] = str(g["analysis_label"].iloc[0])
    row["position"] = normalize_position(g["position"].iloc[0])
    row["target_major_aa"] = normalize_aa(g["target_major_aa"].iloc[0])

    # Counts and frequencies by layer.
    row["n_total_sequences"] = int(len(g))
    for code, label in ORDERED_LAYER_NAMES.items():
        sub = g[g["layer_code"] == code]
        n = int(len(sub))
        k = int(sub["target_major_binary"].sum()) if n else 0
        freq = float(k / n) if n else np.nan
        safe = label.replace("-", "_")
        row[f"n_{safe}"] = n
        row[f"target_major_count_{safe}"] = k
        row[f"{safe}_target_major_freq"] = freq

    src_freq = row.get("source_near_target_major_freq", np.nan)
    cen_freq = row.get("center_target_major_freq", np.nan)
    tgt_freq = row.get("target_near_target_major_freq", np.nan)

    row["delta_target_near_minus_source_near"] = (
        float(tgt_freq - src_freq)
        if pd.notna(src_freq) and pd.notna(tgt_freq)
        else np.nan
    )
    row["monotonic_target_major_increase"] = (
        bool(src_freq <= cen_freq <= tgt_freq)
        if pd.notna(src_freq) and pd.notna(cen_freq) and pd.notna(tgt_freq)
        else False
    )

    fit = fit_logistic_trend(
        y=g["target_major_binary"],
        x=g["layer_code"],
        min_total=min_total,
        min_per_layer=min_per_layer,
    )

    row["logistic_beta_refit"] = fit.beta
    row["logistic_beta_ci_low"] = fit.ci_low
    row["logistic_beta_ci_high"] = fit.ci_high
    row["odds_ratio"] = fit.or_value
    row["odds_ratio_ci_low"] = fit.or_ci_low
    row["odds_ratio_ci_high"] = fit.or_ci_high
    row["logistic_refit_status"] = fit.status
    row["trend_p"] = fit.pvalue
    row["effect_size_ci_low"], row["effect_size_ci_high"] = bootstrap_effect_ci(
        y=g["target_major_binary"].to_numpy(),
        layer=g["layer_code"].to_numpy(),
    )
    return row


def add_bh_fdr_within_analysis(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Add Benjamini-Hochberg FDR correction within each analysis_label.

    Invalid or missing P values keep missing q values.
    """
    out = df.copy()
    out["trend_q_BH_within_analysis"] = np.nan
    out["FDR_reject_alpha"] = False
    out["FDR_status"] = "no_valid_p"

    for analysis_label, idx in out.groupby("analysis_label", dropna=False).groups.items():
        idx = list(idx)
        p = pd.to_numeric(out.loc[idx, "trend_p"], errors="coerce")
        valid_mask = p.notna() & np.isfinite(p.to_numpy(dtype=float))
        valid_index = p[valid_mask].index

        if len(valid_index) == 0:
            out.loc[idx, "FDR_status"] = "no_valid_p"
            continue

        reject, qvals, _, _ = multipletests(
            p.loc[valid_index].to_numpy(dtype=float),
            alpha=alpha,
            method="fdr_bh",
        )
        out.loc[valid_index, "trend_q_BH_within_analysis"] = qvals
        out.loc[valid_index, "FDR_reject_alpha"] = reject
        out.loc[valid_index, "FDR_status"] = "ok"

        invalid_index = [i for i in idx if i not in valid_index]
        if invalid_index:
            out.loc[invalid_index, "FDR_status"] = "invalid_or_missing_p"

    return out


def run_analysis(args: argparse.Namespace) -> None:
    site_state = read_csv_keep_na(args.site_state)

    if args.format == "long":
        long_df = site_state
    elif args.format == "wide":
        if not args.candidate_sites:
            raise ValueError("--candidate-sites is required when --format wide is used.")
        candidate_sites = read_csv_keep_na(args.candidate_sites)
        long_df = long_from_wide(site_state, candidate_sites)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    long_df = clean_long_input(long_df)

    group_cols = ["analysis_label", "position"]
    rows = []
    for _, group in long_df.groupby(group_cols, dropna=False, sort=True):
        rows.append(
            summarize_site(
                group=group,
                min_total=args.min_total,
                min_per_layer=args.min_per_layer,
            )
        )

    result = pd.DataFrame(rows)
    result = add_bh_fdr_within_analysis(result, alpha=args.alpha)

    # Put important columns first when present.
    preferred_order = [
        "protein",
        "subtype",
        "direction",
        "analysis_label",
        "position",
        "site_change",
        "source_major_aa",
        "target_major_aa",
        "n_total_sequences",
        "n_source_near",
        "n_center",
        "n_target_near",
        "target_major_count_source_near",
        "target_major_count_center",
        "target_major_count_target_near",
        "source_near_target_major_freq",
        "center_target_major_freq",
        "target_near_target_major_freq",
        "delta_target_near_minus_source_near",
        "effect_size_ci_low",
        "effect_size_ci_high",
        "monotonic_target_major_increase",
        "logistic_beta_refit",
        "logistic_beta_ci_low",
        "logistic_beta_ci_high",
        "odds_ratio",
        "odds_ratio_ci_low",
        "odds_ratio_ci_high",
        "logistic_refit_status",
        "trend_p",
        "trend_q_BH_within_analysis",
        "FDR_reject_alpha",
        "FDR_status",
    ]
    ordered = [c for c in preferred_order if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    result = result[ordered + remaining]

    write_csv(result, args.output)

    n_sites = len(result)
    n_ok = int((result["logistic_refit_status"] == "ok").sum())
    n_valid_q = int(result["trend_q_BH_within_analysis"].notna().sum())
    print(f"Saved: {args.output}")
    print(f"Sites analyzed: {n_sites}")
    print(f"Logistic fits OK: {n_ok}")
    print(f"Valid BH-FDR q values: {n_valid_q}")


def validate_fdr(args: argparse.Namespace) -> None:
    """
    Validate whether trend_q_BH_within_analysis equals BH correction of trend_p
    within each analysis_label in an existing candidate-site summary table.
    """
    df = read_csv_keep_na(args.table)
    require_columns(
        df,
        ["analysis_label", "trend_p", "trend_q_BH_within_analysis"],
        "candidate-site summary table",
    )

    calc = df.copy()
    calc["_row_id"] = np.arange(len(calc))
    calc["_trend_p_numeric"] = pd.to_numeric(calc["trend_p"], errors="coerce")
    calc["_trend_q_existing_numeric"] = pd.to_numeric(
        calc["trend_q_BH_within_analysis"], errors="coerce"
    )

    calc["_trend_q_recomputed"] = np.nan

    reports = []
    for analysis_label, idx in calc.groupby("analysis_label", dropna=False).groups.items():
        idx = list(idx)
        p = calc.loc[idx, "_trend_p_numeric"]
        valid = p.notna() & np.isfinite(p.to_numpy(dtype=float))
        valid_idx = p[valid].index

        if len(valid_idx) > 0:
            _, qvals, _, _ = multipletests(
                p.loc[valid_idx].to_numpy(dtype=float),
                alpha=args.alpha,
                method="fdr_bh",
            )
            calc.loc[valid_idx, "_trend_q_recomputed"] = qvals

        existing = calc.loc[valid_idx, "_trend_q_existing_numeric"]
        recomputed = calc.loc[valid_idx, "_trend_q_recomputed"]

        if len(valid_idx) == 0:
            max_abs_diff = np.nan
            n_mismatch = 0
            status = "no_valid_p"
        else:
            diff = (existing - recomputed).abs()
            max_abs_diff = float(diff.max()) if len(diff) else np.nan
            n_mismatch = int((diff > args.tolerance).sum())
            status = "ok" if n_mismatch == 0 else "mismatch"

        reports.append(
            {
                "analysis_label": analysis_label,
                "n_rows": len(idx),
                "n_valid_p": len(valid_idx),
                "n_missing_p": int(calc.loc[idx, "_trend_p_numeric"].isna().sum()),
                "max_abs_q_difference": max_abs_diff,
                "n_q_mismatch": n_mismatch,
                "status": status,
            }
        )

    report = pd.DataFrame(reports).sort_values("analysis_label")

    if args.output:
        write_csv(report, args.output)
        print(f"Saved validation report: {args.output}")

    n_groups = len(report)
    n_ok = int((report["status"].isin(["ok", "no_valid_p"])).sum())
    n_mismatch = int((report["status"] == "mismatch").sum())

    print(f"Analysis groups checked: {n_groups}")
    print(f"Groups passing BH-FDR validation: {n_ok}")
    print(f"Groups with q mismatches: {n_mismatch}")

    if n_mismatch > 0:
        print("WARNING: Some analysis groups did not match existing q values.", file=sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ARNLE-IAV ordered-layer logistic trend test with within-analysis "
            "Benjamini-Hochberg FDR correction."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run logistic trend tests from sequence-level site-state input.",
    )
    run.add_argument(
        "--site-state",
        required=True,
        help="CSV containing sequence-level site states, either long or wide format.",
    )
    run.add_argument(
        "--format",
        choices=["long", "wide"],
        default="long",
        help="Input format of --site-state. Default: long.",
    )
    run.add_argument(
        "--candidate-sites",
        default=None,
        help=(
            "Candidate-site summary CSV. Required for wide input. "
            "Must contain analysis_label, position, target_major_aa."
        ),
    )
    run.add_argument(
        "--output",
        required=True,
        help="Output CSV path for site-level trend statistics.",
    )
    run.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR alpha for BH correction. Default: 0.05.",
    )
    run.add_argument(
        "--min-total",
        type=int,
        default=30,
        help="Minimum total sequence count required to fit logistic model. Default: 30.",
    )
    run.add_argument(
        "--min-per-layer",
        type=int,
        default=1,
        help=(
            "Minimum sequence count required in each transition layer for a site. "
            "Default: 1. Use 20 or 30 for stricter analyses."
        ),
    )
    run.set_defaults(func=run_analysis)

    validate = subparsers.add_parser(
        "validate-fdr",
        help=(
            "Validate that trend_q_BH_within_analysis in an existing table equals "
            "BH correction of trend_p within each analysis_label."
        ),
    )
    validate.add_argument(
        "--table",
        required=True,
        help="Existing candidate-site summary table, such as final Table S8 CSV.",
    )
    validate.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path for validation report.",
    )
    validate.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR alpha for BH correction. Default: 0.05.",
    )
    validate.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Allowed absolute numerical difference for q-value comparison. Default: 1e-12.",
    )
    validate.set_defaults(func=validate_fdr)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
