#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign the candidate-site evidence-level field used in Table S8.

The rule is a documented reconstruction because the original Table S8 generator
was not recovered. It has been validated against all 543 exported records.

Sequential rule:
1. negative_or_reverse: effect_size <= -0.10
2. exploratory: -0.10 < effect_size < 0.10
3. high: effect_size >= 0.25 AND q <= 0.05 AND trajectory_score >= 1.00
4. moderate: all remaining rows

The execution order is part of the definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "delta_target_near_minus_source_near",
    "trend_q_BH_within_analysis",
    "trajectory_score",
}


def classify_evidence_level(
    effect_size: float,
    q_value: float,
    trajectory_score: float,
) -> str:
    """Return the reconstructed Figure 5 / Table S8 evidence tier."""
    if effect_size <= -0.10:
        return "negative_or_reverse"
    if effect_size < 0.10:
        return "exploratory"
    if (
        effect_size >= 0.25
        and q_value <= 0.05
        and trajectory_score >= 1.00
    ):
        return "high"
    return "moderate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument(
        "--rule-version",
        default="TableS8-evidence-v1.0-reconstructed-543of543",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="raise")

    reconstructed = [
        classify_evidence_level(effect, q_value, score)
        for effect, q_value, score in zip(
            df["delta_target_near_minus_source_near"],
            df["trend_q_BH_within_analysis"],
            df["trajectory_score"],
        )
    ]

    df["evidence_level_reconstructed"] = reconstructed
    df["evidence_level_rule_version"] = args.rule_version
    df["evidence_level_rule_provenance"] = (
        "reconstructed_from_exported_Table_S8; exact_543_of_543_validation"
    )

    validation = {
        "n_rows": int(len(df)),
        "rule_version": args.rule_version,
        "existing_evidence_level_present": bool("evidence_level" in df.columns),
    }

    if "evidence_level" in df.columns:
        matches = df["evidence_level"].astype(str).eq(
            df["evidence_level_reconstructed"]
        )
        validation.update(
            {
                "n_matching_existing_labels": int(matches.sum()),
                "n_mismatches": int((~matches).sum()),
                "match_rate": float(matches.mean()),
            }
        )
        if not matches.all():
            mismatch_path = args.validation_output.with_name(
                args.validation_output.stem + "_mismatches.csv"
            )
            df.loc[~matches].to_csv(mismatch_path, index=False)
            raise RuntimeError(
                f"Reconstructed tiers did not match all existing labels. "
                f"See {mismatch_path}"
            )

    # Make the reconstructed field the public final field while preserving any
    # prior exported field for audit.
    if "evidence_level" in df.columns:
        df = df.rename(columns={"evidence_level": "evidence_level_exported"})
    df["evidence_level"] = df["evidence_level_reconstructed"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    args.validation_output.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
