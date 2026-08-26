#!/usr/bin/env python3
"""
Map candidate-site positions from an MSA/alignment coordinate system to
ungapped reference-sequence coordinates.

Why this is needed:
  Candidate positions produced after multiple sequence alignment are often
  alignment columns. Alignment columns include gap characters, so an alignment
  column is not necessarily equal to the residue number in a real protein
  sequence. This script creates a column-to-residue map using a chosen reference
  sequence in the same alignment.

Inputs:
  --alignment_fasta       MSA FASTA file with equal-length aligned sequences
  --candidate_csv         Table containing candidate positions, e.g. Table S8
  --position_col          Candidate position column, default: position
  --reference_id          Exact reference sequence ID in the alignment, OR
  --reference_contains    A substring used to identify the reference ID

Outputs:
  candidate_sites_mapped_to_reference.csv
  reference_alignment_position_map.csv

Position modes:
  alignment_column        candidate positions are 1-based MSA columns
  reference_position      candidate positions are already ungapped reference
                          residue positions; script still checks whether the
                          reference sequence contains this residue position.

Optional:
  --map_all_sequences     also outputs long per-sequence ungapped coordinates
                          for each candidate alignment column; can be large.
"""

import argparse
from pathlib import Path
import re
import sys
import pandas as pd

GAP_CHARS = set(["-", ".", "~"])


def read_fasta(path):
    records = []
    name = None
    seq_parts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if name is not None:
            records.append((name, "".join(seq_parts)))
    if not records:
        raise ValueError("No FASTA records found: %s" % path)
    lens = {len(s) for _, s in records}
    if len(lens) != 1:
        raise ValueError("Alignment FASTA sequences are not equal length. Lengths: %s" % sorted(lens)[:10])
    return records


def choose_reference(records, reference_id=None, reference_contains=None):
    if reference_id:
        hits = [(n, s) for n, s in records if n == reference_id]
        if len(hits) == 1:
            return hits[0]
        raise ValueError("reference_id not found exactly or not unique: %r. Matches=%d" % (reference_id, len(hits)))
    if reference_contains:
        hits = [(n, s) for n, s in records if reference_contains in n]
        if len(hits) == 1:
            return hits[0]
        raise ValueError("reference_contains is not unique: %r. Matches=%d. First hits=%s" % (reference_contains, len(hits), [h[0] for h in hits[:5]]))
    # fallback: first sequence
    return records[0]


def build_reference_map(ref_name, ref_seq):
    rows = []
    ref_pos = 0
    for idx0, aa in enumerate(ref_seq):
        col = idx0 + 1
        is_gap = aa in GAP_CHARS
        if not is_gap:
            ref_pos += 1
            pos_val = ref_pos
        else:
            pos_val = None
        rows.append({
            "alignment_column": col,
            "reference_id": ref_name,
            "reference_aa_aligned": aa,
            "is_gap_in_reference": bool(is_gap),
            "reference_position": pos_val,
        })
    return pd.DataFrame(rows)


def map_all_sequence_positions(records, candidate_columns):
    candidate_columns = sorted(set(int(x) for x in candidate_columns if pd.notna(x)))
    output = []
    for name, seq in records:
        pos = 0
        col_to_pos = {}
        col_to_aa = {}
        for i, aa in enumerate(seq, start=1):
            if aa not in GAP_CHARS:
                pos += 1
                col_to_pos[i] = pos
            else:
                col_to_pos[i] = None
            col_to_aa[i] = aa
        for col in candidate_columns:
            output.append({
                "sequence_id": name,
                "alignment_column": col,
                "sequence_position": col_to_pos.get(col),
                "sequence_aa_aligned": col_to_aa.get(col),
                "is_gap_in_sequence": col_to_aa.get(col) in GAP_CHARS if col in col_to_aa else True,
            })
    return pd.DataFrame(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment_fasta", required=True)
    ap.add_argument("--candidate_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--position_col", default="position")
    ap.add_argument("--position_mode", default="alignment_column", choices=["alignment_column", "reference_position"])
    ap.add_argument("--reference_id", default=None)
    ap.add_argument("--reference_contains", default=None)
    ap.add_argument("--map_all_sequences", action="store_true")
    ap.add_argument("--strip_accession_version", action="store_true", help="Only affects all-sequence output sequence_id_short column")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_fasta(args.alignment_fasta)
    ref_name, ref_seq = choose_reference(records, args.reference_id, args.reference_contains)
    refmap = build_reference_map(ref_name, ref_seq)
    refmap.to_csv(out_dir / "reference_alignment_position_map.csv", index=False)

    cand = pd.read_csv(args.candidate_csv, keep_default_na=False, na_filter=False)
    if args.position_col not in cand.columns:
        raise ValueError("position_col %r not found in candidate_csv. Columns=%s" % (args.position_col, list(cand.columns)))
    cand[args.position_col] = pd.to_numeric(cand[args.position_col], errors="coerce")

    if args.position_mode == "alignment_column":
        mapdf = refmap.rename(columns={"alignment_column": args.position_col})
        merged = cand.merge(mapdf[[args.position_col, "reference_id", "reference_position", "reference_aa_aligned", "is_gap_in_reference"]], on=args.position_col, how="left")
        merged = merged.rename(columns={args.position_col: "alignment_position"})
    else:
        # Candidate positions are already reference positions; recover matching columns.
        pos_to_col = refmap.dropna(subset=["reference_position"]).copy()
        pos_to_col["reference_position"] = pos_to_col["reference_position"].astype(int)
        cand = cand.rename(columns={args.position_col: "reference_position"})
        cand["reference_position"] = pd.to_numeric(cand["reference_position"], errors="coerce")
        merged = cand.merge(pos_to_col[["reference_position", "alignment_column", "reference_id", "reference_aa_aligned", "is_gap_in_reference"]], on="reference_position", how="left")
        merged = merged.rename(columns={"alignment_column": "alignment_position"})

    # Make readable labels.
    merged["reference_position_label"] = merged.apply(
        lambda r: "unmapped_gap_or_missing" if pd.isna(r.get("reference_position")) else f"{int(r['reference_position'])}{str(r.get('reference_aa_aligned',''))}",
        axis=1,
    )
    merged.to_csv(out_dir / "candidate_sites_mapped_to_reference.csv", index=False)

    if args.map_all_sequences:
        if args.position_mode == "alignment_column":
            candidate_cols = merged["alignment_position"].dropna().astype(int).tolist()
        else:
            candidate_cols = merged["alignment_position"].dropna().astype(int).tolist()
        longdf = map_all_sequence_positions(records, candidate_cols)
        if args.strip_accession_version:
            longdf["sequence_id_short"] = longdf["sequence_id"].astype(str).str.replace(r"\.\d+(\s|$)", r"\1", regex=True)
        longdf.to_csv(out_dir / "candidate_sites_per_sequence_position_long.csv", index=False)

    n_total = len(merged)
    n_mapped = merged["reference_position"].notna().sum()
    print("Reference:", ref_name)
    print("Alignment length:", len(ref_seq))
    print("Ungapped reference length:", int(refmap["reference_position"].max()))
    print("Candidate sites:", n_total)
    print("Mapped to reference residues:", n_mapped)
    print("Unmapped/gap in reference:", n_total - n_mapped)
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
