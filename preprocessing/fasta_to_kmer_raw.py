#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean protein FASTA sequences and convert them into non-overlapping k-mer RAW
files for ELMo tokenization experiments. By default, k values 2 through 10 are
written. Tokens are tab-separated, and the final remainder shorter than k is
retained unless --drop_remainder is specified.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

# Twenty standard amino acids.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_fasta(input_fasta: str, cleaned_fasta: str, report_file: str) -> int:
    """
    Clean FASTA input by removing BOM/whitespace artifacts and non-standard
    amino-acid characters, then write normalized FASTA and a cleaning report.

    Returns
    -------
    int
        Number of retained sequences.
    """
    illegal_chars: Dict[str, int] = defaultdict(int)
    cleaned: List[Tuple[str, str]] = []
    title = None
    seq: List[str] = []

    with open(input_fasta, "r", encoding="utf-8") as fin:
        for line in fin:
            # Remove BOM, line breaks, and surrounding whitespace.
            line = line.replace("\ufeff", "").rstrip("\n").strip()

            if not line:
                continue

            # FASTA header line.
            if line.startswith(">") or line.lstrip().startswith(">"):
                if title and seq:
                    cleaned.append((title, "".join(seq)))
                title = line.lstrip()
                seq = []
                continue

            # Keep only the 20 standard amino acids.
            for ch in line.upper():
                if ch in VALID_AA:
                    seq.append(ch)
                else:
                    illegal_chars[ch] += 1

    # Save the final sequence.
    if title and seq:
        cleaned.append((title, "".join(seq)))

    # Write normalized FASTA.
    with open(cleaned_fasta, "w", encoding="utf-8") as fout:
        for t, s in cleaned:
            fout.write(f"{t}\n")
            for i in range(0, len(s), 60):
                fout.write(s[i:i + 60] + "\n")

    # Write the cleaning report.
    with open(report_file, "w", encoding="utf-8") as fr:
        fr.write("Invalid-character counts:\n")
        if not illegal_chars:
            fr.write("No invalid characters\n")
        else:
            for ch, cnt in sorted(illegal_chars.items()):
                fr.write(f"{repr(ch)}: {cnt}\n")
        fr.write(f"\nSequences retained after cleaning: {len(cleaned)}\n")

    print(f"[OK] FASTA cleaning complete → {cleaned_fasta}")
    print(f"[OK] Cleaning report written → {report_file}")
    return len(cleaned)


def iter_fasta_sequences(cleaned_fasta: str) -> Iterable[Tuple[str, str]]:
    """
    Iterate over sequences in a cleaned FASTA file.
    """
    title = None
    seq: List[str] = []

    with open(cleaned_fasta, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if title is not None and seq:
                    yield title, "".join(seq)
                title = line
                seq = []
            else:
                seq.append(line)

    if title is not None and seq:
        yield title, "".join(seq)


def split_to_kmer_tokens(seq: str, k: int, keep_remainder: bool = True) -> List[str]:
    """
    Split a sequence into non-overlapping k-mer tokens.

    Examples
    --------
    seq = "ABCDEFGHIJ", k = 3, keep_remainder = True
        -> ["ABC", "DEF", "GHI", "J"]

    seq = "ABCDEFGHIJ", k = 3, keep_remainder = False
        -> ["ABC", "DEF", "GHI"]
    """
    tokens = []
    for i in range(0, len(seq), k):
        token = seq[i:i + k]
        if len(token) == k or keep_remainder:
            tokens.append(token)
    return tokens


def write_tokens_in_chunks(
    tokens: List[str],
    fout,
    max_tokens_per_line: int,
) -> int:
    """
    Write token lists in chunks of at most max_tokens_per_line tokens.

    Returns
    -------
    int
        Number of RAW lines written.
    """
    n_lines = 0
    for i in range(0, len(tokens), max_tokens_per_line):
        chunk = tokens[i:i + max_tokens_per_line]
        if chunk:
            fout.write("\t".join(chunk) + "\n")
            n_lines += 1
    return n_lines


def fasta_to_raw_kmer(
    cleaned_fasta: str,
    raw_output: str,
    k: int,
    max_tokens_per_line: int = 1500,
    keep_remainder: bool = True,
) -> Dict[str, int]:
    """
    Convert cleaned FASTA to k-mer RAW format.

    Parameters
    ----------
    cleaned_fasta : str
        Path to the cleaned FASTA file.
    raw_output : str
        Output RAW file path.
    k : int
        Number of amino acids per token.
    max_tokens_per_line : int
        Maximum number of k-mer tokens per RAW line.
    keep_remainder : bool
        Whether to retain the final remainder shorter than k.

    Returns
    -------
    dict
        Return conversion statistics.
    """
    n_seq = 0
    n_raw_lines = 0
    n_tokens = 0

    with open(raw_output, "w", encoding="utf-8") as fout:
        for _, seq in iter_fasta_sequences(cleaned_fasta):
            n_seq += 1
            tokens = split_to_kmer_tokens(seq, k=k, keep_remainder=keep_remainder)
            n_tokens += len(tokens)
            n_raw_lines += write_tokens_in_chunks(
                tokens=tokens,
                fout=fout,
                max_tokens_per_line=max_tokens_per_line,
            )

    print(
        f"[OK] k={k:<2d} RAW file generated → {raw_output} "
        f"| sequences={n_seq}, raw_lines={n_raw_lines}, tokens={n_tokens}"
    )

    return {
        "k": k,
        "sequences": n_seq,
        "raw_lines": n_raw_lines,
        "tokens": n_tokens,
    }


def batch_fasta_to_raw_kmer(
    cleaned_fasta: str,
    output_prefix: str,
    ks: List[int],
    max_tokens_per_line: int = 1500,
    keep_remainder: bool = True,
) -> List[Dict[str, int]]:
    """
    Generate RAW files for multiple k values.
    """
    stats = []
    for k in ks:
        raw_output = f"{output_prefix}.k{k}.raw"
        stat = fasta_to_raw_kmer(
            cleaned_fasta=cleaned_fasta,
            raw_output=raw_output,
            k=k,
            max_tokens_per_line=max_tokens_per_line,
            keep_remainder=keep_remainder,
        )
        stats.append(stat)
    return stats


def write_batch_summary(summary_file: str, stats: List[Dict[str, int]]) -> None:
    """
    Write the batch-conversion summary table.
    """
    with open(summary_file, "w", encoding="utf-8") as fout:
        fout.write("k\tsequences\traw_lines\ttokens\n")
        for stat in stats:
            fout.write(
                f"{stat['k']}\t{stat['sequences']}\t"
                f"{stat['raw_lines']}\t{stat['tokens']}\n"
            )
    print(f"[OK] Batch conversion summary written → {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean protein FASTA and convert it to multiple k-mer RAW files."
    )
    parser.add_argument(
        "--input_fasta",
        required=True,
        help="Input FASTA file path.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(range(2, 11)),
        help="k-mer sizes to generate; default: 2 3 4 5 6 7 8 9 10.",
    )
    parser.add_argument(
        "--max_tokens_per_line",
        type=int,
        default=1500,
        help="Maximum k-mer tokens per RAW line; default 1500.",
    )
    parser.add_argument(
        "--drop_remainder",
        action="store_true",
        help="Drop the final sequence remainder when its length is shorter than k.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_fasta = args.input_fasta
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input FASTA file does not exist: {input_fasta}")

    # Validate k values.
    ks = sorted(set(args.ks))
    if any(k < 1 for k in ks):
        raise ValueError(f"k values must be positive integers; received: {args.ks}")

    cleaned_fasta = input_fasta + ".cleaned.fasta"
    report_file = input_fasta + ".clean_report.txt"
    output_prefix = input_fasta
    summary_file = input_fasta + ".kmer_raw_summary.tsv"

    print("Starting FASTA-to-k-mer RAW conversion")
    print(f"Input FASTA: {input_fasta}")
    print(f"k values: {ks}")
    print(f"Maximum tokens per line: {args.max_tokens_per_line}")
    print(f"Keep final remainder shorter than k: {not args.drop_remainder}")
    print("-" * 80)

    clean_fasta(input_fasta, cleaned_fasta, report_file)

    stats = batch_fasta_to_raw_kmer(
        cleaned_fasta=cleaned_fasta,
        output_prefix=output_prefix,
        ks=ks,
        max_tokens_per_line=args.max_tokens_per_line,
        keep_remainder=not args.drop_remainder,
    )
    write_batch_summary(summary_file, stats)

    print("\nProcessing complete.")
    print(f"Cleaned FASTA: {cleaned_fasta}")
    print(f"Cleaning report: {report_file}")
    print(f"Batch summary: {summary_file}")
    print("RAW output files:")
    for k in ks:
        print(f"  - {output_prefix}.k{k}.raw")


if __name__ == "__main__":
    main()
