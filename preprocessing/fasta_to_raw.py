import os
from collections import defaultdict

# Twenty standard amino acids.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_fasta(input_fasta, cleaned_fasta, report_file):
    """
    Clean FASTA input by removing BOM/whitespace artifacts and invalid
    amino-acid characters, then write normalized FASTA output.
    """
    illegal_chars = defaultdict(int)
    cleaned = []
    title = None
    seq = []

    with open(input_fasta, "r", encoding="utf-8") as fin:
        for line in fin:
            # Remove BOM and surrounding whitespace.
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

            # Clean sequence line.
            clean_line = []
            for ch in line.upper():
                if ch in VALID_AA:
                    clean_line.append(ch)
                else:
                    illegal_chars[ch] += 1
            seq.extend(clean_line)

    # Save the final sequence.
    if title and seq:
        cleaned.append((title, "".join(seq)))

    # Write normalized FASTA.
    with open(cleaned_fasta, "w", encoding="utf-8") as fout:
        for t, s in cleaned:
            fout.write(f"{t}\n")
            for i in range(0, len(s), 60):
                fout.write(s[i:i+60] + "\n")

    # Cleaning report.
    with open(report_file, "w", encoding="utf-8") as fr:
        fr.write("Invalid-character counts:\n")
        if not illegal_chars:
            fr.write("No invalid characters\n")
        else:
            for ch, cnt in illegal_chars.items():
                fr.write(f"{ch}: {cnt}\n")
        fr.write(f"\nSequences retained after cleaning: {len(cleaned)}\n")

    print(f"[OK] FASTA cleaning complete → {cleaned_fasta}")


def fasta_to_raw(cleaned_fasta, raw_output, max_len=1500):
    """
    FASTA → RAW
    Write one sequence segment per line, split segments longer than max_len,
    and separate amino acids with tabs.
    """
    with open(cleaned_fasta, "r", encoding="utf-8") as fin, \
         open(raw_output, "w", encoding="utf-8") as fout:

        seq = []

        for line in fin:
            line = line.strip()

            if line.startswith(">"):
                # Write the previous sequence.
                if seq:
                    # Split into segments no longer than max_len.
                    for i in range(0, len(seq), max_len):
                        chunk = seq[i:i + max_len]
                        fout.write("\t".join(chunk) + "\n")
                seq = []
            else:
                seq.extend(list(line))

        # Final sequence.
        if seq:
            for i in range(0, len(seq), max_len):
                chunk = seq[i:i + max_len]
                fout.write("\t".join(chunk) + "\n")

    print(f"[OK] RAW file generated → {raw_output}")


# ------------------------------
# Command-line entry point.
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert protein FASTA to ARNLE-compatible RAW format.")
    parser.add_argument("--input_fasta", required=True, help="Input protein FASTA file.")
    parser.add_argument("--output_prefix", default=None, help="Optional output prefix. Default: input path.")
    parser.add_argument("--max_len", type=int, default=1500, help="Maximum sequence length.")
    args = parser.parse_args()

    input_fasta = args.input_fasta
    prefix = args.output_prefix if args.output_prefix else input_fasta
    cleaned_fasta = prefix + ".cleaned.fasta"
    report_file   = prefix + ".clean_report.txt"
    raw_output    = prefix + ".raw"

    clean_fasta(input_fasta, cleaned_fasta, report_file)
    fasta_to_raw(cleaned_fasta, raw_output, max_len=args.max_len)

    print("\nProcessing finished.")
    print(f"Output cleaned FASTA: {cleaned_fasta}")
    print(f"Output RAW file: {raw_output}")
