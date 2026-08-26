#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Influenza A virus protein extraction and metadata normalization.

The script extracts HA, NA, PB1, PB2, PA, NP, M1, M2, and NS proteins from
NCBI protein FASTA files. NS combines NS1 and NS2/NEP records. FASTA headers
are normalized to include accession, host category, collection date, and
country. Metadata are parsed from FASTA headers first and can be supplemented
from GenPept records. Host categories are artiodactyla, primates, aves, and
other.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from Bio import SeqIO

TARGET_GROUPS = ["HA", "NA", "PB1", "PB2", "PA", "NP", "M1", "M2", "NS"]
MAX_YEAR = 2025


# -------------------- Protein classification --------------------

def classify_protein(desc: str) -> Optional[str]:
    """
    Classify a FASTA description into HA, NA, PB1, PB2, PA, NP, M1, or M2.
    Return None when no target protein is recognized.
    """
    s = (desc or "").lower()

    # PB2 / PB1 / PA
    if "polymerase basic protein 2" in s or re.search(r"\bpb2\b", s):
        return "PB2"

    if "polymerase basic protein 1" in s or re.search(r"\bpb1\b", s):
        # Exclude PB1-F2.
        if "pb1-f2" in s or "pb1 f2" in s:
            return None
        return "PB1"

    # Retain the standalone PA pattern for compatibility.
    if "polymerase acidic" in s or re.search(r"\bpa\b", s):
        # Exclude PA-X.
        if "pa-x" in s or "pax" in s:
            return None
        return "PA"

    # HA / NA
    if "hemagglutinin" in s or "haemagglutinin" in s or re.search(r"\bha\b", s):
        return "HA"
    if "neuraminidase" in s or re.search(r"\bna\b", s):
        return "NA"

    # Include common NP synonyms such as nucleocapsid protein.
    if (
        "nucleoprotein" in s
        or "nucleocapsid protein" in s
        or "nucleocapsid" in s
        or re.search(r"\bnp\b", s)
    ):
        return "NP"

    # Detect M2 before generic matrix-protein patterns.
    if (
        "matrix protein 2" in s
        or re.search(r"\bm2\b", s)
        or "m2 protein" in s
        or ("ion channel" in s and "influenza" in s)
    ):
        return "M2"

    # Detect M1 while excluding M2/ion-channel records.
    if "matrix protein 1" in s or re.search(r"\bm1\b", s):
        return "M1"
    if "matrix protein" in s:
        if "matrix protein 2" in s or re.search(r"\bm2\b", s) or "ion channel" in s:
            return None
        return "M1"

    # NS1
    if (
        "nonstructural protein 1" in s
        or "non-structural protein 1" in s
        or re.search(r"\bns1\b", s)
    ):
        return "NS1"

    # NS2 / NEP / nuclear export protein
    if (
        "nuclear export protein" in s
        or "nonstructural protein 2" in s
        or "non-structural protein 2" in s
        or "nep" in s
        or re.search(r"\bns2\b", s)
    ):
        return "NS2"

    return None


# -------------------- Accession parsing --------------------

def parse_accession(fasta_id: str, fasta_desc: str) -> str:
    """
    Extract an accession from FASTA headers such as
    gi|2446897760|gb|WDK94173.1| nuclear export protein [Influenza A virus].
    """
    header = f"{fasta_id} {fasta_desc}".strip()
    m = re.search(r"\|(gb|ref|emb|dbj|sp|pdb)\|([^|]+)\|", header)
    if m:
        return m.group(2).strip()
    # Fallback when the identifier is already an accession.
    return (fasta_id or "UnknownAccession").strip()


def strip_version(acc: str) -> str:
    return acc.split(".", 1)[0] if acc else acc


# -------------------- Host normalization & category mapping --------------------

def normalize_host(host_raw: Optional[str]) -> str:
    """Normalize the host field."""
    if not host_raw:
        return "Unknown"

    s = host_raw.strip()
    if not s:
        return "Unknown"

    s = s.split(";", 1)[0]
    s = s.split(",", 1)[0]

    return s.strip() if s else "Unknown"


# ---------- primates ----------

PRIMATES_PATTERNS = [
    r"\bhomo sapiens\b",
    r"\bhuman\b",
    r"\bprimates?\b",
    r"\bmacaca\b",
    r"\bmacaque\b",
    r"\bmonkey\b",
]


# ---------- artiodactyla ----------

ARTIODACTYLA_PATTERNS = [
    r"\bartiodactyla\b",

    r"\bswine\b",
    r"\bpig\b",
    r"\bsus scrofa\b",

    r"\bcattle\b",
    r"\bcow\b",
    r"\bbovine\b",
    r"\bbos taurus\b",

    r"\bsheep\b",
    r"\bovis\b",

    r"\bgoat\b",
    r"\bcapra\b",

    r"\bdeer\b",
    r"\bcervus\b",
    r"\bcervidae\b",

    r"\bbuffalo\b",
    r"\bbison\b",

    r"\bcamel\b",
    r"\bcamelus\b",

    r"\balpaca\b",
    r"\bllama\b",

    r"\bwild boar\b",
]


# ---------- aves ----------

AVES_PATTERNS = [

    r"\bavian\b",

    r"\bchicken\b",
    r"\bgallus\b",

    r"\bturkey\b",
    r"\bmeleagris\b",

    r"\bduck\b",
    r"\banas\b",

    r"\bgoose\b",
    r"\bans[ae]r\b",

    r"\bswan\b",
    r"\bcygnus\b",

    r"\bquail\b",
    r"\bcolinus\b",

    r"\bpigeon\b",
    r"\bcolumba\b",

    r"\bwild bird\b",
    r"\bshorebird\b",
]


# ---------- other mammals / environment ----------

OTHER_PATTERNS = [

    r"\bequine\b",
    r"\bhorse\b",
    r"\bequus\b",

    r"\bdog\b",
    r"\bcanine\b",
    r"\bcanis\b",
    r"\braccoon dog\b",

    r"\bcat\b",
    r"\bfeline\b",
    r"\bfelis\b",

    r"\bmink\b",
    r"\bferret\b",
    r"\bmustela\b",
    r"\bweasel\b",

    r"\bfox\b",
    r"\bvulpes\b",

    r"\bseal\b",
    r"\bpinniped\b",

    r"\bbat\b",
    r"\bchiroptera\b",

    r"\bmouse\b",
    r"\bmus\b",

    r"\brat\b",
    r"\brattus\b",

    r"\brabbit\b",

    r"\benvironment\b",
    r"\benv\b",
    r"\bswab\b",
    r"\bsewage\b",
    r"\bfeces\b",
]


def host_to_category(host_norm: str) -> str:
    """
    Classification order: primates, artiodactyla, aves, then other.
    """

    if not host_norm or host_norm == "Unknown":
        return "other"

    s = host_norm.lower()

    for pat in PRIMATES_PATTERNS:
        if re.search(pat, s):
            return "primates"

    for pat in ARTIODACTYLA_PATTERNS:
        if re.search(pat, s):
            return "artiodactyla"

    for pat in AVES_PATTERNS:
        if re.search(pat, s):
            return "aves"

    for pat in OTHER_PATTERNS:
        if re.search(pat, s):
            return "other"

    # Default.
    return "other"


# -------------------- Date normalization (month precision + year<=2025) --------------------

MONTHS = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}


def _valid_year(year: int) -> bool:
    return 1700 <= year <= MAX_YEAR


def _valid_month(month: int) -> bool:
    return 1 <= month <= 12


def normalize_collection_date_to_month(date_raw: Optional[str]) -> str:
    """
    Normalize date strings to YYYY-MM, YYYY, or Unknown.
    """
    if not date_raw:
        return "Unknown"
    s = str(date_raw).strip()
    if not s or s.lower() in {"na", "n/a", "none", "unknown"}:
        return "Unknown"

    # Normalize separators.
    s_clean = s.replace("_", " ").replace(".", "-").strip()

    # 1) YYYY-MM, YYYY/MM, or YYYY-MM-DD.
    m = re.search(r"\b((?:18|19|20)\d{2})\s*[-/]\s*(\d{1,2})\b", s_clean)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        if _valid_year(y) and _valid_month(mo):
            return f"{y:04d}-{mo:02d}"
        return "Unknown"

    # 2) DD-Mon-YYYY or Mon-DD-YYYY.
    m = re.search(r"\b(\d{1,2})\s*[-/ ]\s*([A-Za-z]{3,9})\s*[-/ ]\s*((?:18|19|20)\d{2})\b", s_clean)
    if m:
        mon = MONTHS.get(m.group(2).lower())
        y = int(m.group(3))
        if mon and _valid_year(y):
            return f"{y:04d}-{mon}"
        return "Unknown"

    m = re.search(r"\b([A-Za-z]{3,9})\s*[-/ ]\s*(\d{1,2})\s*[-/ ]\s*((?:18|19|20)\d{2})\b", s_clean)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        y = int(m.group(3))
        if mon and _valid_year(y):
            return f"{y:04d}-{mon}"
        return "Unknown"

    # 3) Mon-YYYY or Mon YYYY.
    m = re.search(r"\b([A-Za-z]{3,9})\s*[-/ ]\s*((?:18|19|20)\d{2})\b", s_clean)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        y = int(m.group(2))
        if mon and _valid_year(y):
            return f"{y:04d}-{mon}"
        return "Unknown"

    # 4) Year only; restrict to plausible 18xx/19xx/20xx values.
    m = re.search(r"\b((?:18|19|20)\d{2})\b", s_clean)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return f"{y:04d}"
        return "Unknown"

    return "Unknown"


# -------------------- Metadata handling --------------------

@dataclass
class Meta:
    host: str = "Unknown"            # raw host (will be mapped to category at output)
    collection_date: str = "Unknown" # normalized
    country: str = "Unknown"

    def merge_missing_from(self, other: "Meta") -> "Meta":
        def pick(a: str, b: str) -> str:
            return a if (a and a != "Unknown") else (b if b else a)
        return Meta(
            host=pick(self.host, other.host),
            collection_date=pick(self.collection_date, other.collection_date),
            country=pick(self.country, other.country),
        )


def _normalize_value(v: Optional[str]) -> str:
    s = (v or "").strip()
    if not s:
        return "Unknown"
    if s.lower() in {"na", "n/a", "none"}:
        return "Unknown"
    # Remove pipe characters that would break the normalized header format.
    s = s.replace("|", " ").strip()
    return s if s else "Unknown"


def sanitize_field(v: str) -> str:
    """Normalize text for FASTA titles by trimming whitespace and replacing spaces with underscores."""
    v = _normalize_value(v)
    if v == "Unknown":
        return v
    v = re.sub(r"\s+", "_", v)
    return v


def infer_meta_from_header(desc: str) -> Meta:
    """
    Infer host, country, and date from explicit header fields or influenza
    strain-name patterns such as A/host/location/isolate/year.
    """
    host = "Unknown"
    country = "Unknown"
    collection_date = "Unknown"

    if not desc:
        return Meta()

    # 1) Explicit key-value fields.
    m_host = re.search(r"host\s*=\s*([^|\]]+)", desc, flags=re.IGNORECASE)
    if m_host:
        host = _normalize_value(m_host.group(1))

    m_country = re.search(r"country\s*=\s*([^|\]]+)", desc, flags=re.IGNORECASE)
    if m_country:
        country = _normalize_value(m_country.group(1))

    m_date = re.search(r"collection[_\s]*date\s*=\s*([^|\]]+)", desc, flags=re.IGNORECASE)
    if m_date:
        collection_date = normalize_collection_date_to_month(m_date.group(1))

    # 2) Influenza strain-name format, for example A/India/.../2017.
    name_match = re.search(r"\b([ABCD]/[^)\s\]]+)", desc)
    if name_match:
        raw = name_match.group(1).split("(", 1)[0]
        parts = raw.split("/")
        if len(parts) >= 4 and parts[0] in {"A", "B", "C", "D"}:
            year_candidate = parts[-1].strip()
            if re.fullmatch(r"(?:18|19|20)\d{2}", year_candidate) and collection_date == "Unknown":
                collection_date = normalize_collection_date_to_month(year_candidate)

            if country == "Unknown":
                # A/host/location/isolate/year -> location index 2
                # A/location/isolate/year -> location index 1
                if len(parts) >= 5:
                    country = _normalize_value(parts[2])
                else:
                    country = _normalize_value(parts[1])

            if host == "Unknown" and len(parts) >= 5:
                host = _normalize_value(parts[1])

    return Meta(
        host=_normalize_value(host),
        collection_date=_normalize_value(collection_date),
        country=_normalize_value(country),
    )


# -------------------- GenPept (.gp) indexing --------------------

def genbank_paths(gp_or_dir: Optional[str]) -> Iterable[Path]:
    """
    Accept a single GenPept file or a directory containing common GenPept/GenBank extensions.
    """
    if not gp_or_dir:
        return []
    p = Path(gp_or_dir)
    if p.is_file():
        return [p]
    if p.is_dir():
        exts = [
            "*.gp", "*.gpff", "*.genpept", "*.genpept.txt",
            "*.gb", "*.gbk", "*.gbff", "*.genbank", "*.gbf", "*.gbtxt", "*.gb*",
        ]
        files = []
        for pat in exts:
            files.extend(p.glob(pat))
        return sorted({f.resolve() for f in files if f.is_file()})
    return []


def meta_from_genbank_record(record) -> Meta:
    """
    Extract host, collection_date, and country/geo_loc_name from the source feature.
    """
    host = "Unknown"
    collection_date = "Unknown"
    country = "Unknown"

    for feat in getattr(record, "features", []) or []:
        if feat.type != "source":
            continue
        q = feat.qualifiers or {}

        host = q.get("host", [host])[0]

        # Country preference: /country else /geo_loc_name
        if "country" in q:
            country = q.get("country", [country])[0]
        elif "geo_loc_name" in q:
            country = q.get("geo_loc_name", [country])[0]
        elif "geographic location" in q:
            country = q.get("geographic location", [country])[0]

        # Keep only the country part of values such as "Country: subregion".
        if isinstance(country, str) and ":" in country:
            country = country.split(":", 1)[0]

        collection_date = q.get("collection_date", [collection_date])[0]
        break

    return Meta(
        host=_normalize_value(host),
        collection_date=_normalize_value(normalize_collection_date_to_month(collection_date)),
        country=_normalize_value(country),
    )


def build_gp_index(gp_or_dir: Optional[str], cache_path: Optional[str] = None) -> Dict[str, Meta]:
    """
    Build a metadata dictionary from GenPept records using record identifiers,
    accessions, and feature protein_id values. Version-stripped keys are stored
    for accession matching. If cache_path is supplied, load an existing cache
    or create and save one.
    """
    if cache_path:
        cp = Path(cache_path)
        if cp.exists():
            print(f"[INFO] Loading cached gp metadata: {cp}")
            with cp.open("rb") as f:
                return pickle.load(f)

    idx: Dict[str, Meta] = {}
    paths = list(genbank_paths(gp_or_dir))
    if not paths:
        return idx

    total_records = 0
    for fp in paths:
        try:
            # Parse GenPept records with Biopython's genbank parser.
            for record in SeqIO.parse(str(fp), "genbank"):
                total_records += 1
                m = meta_from_genbank_record(record)

                accs = set()
                if getattr(record, "id", None):
                    accs.add(str(record.id))
                if getattr(record, "name", None):
                    accs.add(str(record.name))
                if hasattr(record, "annotations"):
                    for a in record.annotations.get("accessions", []) or []:
                        accs.add(str(a))

                # Some records expose protein_id in feature qualifiers.
                for feat in getattr(record, "features", []) or []:
                    for pid in feat.qualifiers.get("protein_id", []) or []:
                        pid = str(pid).strip()
                        if pid:
                            accs.add(pid)

                for a in accs:
                    idx[a] = m
                    idx[strip_version(a)] = m

                if total_records % 200000 == 0:
                    print(f"[INFO] Parsed gp records: {total_records}  (current index size={len(idx)})")

        except Exception as e:
            print(f"[WARN] Failed to parse {fp}: {e}", file=sys.stderr)

    print(f"[INFO] Parsed gp total records: {total_records}")
    print(f"[INFO] GP index size: {len(idx)}")

    if cache_path:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving gp metadata cache: {cp}")
        with cp.open("wb") as f:
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    return idx


# -------------------- FASTA writing --------------------

def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def write_fasta_record(out_fp: Path, header: str, seq: str) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with out_fp.open("a", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        f.write(wrap_fasta(seq) + "\n")


# -------------------- Main --------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Input NCBI protein FASTA file")
    ap.add_argument("--gp", required=False, default=None, help="GenPept .gp file or directory used to supplement host/country/collection_date metadata")
    ap.add_argument("--outdir", required=True, help="Output directory for per-protein FASTA files (HA.fasta, ...)")
    ap.add_argument("--overwrite", action="store_true", help="If set, remove existing output FASTA files first.")
    ap.add_argument("--keep_version", action="store_true", help="Keep accession version in output (default: strip version).")
    ap.add_argument("--cache", default=None, help="Optional pickle cache path for gp metadata to speed up reruns.")
    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    outdir = Path(args.outdir)

    if not fasta_path.exists():
        print(f"[ERROR] FASTA not found: {fasta_path}", file=sys.stderr)
        return 2

    gp_idx = build_gp_index(args.gp, cache_path=args.cache) if args.gp else {}
    print(f"[DONE] Output dir: {outdir}")
    print(f"[DONE] Output groups: {', '.join(TARGET_GROUPS)}")

    if args.overwrite:
        for g in TARGET_GROUPS:
            fp = outdir / f"{g}.fasta"
            if fp.exists():
                fp.unlink()

    kept = 0
    unclassified = 0

    counts_by_group = {g: 0 for g in TARGET_GROUPS}
    # unknown-field counters per group
    unknown_by_group = {g: {"host": 0, "collection_date": 0, "country": 0} for g in TARGET_GROUPS}
    ns_breakdown = {"NS1": 0, "NS2": 0}

    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        acc = parse_accession(rec.id, rec.description)
        acc_out = acc if args.keep_version else strip_version(acc)

        group_raw = classify_protein(rec.description)
        if group_raw is None:
            unclassified += 1
            continue

        # Merge NS1 and NS2/NEP into the NS output group.
        if group_raw in {"NS1", "NS2"}:
            group = "NS"
        else:
            group = group_raw

        # 1) Metadata parsed from the FASTA header.
        meta = infer_meta_from_header(rec.description)

        # 2) Supplement missing metadata from GenPept.
        gb_meta = gp_idx.get(acc) or gp_idx.get(strip_version(acc))
        if gb_meta:
            meta = meta.merge_missing_from(gb_meta)

        # Normalize fields defensively.
        meta.collection_date = normalize_collection_date_to_month(meta.collection_date)

        host_norm = normalize_host(meta.host)
        # Classify hosts as primates/artiodactyla/aves/other; use Unknown when host metadata are absent.
        host_field = "Unknown" if host_norm == "Unknown" else host_to_category(host_norm)

        # Count final Unknown fields.
        if host_field == "Unknown":
            unknown_by_group[group]["host"] += 1
        if meta.collection_date == "Unknown":
            unknown_by_group[group]["collection_date"] += 1
        if meta.country == "Unknown":
            unknown_by_group[group]["country"] += 1

        header = (
            f">{sanitize_field(acc_out)} | "
            f"host={sanitize_field(host_field)} | "
            f"collection_date={sanitize_field(meta.collection_date)} | "
            f"Country={sanitize_field(meta.country)}"
        )

        out_fp = outdir / f"{group}.fasta"
        write_fasta_record(out_fp, header, str(rec.seq))

        kept += 1
        counts_by_group[group] += 1
        if group_raw in {"NS1", "NS2"}:
            ns_breakdown[group_raw] += 1

    print(f"[STATS] kept={kept}, unclassified={unclassified}")
    print("\n[COUNTS] Extracted sequences per protein group:")
    for g in TARGET_GROUPS:
        if g == "NS":
            print(f"{g}\t{counts_by_group[g]}\t(NS1={ns_breakdown['NS1']}, NS2={ns_breakdown['NS2']})")
        else:
            print(f"{g}\t{counts_by_group[g]}")

    print("\n[UNKNOWN] Unknown-field counts per protein group:")
    print("group\thost_unknown\tcollection_date_unknown\tcountry_unknown")
    for g in TARGET_GROUPS:
        u = unknown_by_group[g]
        print(f"{g}\t{u['host']}\t{u['collection_date']}\t{u['country']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
