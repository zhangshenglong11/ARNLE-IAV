# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import re
from elmoformanylangs import Embedder
import os

def fasta_iter(path):
    header = None
    seq_parts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lstrip().startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line.lstrip()[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.replace(" ", "").replace("\t", "").upper())
        if header is not None:
            yield header, "".join(seq_parts)

def count_fasta_records(path):
    n = 0
    for _h, _s in fasta_iter(path):
        n += 1
    return n

def split_tokens(seq, split_n):
    if split_n <= 1:
        return list(seq)
    return [seq[i:i + split_n] for i in range(0, len(seq), split_n)]

def truncate_before_forward(tokens, max_len):
    if max_len is None or max_len <= 0:
        return tokens
    return tokens[:max_len]

def pad_or_truncate(emb, max_len):
    emb = np.asarray(emb, dtype=np.float32)
    if max_len is None or max_len <= 0:
        return emb
    if emb.shape[0] >= max_len:
        return emb[:max_len]
    pad = np.zeros((max_len - emb.shape[0], emb.shape[1]), dtype=np.float32)
    return np.vstack([emb, pad])

def is_cuda_oom(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return ("cuda out of memory" in msg) or ("cudnn error" in msg and "alloc" in msg)

def sents2elmo_safe(elmo: Embedder, batch_sents):
    try:
        with torch.no_grad():
            return elmo.sents2elmo(batch_sents)
    except RuntimeError as e:
        if is_cuda_oom(e) and len(batch_sents) > 1:
            mid = len(batch_sents) // 2
            left = sents2elmo_safe(elmo, batch_sents[:mid])
            right = sents2elmo_safe(elmo, batch_sents[mid:])
            return left + right
        raise

def extract_label(title: str, pattern: str, group_id: int) -> str:
    m = re.search(pattern, title)
    if not m:
        return "Unknown"
    try:
        return m.group(group_id).strip()
    except IndexError:
        return "Unknown"

def main():
    ap = argparse.ArgumentParser("Protein FASTA -> ELMo embeddings (Chunked Version)")
    ap.add_argument("--file", required=True, help="Input FASTA file")
    ap.add_argument("--model_path", required=True, help="ELMo model directory")
    ap.add_argument("--output", required=True, help="Base output .npy path (will append _partXXX)")
    ap.add_argument("--batchsize", type=int, default=2, help="Batch size for inference")
    ap.add_argument("--max_length", type=int, default=264, help="Max token length")
    ap.add_argument("--split", type=int, default=1, help="Token size")
    ap.add_argument("--pool", choices=["none"], default="none", help="Must be none for Bi-LSTM")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N sequences (0=all)")
    ap.add_argument("--write_labels", action="store_true", help="Write labels")
    ap.add_argument("--label_regex", default=r"host=([^|]+)", help="Regex to extract label")
    ap.add_argument("--label_regex_group", type=int, default=1, help="Regex group id")
    
    # Chunk-size parameter.
    ap.add_argument("--chunk_size", type=int, default=20000, help="Number of sequences per .npy file to prevent OOM")

    args = ap.parse_args()

    total_records = count_fasta_records(args.file)
    if args.limit and args.limit > 0:
        total_records = min(total_records, args.limit)

    print(f"Total FASTA records to process: {total_records}")
    print(f"Data will be chunked into files of {args.chunk_size} records each.")

    # Split the base filename and extension.
    base_output_path, ext = os.path.splitext(args.output)
    if ext == '':
        ext = '.npy'

    titles_path = base_output_path + ".titles.txt"
    lengths_path = base_output_path + ".lengths.txt"
    labels_path = base_output_path + ".labels.txt"

    elmo = Embedder(args.model_path)
    try:
        if hasattr(elmo, "model"):
            elmo.model.eval()
    except Exception:
        pass

    idx = 0
    current_chunk_idx = -1
    mmap = None

    batch_tokens = []
    batch_titles = []
    batch_token_lens = []

    ft = open(titles_path, "w", encoding="utf-8")
    fl = open(lengths_path, "w", encoding="utf-8")
    fz = open(labels_path, "w", encoding="utf-8") if args.write_labels else None

    def process_and_save_batch(b_tokens, b_titles, b_lens):
        nonlocal idx, current_chunk_idx, mmap
        embs = sents2elmo_safe(elmo, b_tokens)
        
        for t, tl, e in zip(b_titles, b_lens, embs):
            # Create a new chunk file when the current chunk reaches its size limit.
            if idx % args.chunk_size == 0:
                if mmap is not None:
                    mmap.flush()
                current_chunk_idx += 1
                remain_records = total_records - idx
                cur_chunk_size = min(args.chunk_size, remain_records)
                
                chunk_filename = f"{base_output_path}_part{current_chunk_idx:03d}{ext}"
                chunk_shape = (cur_chunk_size, args.max_length, 1024)
                print(f"\n[Info] Creating chunk {current_chunk_idx}: {chunk_filename}, shape: {chunk_shape}")
                mmap = np.lib.format.open_memmap(chunk_filename, mode="w+", dtype=np.float32, shape=chunk_shape)
            
            # Compute the relative index within the current chunk.
            local_idx = idx % args.chunk_size
            
            e = np.asarray(e, dtype=np.float32)
            mat = pad_or_truncate(e, args.max_length)
            mmap[local_idx, :, :] = mat

            ft.write(t + "\n")
            fl.write(str(tl) + "\n")
            if fz is not None:
                lab = extract_label(t, args.label_regex, args.label_regex_group)
                fz.write(lab + "\n")

            idx += 1

    try:
        for header, seq in fasta_iter(args.file):
            if args.limit and args.limit > 0 and idx >= args.limit:
                break

            tokens = split_tokens(seq, args.split)
            tokens = truncate_before_forward(tokens, args.max_length)
            if len(tokens) == 0:
                continue

            batch_tokens.append(tokens)
            batch_titles.append(header)
            batch_token_lens.append(len(tokens))

            if len(batch_tokens) >= args.batchsize:
                process_and_save_batch(batch_tokens, batch_titles, batch_token_lens)
                batch_tokens, batch_titles, batch_token_lens = [], [], []
                if idx % 100 == 0:
                    print(f"\rProcessed: {idx}/{total_records}", end="")

        # Process the final partial batch.
        if batch_tokens and idx < total_records:
            process_and_save_batch(batch_tokens, batch_titles, batch_token_lens)
            print(f"\rProcessed: {idx}/{total_records}", end="")
            
    finally:
        if mmap is not None:
            mmap.flush()
        ft.close()
        fl.close()
        if fz is not None:
            fz.close()

    print("\nDone.")

if __name__ == "__main__":
    main()
