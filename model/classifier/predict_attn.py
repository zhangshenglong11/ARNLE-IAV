import os
import math
import sys
import argparse
import glob
import re

import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm

tf.disable_v2_behavior()
tf.disable_eager_execution()


def _token_length(aa_length, n_split, max_length=None):
    """Match the keep-remainder tokenization used by elmo_embedding_stream.py."""
    if n_split <= 0:
        raise ValueError("n_split must be a positive integer")
    token_length = (aa_length + n_split - 1) // n_split
    if max_length is not None and max_length > 0:
        token_length = min(token_length, max_length)
    return token_length


def fasta_token_lengths(path, n_split, max_length=None):
    """
    Parse FASTA by records (header starts with '^>'), supports multi-line sequences.
    Retain a final token shorter than n_split and clip lengths to the embedding
    time dimension, matching model/embedding/elmo_embedding_stream.py.
    """
    lengths = []
    cur_len = 0
    seen = False

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if line == "":
                continue

            if line.startswith(">"):  # record boundary
                if seen:
                    lengths.append(_token_length(cur_len, n_split, max_length))
                seen = True
                cur_len = 0
            else:
                if not seen:
                    # ignore any preamble before first header
                    continue
                # remove all whitespace inside sequence lines
                cur_len += len("".join(line.split()))

    if seen:
        lengths.append(_token_length(cur_len, n_split, max_length))

    return lengths


def discover_embedding_files(path):
    """
    Return all embedding chunks represented by path.

    A path ending in _partNNN.npy discovers every matching chunk. A base .npy
    path also prefers matching _part*.npy files and otherwise falls back to the
    exact file. This mirrors the classifier-training input behavior.
    """
    base_dir, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".npy":
        raise ValueError("--data must point to a .npy embedding file")

    match = re.match(r"^(.*)_part\d+$", name)
    chunk_stem = match.group(1) if match else name
    pattern = os.path.join(base_dir, "{}_part*{}".format(chunk_stem, ext))
    files = sorted(glob.glob(pattern))
    if files:
        return files
    if os.path.exists(path):
        return [path]
    raise FileNotFoundError("No matching embedding files found: {} or {}".format(pattern, path))


def get_batch(x, y, batchsize, length, training):
    perm = np.arange(x.shape[0])
    y = np.asarray(y)
    length = np.asarray(length)

    if training:
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]
        length = length[perm]

    numbatch = int(math.ceil(float(x.shape[0]) / float(batchsize)))
    for i in range(numbatch):
        start = i * batchsize
        end = start + batchsize
        batchx = x[start:end]
        batchy = y[start:end]

        # Critical safety: ensure seq_length vector matches REAL batch size (esp. last batch)
        bs = batchx.shape[0]
        batchlength = length[start:start + bs]

        yield batchx, batchy, batchlength


parser = argparse.ArgumentParser('Predict IAV host adaptation with Bi-LSTM model')
parser.add_argument('--model_path', type=str, required=True, help='model path of Bi-LSTM model')
parser.add_argument('--num_class', type=int, default=6, help='amount of label classes')
parser.add_argument('--data', type=str, required=True,
                    help='embedding .npy file, base path, or _part000.npy path; matching chunks are loaded automatically')
parser.add_argument('--batchsize', type=int, default=256, help='batch size in predicting')
parser.add_argument('--file', type=str, required=True, help='FASTA file (for sequence length)')
parser.add_argument('--n_split', type=int, default=6, help='amount of amino acid in a token')
parser.add_argument('--out_path', type=str, required=True, help='output path: directory or *.npy file')
parser.add_argument('--attn_name', type=str, default='attn_out.npy',
                    help='Backward-compatible filename for the attention-pooled sequence feature output')
args = parser.parse_args(sys.argv[1:])

ckpt = tf.train.get_checkpoint_state(args.model_path)
if ckpt is None or ckpt.model_checkpoint_path is None:
    raise RuntimeError("No checkpoint found under: {}".format(args.model_path))

saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    g = tf.get_default_graph()

    x = g.get_operation_by_name('x').outputs[0]
    y = g.get_operation_by_name('y').outputs[0]
    keep_prob = g.get_operation_by_name('keep_prob').outputs[0]
    seq_length = g.get_operation_by_name('seq_length').outputs[0]
    logits = g.get_operation_by_name('output/logits').outputs[0]
    prediction = g.get_operation_by_name('output/prediction').outputs[0]
    pooled_feature = g.get_operation_by_name('Attention/attn').outputs[0]
    # Attention/attn is the attention-pooled sequence representation, not the
    # per-position attention-weight matrix (alpha).
    pooled_feature_batches = []

    # Define softmax ONCE on the Tensor (avoid graph growth inside loop)
    softmax_logits = tf.nn.softmax(logits)

    embedding_files = discover_embedding_files(args.data)
    data_chunks = [np.load(path, mmap_mode="r") for path in embedding_files]
    if not data_chunks:
        raise RuntimeError("No embedding data were loaded")
    if any(chunk.ndim != 3 for chunk in data_chunks):
        raise ValueError("Every embedding chunk must have shape [sequences, max_length, features]")

    max_length = int(data_chunks[0].shape[1])
    feature_size = int(data_chunks[0].shape[2])
    for path, chunk in zip(embedding_files, data_chunks):
        if int(chunk.shape[1]) != max_length or int(chunk.shape[2]) != feature_size:
            raise ValueError(
                "Embedding chunk shape mismatch for {}: expected (*, {}, {}), got {}".format(
                    path, max_length, feature_size, chunk.shape
                )
            )
    total_rows = int(sum(chunk.shape[0] for chunk in data_chunks))

    # Robust FASTA parsing by record boundary
    length_test = fasta_token_lengths(args.file, args.n_split, max_length=max_length)

    # ===== Quick self-check (before long inference) =====
    header_count = sum(1 for line in open(args.file, "r") if line.startswith(">"))
    print("[CHECK] embedding files:", len(embedding_files))
    print("[CHECK] total embedding rows:", total_rows)
    print("[CHECK] embedding time/features:", max_length, feature_size)
    print("[CHECK] len(length_test):", len(length_test))
    print("[CHECK] header_count(^>):", header_count)
    if len(length_test) > 0:
        print("[CHECK] length_test min/max:", min(length_test), max(length_test))

    # Hard alignment checks
    if len(length_test) != total_rows:
        raise RuntimeError(
            "FASTA and embedding not aligned: len(length_test)={} != total embedding rows={}".format(
                len(length_test), total_rows
            )
        )
    if len(length_test) != header_count:
        raise RuntimeError(
            "FASTA parse mismatch: len(length_test)={} != header_count={}".format(
                len(length_test), header_count
            )
        )

    first_bs = min(args.batchsize, data_chunks[0].shape[0])
    print("[CHECK] first batch shapes:", data_chunks[0][:first_bs].shape, np.asarray(length_test[:first_bs]).shape)
    # ==============================================

    prob_list = []
    pred_list = []

    total_batches = sum(
        int(math.ceil(float(chunk.shape[0]) / float(args.batchsize)))
        for chunk in data_chunks
    )
    row_offset = 0
    with tqdm(total=total_batches) as pbar:
        for chunk_path, data in zip(embedding_files, data_chunks):
            chunk_rows = int(data.shape[0])
            chunk_lengths = length_test[row_offset:row_offset + chunk_rows]
            # y is not used for inference; feed safe zeros for the placeholder.
            chunk_labels = np.zeros((chunk_rows,), dtype=np.int32)
            print("[INFO] Predicting embedding chunk:", chunk_path, "rows=", chunk_rows)

            for batchx, batchy, batchlen in get_batch(
                data, chunk_labels, args.batchsize, chunk_lengths, False
            ):
                batchlen = np.asarray(batchlen, dtype=np.int32)

                if batchx.shape[0] != batchlen.shape[0]:
                    raise RuntimeError(
                        "Batch mismatch: x batch = {}, seq_length = {}".format(
                            batchx.shape[0], batchlen.shape[0]
                        )
                    )

                feed_dict_test = {x: batchx, y: batchy, keep_prob: 1.0, seq_length: batchlen}
                batch_prob, batch_pred, batch_feature = sess.run(
                    [softmax_logits, prediction, pooled_feature],
                    feed_dict=feed_dict_test,
                )
                print("pooled feature shape:", batch_feature.shape, "dtype:", batch_feature.dtype)
                pooled_feature_batches.append(batch_feature)

                prob_list.append(batch_prob)
                pred_list.append(batch_pred)
                pbar.update(1)
            row_offset += chunk_rows

    prob = np.concatenate(prob_list, axis=0) if prob_list else np.zeros((0, args.num_class), dtype=np.float32)
    pred_test = np.concatenate(pred_list, axis=0) if pred_list else np.zeros((0,), dtype=np.int32)
    pooled_features = (
        np.concatenate(pooled_feature_batches, axis=0)
        if pooled_feature_batches else np.zeros((0,), dtype=np.float32)
    )

    # Output handling:
    # - if out_path endswith .npy -> save exactly there
    # - else treat as directory -> save prediction_probability.npy under it
    if args.out_path.endswith(".npy"):
        out_file = args.out_path
        out_dir = os.path.dirname(out_file)
        if out_dir and (not os.path.exists(out_dir)):
            os.makedirs(out_dir)
        np.save(out_file, prob)
        # Preserve --attn_name for compatibility with existing downstream commands.
        out_dir = os.path.dirname(out_file) or "."
        np.save(os.path.join(out_dir, args.attn_name), pooled_features)
    else:
        outpath = args.out_path if args.out_path.endswith("/") else (args.out_path + "/")
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        np.save(os.path.join(outpath, "prediction_probability.npy"), prob)
        np.save(os.path.join(outpath, args.attn_name), pooled_features)

    print(pred_test[pred_test == 0].shape)
