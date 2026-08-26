# -*- coding: utf-8 -*-
"""
Bi-LSTM host-classifier training and validation script for ARNLE-IAV.

Purpose
-------
This script keeps the original Bi-LSTM training logic unchanged and only adds
validation metric outputs comparable to Table S3 / Figure S3.

Kept unchanged relative to the original script:
1. get_batch() training/evaluation batching logic.
2. attention() implementation.
3. build_graph() implementation, including the original alpha return call.
4. optimizer, loss, dropout, Bi-LSTM architecture and output layer.
5. train loop update call: sess.run([train_step, loss, accuracy, prediction], ...).
6. max_length is taken directly from --max_length, not inferred from embedding shape.
7. sequence_length and length_val are read directly and are not clipped.

Allowed data-reading improvements:
1. label files may be plain host labels OR FASTA/title lines containing host=...
2. *_part000.npy input paths are expanded to all matching *_part*.npy chunks.

Outputs added after validation prediction:
- metrics_global.csv
- TableS3_host_level_metrics.csv
- TableS3_by_protein_metrics.csv, if --val_metadata_csv is provided
- TableS3_by_protein_subtype_metrics.csv, if --val_metadata_csv is provided
- confusion matrices
- FigureS3_host_level_performance.png/pdf
- training_epoch_summary.csv
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
import math
from sklearn import metrics
import sys
import argparse
import glob
import re
import json
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


tf.disable_v2_behavior()
tf.disable_eager_execution()

# =====================================================================
# Data wrapper. Training logic is unchanged; this only controls how .npy
# blocks are read from disk.
# =====================================================================
class ChunkedMemmapSequence:
    def __init__(self, base_path):
        base_dir, filename = os.path.split(base_path)
        name, ext = os.path.splitext(filename)

        # Robust chunk discovery. If user passes either:
        #   xxx.npy
        # or:
        #   xxx_part000.npy
        # this loads xxx_part000.npy, xxx_part001.npy, ... when present.
        m = re.match(r"^(.*)_part\d+$", name)
        chunk_stem = m.group(1) if m else name
        pattern = os.path.join(base_dir, "%s_part*%s" % (chunk_stem, ext))

        self.files = sorted(glob.glob(pattern))
        if not self.files:
            # Fallback to the exact file path.
            if os.path.exists(base_path):
                self.files = [base_path]
            else:
                raise FileNotFoundError("No matching files found: %s or %s" % (pattern, base_path))

        print("[Info] Loaded %d data chunks: %s" % (len(self.files), filename))
        if len(self.files) > 1:
            print("[Info] First data chunk:", self.files[0])
            print("[Info] Last data chunk:", self.files[-1])

        self.mmaps = [np.load(f, mmap_mode='r') for f in self.files]
        self.lengths = [m.shape[0] for m in self.mmaps]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = int(self.cumulative_lengths[-1])
        self.shape = (self.total_length, self.mmaps[0].shape[1], self.mmaps[0].shape[2])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(self.total_length))
        elif isinstance(idx, int):
            if idx < 0 or idx >= self.total_length:
                raise IndexError("Index out of bounds")
            chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
            local_idx = idx if chunk_idx == 0 else idx - self.cumulative_lengths[chunk_idx - 1]
            return self.mmaps[chunk_idx][local_idx]

        idx = np.asarray(idx)
        result = np.empty((len(idx), self.shape[1], self.shape[2]), dtype=self.mmaps[0].dtype)
        chunk_indices = np.searchsorted(self.cumulative_lengths, idx, side='right')
        for c in np.unique(chunk_indices):
            mask = (chunk_indices == c)
            global_idxs_in_chunk = idx[mask]
            if c == 0:
                local_idxs = global_idxs_in_chunk
            else:
                local_idxs = global_idxs_in_chunk - self.cumulative_lengths[c - 1]
            result[mask] = self.mmaps[c][local_idxs]
        return result




class IndexedSequence:
    """View over a sequence-like object using selected row indices.

    This is a data-reading filter only. It is used to drop rows whose title/label
    has host=Unknown/Other while keeping the Bi-LSTM graph and training loop unchanged.
    """
    def __init__(self, base, indices, name="data"):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.name = name
        self.shape = (len(self.indices), base.shape[1], base.shape[2])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            local = np.arange(*idx.indices(len(self.indices)))
            return self.base[self.indices[local]]
        elif isinstance(idx, int):
            if idx < 0 or idx >= len(self.indices):
                raise IndexError("Index out of bounds")
            return self.base[int(self.indices[idx])]
        else:
            local = np.asarray(idx)
            return self.base[self.indices[local]]


# =====================================================================
# Original training functions: intentionally kept unchanged.
# =====================================================================
def get_batch(x, y, batchsize, length, training):
    n = x.shape[0]
    y = np.asarray(y)
    length = np.asarray(length)

    perm = np.arange(n)
    if training:
        np.random.shuffle(perm)

    numbatch = math.ceil(n / batchsize)
    for i in range(numbatch):
        start = i * batchsize
        end = start + batchsize
        if training:
            idx = perm[start:end]
            batchx = np.asarray(x[idx])
            batchy = y[idx]
            batchlength = length[idx]
        else:
            batchx = np.asarray(x[start:end])
            batchy = y[start:end]
            batchlength = length[start:end]
        yield batchx, batchy, batchlength


def predict_in_batches(sess, g, x, y, length, batchsize):
    preds = []
    for batchx, batchy, batchlen in get_batch(x, y, batchsize, length, training=False):
        feed_dict = {g['x']: batchx, g['y']: batchy, g['keep_prob']: 1.0, g['seq_length']: batchlen}
        preds.append(sess.run(g['prediction'], feed_dict=feed_dict))
    if not preds:
        return np.array([], dtype=np.int32)
    return np.concatenate(preds, axis=0)


def attention(h, keep_prob):
    size = hidden_size[-1]
    w = tf.Variable(tf.random_normal([size], stddev=0.1, dtype=tf.float32))
    m = tf.tanh(h, name='m')
    newm = tf.matmul(tf.reshape(m, [-1, size]), tf.reshape(w, [-1, 1]), name='new_m')
    restorem = tf.reshape(newm, [-1, max_length], name='restore_m')
    alpha = tf.nn.softmax(restorem, name='alpha')
    r = tf.matmul(tf.transpose(h, [0, 2, 1]), tf.reshape(alpha, [-1, max_length, 1]), name='r')
    sequeeze_r = tf.reshape(r, [-1, size])
    repre = tf.tanh(sequeeze_r, name='attn')
    output = tf.nn.dropout(repre, keep_prob=keep_prob, name='h')
    return output, alpha


def build_graph():
    x = tf.placeholder(tf.float32, [None, max_length, 1024], name='x')
    y = tf.placeholder(tf.int32, [None], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    seq_length = tf.placeholder(tf.int32, name='seq_length')
    embedding = x

    with tf.name_scope('Bi_LSTM'):
        for idx, hiddensize in enumerate(hidden_size):
            with tf.name_scope('Bi-LSTM'+str(idx)):
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddensize),
                                                        output_keep_prob=keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddensize),
                                                        output_keep_prob=keep_prob)
                rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding, dtype=tf.float32,
                                                                scope='bi-lstm'+str(idx), sequence_length=seq_length)
                embedding = tf.concat(rnn_output, 2)
    rnn_output = tf.split(embedding, 2, -1)

    with tf.name_scope('Attention'):
        h = rnn_output[0]+rnn_output[1]
        output = attention(h, keep_prob)[0]
        outputsize = hidden_size[-1]

    with tf.name_scope('output'):
        output_w = tf.get_variable('output_w', shape=[outputsize, num_class], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   dtype=tf.float32)
        output_b = tf.Variable(tf.constant(0.1, shape=[num_class], dtype=tf.float32), name='output_b')
        logits = tf.nn.xw_plus_b(output, output_w, output_b, name='logits')
        prediction = tf.argmax(logits, axis=-1, name='prediction', output_type=tf.int32)

    with tf.name_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(losses)

    correct_predict = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    train_step = opt.minimize(loss, global_step=global_step)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(writer_path, tf.get_default_graph())
    return dict(x=x, y=y, keep_prob=keep_prob, loss=loss, train_step=train_step, merged=merged,
                train_writer=train_writer, saver=tf.train.Saver(), prediction=prediction, accuracy=accuracy,
                seq_length=seq_length, alpha=attention(h, keep_prob)[1], logits=logits)


# =====================================================================
# Added metric helpers. These do not affect training updates.
# =====================================================================
HOST_TO_ID = {"artiodactyla": 0, "primates": 1, "aves": 2}
ID_TO_HOST = {0: "artiodactyla", 1: "primates", 2: "aves"}
HOST_ORDER = ["artiodactyla", "primates", "aves"]
HOST_IDS = [0, 1, 2]
UNKNOWN_HOST_VALUES = set(["unknown", "unk", "na", "n/a", "none", "null", "", "not_available", "not-available", "unclassified", "other", "others", "misc", "miscellaneous"] )
HOST_SHORT = {"artiodactyla": "AR", "primates": "PR", "aves": "AV"}
HOST_COLORS = {"artiodactyla": "#4C78A8", "primates": "#F58518", "aves": "#59A14F"}
METRIC_COLORS = {"precision": "#4C78A8", "recall": "#F58518", "f1": "#59A14F"}


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def extract_host_label(raw):
    """Return normalized host label, or None for explicit unknown hosts.

    Accepted classes are artiodactyla, primates and aves. Title lines such as
    "ACC | host=aves | ..." are supported. Explicit host=Unknown/Other or host=Other is returned
    as None so the corresponding sequence row can be skipped.
    """
    text = str(raw).strip()
    low = text.lower().strip()

    if low in UNKNOWN_HOST_VALUES:
        return None
    if low in ("anseriformes", "galliformes"):
        return "aves"
    if low in HOST_TO_ID:
        return low

    # title mode: parse host=...
    m = re.search(r"(?:^|[|;,\s])host\s*=\s*([^|;,]+)", low)
    if m:
        h = m.group(1).strip().strip("'\"[](){}").lower()
        h = h.strip()
        if h in UNKNOWN_HOST_VALUES:
            return None
        if h in ("anseriformes", "galliformes"):
            h = "aves"
        if h in HOST_TO_ID:
            return h
        raise ValueError("Unknown host value parsed from title: %r in line %r" % (h, raw))

    # very conservative fallback: accept one unambiguous host token.
    found = []
    for h in HOST_ORDER:
        if re.search(r"\b%s\b" % re.escape(h), low):
            found.append(h)
    if len(found) == 1:
        return found[0]

    raise ValueError("Unknown host label/title: %r. Expected plain host labels or lines containing host=<artiodactyla|primates|aves|Unknown|Other>." % raw)


def map_label(raw):
    h = extract_host_label(raw)
    if h is None:
        return None
    return HOST_TO_ID[h]


def read_label_file(path, dataset_name="dataset", skip_unknown=True):
    labels = []
    keep_indices = []
    skipped_excluded = 0
    n_raw = 0
    with open(path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.rstrip("\n")
            if str(line).strip() == "":
                continue
            n_raw += 1
            lab = map_label(line)
            if lab is None:
                if skip_unknown:
                    skipped_excluded += 1
                    continue
                raise ValueError("Unknown/Other host encountered in %s at row %d: %r" % (path, row_idx, line))
            labels.append(lab)
            # row_idx is used as the embedding row index. This assumes one non-empty
            # title/label line per embedding row, which is how the ELMo title files are written.
            keep_indices.append(row_idx)
    counts = {h: labels.count(HOST_TO_ID[h]) for h in HOST_ORDER}
    print("[Info] Parsed labels from %s: raw=%d, kept=%d, skipped_excluded=%d, %s" % (path, n_raw, len(labels), skipped_excluded, counts))
    return labels, np.asarray(keep_indices, dtype=np.int64), {"n_raw": n_raw, "n_kept": len(labels), "n_skipped_excluded": skipped_excluded, "dataset": dataset_name}


def read_length_file(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            vals.append(int(line))
    return vals


def confusion_counts(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if int(t) in HOST_IDS and int(p) in HOST_IDS:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    tp = np.diag(cm)
    total = cm.sum()
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted_f1 = float(np.sum(f1 * support) / total) if total > 0 else 0.0
    balanced_accuracy = float(np.mean(recall)) if len(recall) else 0.0
    # For single-label multiclass classification, micro-F1 equals accuracy.
    micro_f1 = accuracy
    return {
        "n": int(total),
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
        "pred_count": pred_count.astype(int),
    }


def build_overall_row(metric_dict):
    row = {
        "n": metric_dict["n"],
        "accuracy": metric_dict["accuracy"],
        "micro_f1": metric_dict["micro_f1"],
        "macro_f1": metric_dict["macro_f1"],
        "weighted_f1": metric_dict["weighted_f1"],
        "balanced_accuracy": metric_dict["balanced_accuracy"],
    }
    for i, h in enumerate(HOST_ORDER):
        row["%s_precision" % h] = float(metric_dict["precision"][i])
        row["%s_recall" % h] = float(metric_dict["recall"][i])
        row["%s_f1" % h] = float(metric_dict["f1"][i])
        row["%s_support" % h] = int(metric_dict["support"][i])
        row["%s_pred_count" % h] = int(metric_dict["pred_count"][i])
    return row


def host_level_table(metric_dict):
    rows = []
    for i, h in enumerate(HOST_ORDER):
        rows.append({
            "host": h,
            "precision": float(metric_dict["precision"][i]),
            "recall": float(metric_dict["recall"][i]),
            "f1": float(metric_dict["f1"][i]),
            "support": int(metric_dict["support"][i]),
            "pred_count": int(metric_dict["pred_count"][i]),
        })
    return pd.DataFrame(rows)


def confusion_to_frames(cm):
    count_df = pd.DataFrame(cm, index=["true_%s" % h for h in HOST_ORDER], columns=["pred_%s" % h for h in HOST_ORDER])
    denom = cm.sum(axis=1, keepdims=True).astype(float)
    norm = np.divide(cm.astype(float), denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)
    norm_df = pd.DataFrame(norm, index=["true_%s" % h for h in HOST_ORDER], columns=["pred_%s" % h for h in HOST_ORDER])
    long_rows = []
    for i, th in enumerate(HOST_ORDER):
        for j, ph in enumerate(HOST_ORDER):
            long_rows.append({
                "true_host": th,
                "pred_host": ph,
                "count": int(cm[i, j]),
                "row_fraction": float(norm[i, j]),
            })
    return count_df, norm_df, pd.DataFrame(long_rows)


def load_validation_metadata(path, n_expected, protein_col, subtype_col, keep_indices=None, n_original=None):
    if not path:
        return None
    meta = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if keep_indices is not None and n_original is not None and len(meta) == int(n_original):
        meta = meta.iloc[np.asarray(keep_indices, dtype=np.int64)].reset_index(drop=True)
    if len(meta) != n_expected:
        raise ValueError("Validation metadata rows do not match filtered validation data rows: metadata=%d, validation=%d. If metadata is aligned to the original unfiltered validation rows, pass the same label/title file so unknown-host rows can be filtered consistently." % (len(meta), n_expected))
    if protein_col in meta.columns:
        meta[protein_col] = meta[protein_col].astype(str).replace({"nan": "NA", "NaN": "NA", "": "unknown"})
    if subtype_col in meta.columns:
        meta[subtype_col] = meta[subtype_col].astype(str).replace({"nan": "NA", "NaN": "NA", "": "unknown"})
    return meta


def infer_group_label(row, protein_col, subtype_col):
    protein = str(row.get(protein_col, "unknown")).replace("nan", "NA").replace("NaN", "NA")
    subtype = str(row.get(subtype_col, "unknown")).replace("nan", "NA").replace("NaN", "NA")
    if subtype == "" or subtype.lower() == "unknown":
        return protein
    return "%s_%s" % (protein, subtype)


def grouped_metrics(df, group_col, min_n):
    rows = []
    for group, sub in df.groupby(group_col):
        if len(sub) < int(min_n):
            continue
        cm = confusion_counts(sub["y_true"].values, sub["y_pred"].values)
        md = metrics_from_cm(cm)
        row = {group_col: group}
        row.update(build_overall_row(md))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def make_figure_s3(y_true, y_pred, host_df, pred_df, out_png, out_pdf=None, dpi=300):
    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(HOST_ORDER))
    width = 0.24
    for k, metric_name in enumerate(["precision", "recall", "f1"]):
        vals = [host_df.loc[host_df["host"] == h, metric_name].values[0] for h in HOST_ORDER]
        ax1.bar(x + (k - 1) * width, vals, width=width, color=METRIC_COLORS[metric_name], label=metric_name.capitalize())
        for xi, val in zip(x + (k - 1) * width, vals):
            ax1.text(xi, val + 0.015, "%.2f" % val, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(HOST_ORDER, rotation=20, ha="right")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score")
    ax1.set_title("A. Global host-level performance", loc="left", fontweight="bold")
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    ax1.grid(axis="y", alpha=0.18)

    if "protein" in pred_df.columns:
        protein_order = sorted(pred_df["protein"].dropna().unique().tolist())
        recall_mat = []
        f1_mat = []
        arti_rows = []
        for p in protein_order:
            sub = pred_df[pred_df["protein"] == p]
            cm = confusion_counts(sub["y_true"].values, sub["y_pred"].values)
            md = metrics_from_cm(cm)
            recall_mat.append(md["recall"])
            f1_mat.append(md["f1"])
            art_i = HOST_TO_ID["artiodactyla"]
            row_sum = cm[art_i, :].sum()
            if row_sum > 0:
                fractions = cm[art_i, :].astype(float) / row_sum
            else:
                fractions = np.zeros(3, dtype=float)
            arti_rows.append(fractions)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(np.asarray(recall_mat), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax2.set_xticks(np.arange(3))
        ax2.set_xticklabels(HOST_ORDER, rotation=25, ha="right")
        ax2.set_yticks(np.arange(len(protein_order)))
        ax2.set_yticklabels(protein_order)
        for i in range(len(protein_order)):
            for j in range(3):
                ax2.text(j, i, "%.2f" % recall_mat[i][j], ha="center", va="center", fontsize=8)
        ax2.set_title("B. Host-specific recall by protein", loc="left", fontweight="bold")
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
        cb2.set_label("Recall")

        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(np.asarray(f1_mat), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax3.set_xticks(np.arange(3))
        ax3.set_xticklabels(HOST_ORDER, rotation=25, ha="right")
        ax3.set_yticks(np.arange(len(protein_order)))
        ax3.set_yticklabels(protein_order)
        for i in range(len(protein_order)):
            for j in range(3):
                ax3.text(j, i, "%.2f" % f1_mat[i][j], ha="center", va="center", fontsize=8)
        ax3.set_title("C. Host-specific F1 by protein", loc="left", fontweight="bold")
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
        cb3.set_label("F1")

        ax4 = fig.add_subplot(gs[1, 1])
        host_labels = ["Pred AR", "Pred PR", "Pred AV"]
        left = np.zeros(len(protein_order), dtype=float)
        y_pos = np.arange(len(protein_order))
        for j, h in enumerate(HOST_ORDER):
            vals = [r[j] for r in arti_rows]
            ax4.barh(y_pos, vals, left=left, color=HOST_COLORS[h], label=host_labels[j])
            left += np.asarray(vals)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(protein_order)
        ax4.set_xlim(0, 1)
        ax4.set_xlabel("Fraction of true artiodactyla samples")
        ax4.set_title("D. True artiodactyla prediction profile", loc="left", fontweight="bold")
        ax4.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=8)
        ax4.grid(axis="x", alpha=0.18)
    else:
        # Fallback without metadata.
        for pos, label in [(gs[0, 1], "B"), (gs[1, 0], "C"), (gs[1, 1], "D")]:
            ax = fig.add_subplot(pos)
            ax.axis("off")
            ax.text(0.5, 0.5, "Validation metadata not provided\nPer-protein panels unavailable", ha="center", va="center")
            ax.set_title("%s. Per-protein panel" % label, loc="left", fontweight="bold")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_metrics_outputs(y_true, y_pred, epoch, out_dir, token_size=None, val_metadata=None,
                         protein_col="protein", subtype_col="subtype", min_group_n=1,
                         save_val_predictions=False, dpi=300):
    epoch_dir = os.path.join(out_dir, "epoch_%02d" % int(epoch))
    ensure_dir(epoch_dir)

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    cm = confusion_counts(y_true, y_pred)
    md = metrics_from_cm(cm)
    global_row = build_overall_row(md)
    global_row["epoch"] = int(epoch)
    if token_size is not None:
        global_row["token_size"] = token_size

    pd.DataFrame([global_row]).to_csv(os.path.join(epoch_dir, "metrics_global.csv"), index=False)
    host_df = host_level_table(md)
    host_df.insert(0, "epoch", int(epoch))
    if token_size is not None:
        host_df.insert(1, "token_size", token_size)
    host_df.to_csv(os.path.join(epoch_dir, "TableS3_host_level_metrics.csv"), index=False)

    count_df, norm_df, long_df = confusion_to_frames(cm)
    count_df.to_csv(os.path.join(epoch_dir, "confusion_matrix_global_counts.csv"))
    norm_df.to_csv(os.path.join(epoch_dir, "confusion_matrix_global_row_normalized.csv"))
    long_df.to_csv(os.path.join(epoch_dir, "confusion_matrix_global_long.csv"), index=False)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pred_df["true_host"] = pred_df["y_true"].map(ID_TO_HOST)
    pred_df["pred_host"] = pred_df["y_pred"].map(ID_TO_HOST)

    if val_metadata is not None:
        meta = val_metadata.reset_index(drop=True).copy()
        pred_df = pd.concat([pred_df, meta], axis=1)
        if protein_col in pred_df.columns:
            pred_df["protein"] = pred_df[protein_col].astype(str).replace({"nan": "NA", "NaN": "NA"})
        if subtype_col in pred_df.columns:
            pred_df["protein_subtype"] = pred_df.apply(lambda r: infer_group_label(r, protein_col, subtype_col), axis=1)

        if "protein" in pred_df.columns:
            by_protein = grouped_metrics(pred_df, "protein", min_group_n)
            by_protein.insert(0, "epoch", int(epoch))
            if token_size is not None:
                by_protein.insert(1, "token_size", token_size)
            by_protein.to_csv(os.path.join(epoch_dir, "TableS3_by_protein_metrics.csv"), index=False)

            cm_long_rows = []
            for protein, sub in pred_df.groupby("protein"):
                pcm = confusion_counts(sub["y_true"].values, sub["y_pred"].values)
                _, _, pl = confusion_to_frames(pcm)
                pl.insert(0, "protein", protein)
                cm_long_rows.append(pl)
            if cm_long_rows:
                pd.concat(cm_long_rows, ignore_index=True).to_csv(os.path.join(epoch_dir, "confusion_matrix_by_protein_long.csv"), index=False)

        if "protein_subtype" in pred_df.columns:
            by_subtype = grouped_metrics(pred_df, "protein_subtype", min_group_n)
            by_subtype.insert(0, "epoch", int(epoch))
            if token_size is not None:
                by_subtype.insert(1, "token_size", token_size)
            by_subtype.to_csv(os.path.join(epoch_dir, "TableS3_by_protein_subtype_metrics.csv"), index=False)

    if save_val_predictions:
        pred_df.to_csv(os.path.join(epoch_dir, "validation_predictions.csv"), index=False)

    make_figure_s3(y_true, y_pred, host_df, pred_df,
                   os.path.join(epoch_dir, "FigureS3_host_level_performance.png"),
                   os.path.join(epoch_dir, "FigureS3_host_level_performance.pdf"),
                   dpi=dpi)
    return global_row


# =====================================================================
# Parser: original arguments + metric-output arguments.
# =====================================================================
parser = argparse.ArgumentParser('Training a Bi-LSTM model')
parser.add_argument('--data_train', type=str, required=True, help='data for training the model')
parser.add_argument('--label_train', type=str, required=True, help='label of training data')
parser.add_argument('--data_val', type=str, required=True, help='data for validate the model')
parser.add_argument('--label_val', type=str, required=True, help='label of validation data')
parser.add_argument('--length_train', type=str, required=True, help='length file of training data')
parser.add_argument('--length_val', type=str, required=True, help='length file of validation data')
parser.add_argument('--epoch', type=int, default=30, help='epoch for training the model')
parser.add_argument('--keepprob', type=float, default=0.8, help='keep probability in dropout')
parser.add_argument('--num_class', type=int, default=3, help='amount of label classes')
parser.add_argument('--hidden_size', type=str, default='256,128', help='hidden layer size of Bi-LSTM model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate when training the model')
parser.add_argument('--max_length', type=int, default=264, help='maximum length of sequence')
parser.add_argument('--writer_path', type=str, required=True, help='path to writer training log')
parser.add_argument('--model_path', type=str, required=True, help='path to save trained model')
parser.add_argument('--batchsize', type=int, default=256, help='training batchsize')

# New metric arguments. They do not change the training computation graph.
parser.add_argument('--metrics_out_dir', type=str, default=None, help='directory for metric tables and FigureS3 outputs')
parser.add_argument('--token_size', type=str, default=None, help='token size label written into metric tables')
parser.add_argument('--val_metadata_csv', type=str, default=None, help='optional validation metadata CSV aligned to data_val rows')
parser.add_argument('--protein_col', type=str, default='protein', help='protein column in validation metadata')
parser.add_argument('--subtype_col', type=str, default='subtype', help='subtype column in validation metadata')
parser.add_argument('--min_group_n', type=int, default=1, help='minimum rows for per-protein/per-subtype metric rows')
parser.add_argument('--eval_every', type=int, default=1, help='save extended metrics every N epochs; micro-F1 is still computed every epoch')
parser.add_argument('--save_val_predictions', action='store_true', help='save validation prediction table for each evaluated epoch')
parser.add_argument('--metrics_dpi', type=int, default=300, help='DPI for FigureS3 output')
args = parser.parse_args(sys.argv[1:])

# =====================================================================
# Data loading. This is allowed to differ; training graph/update logic is not changed.
# =====================================================================
data_train_raw = ChunkedMemmapSequence(args.data_train)
label_train, train_keep_idx, train_label_diag = read_label_file(args.label_train, dataset_name="train", skip_unknown=True)

data_val_raw = ChunkedMemmapSequence(args.data_val)
label_val, val_keep_idx, val_label_diag = read_label_file(args.label_val, dataset_name="validation", skip_unknown=True)

sequence_length_raw = read_length_file(args.length_train)
length_val_raw = read_length_file(args.length_val)

# Check row correspondence before filtering unknown-host rows.
if train_label_diag["n_raw"] != data_train_raw.shape[0]:
    raise ValueError("training title/label rows do not match training data rows before filtering: title_rows=%d, data=%d" % (train_label_diag["n_raw"], data_train_raw.shape[0]))
if len(sequence_length_raw) != data_train_raw.shape[0]:
    raise ValueError("training length rows do not match training data rows before filtering: lengths=%d, data=%d" % (len(sequence_length_raw), data_train_raw.shape[0]))
if val_label_diag["n_raw"] != data_val_raw.shape[0]:
    raise ValueError("validation title/label rows do not match validation data rows before filtering: title_rows=%d, data=%d" % (val_label_diag["n_raw"], data_val_raw.shape[0]))
if len(length_val_raw) != data_val_raw.shape[0]:
    raise ValueError("validation length rows do not match validation data rows before filtering: lengths=%d, data=%d" % (len(length_val_raw), data_val_raw.shape[0]))

# Data-reading filter only: skip rows whose title/label has host=Unknown/Other.
data_train = IndexedSequence(data_train_raw, train_keep_idx, name="train_filtered") if len(train_keep_idx) != data_train_raw.shape[0] else data_train_raw
data_val = IndexedSequence(data_val_raw, val_keep_idx, name="val_filtered") if len(val_keep_idx) != data_val_raw.shape[0] else data_val_raw
sequence_length = [sequence_length_raw[int(i)] for i in train_keep_idx]
length_val = [length_val_raw[int(i)] for i in val_keep_idx]

if train_label_diag["n_skipped_excluded"] > 0:
    print("[Info] Skipped %d training rows with host=Unknown/Other. Filtered training rows=%d" % (train_label_diag["n_skipped_excluded"], data_train.shape[0]))
if val_label_diag["n_skipped_excluded"] > 0:
    print("[Info] Skipped %d validation rows with host=Unknown/Other. Filtered validation rows=%d" % (val_label_diag["n_skipped_excluded"], data_val.shape[0]))

if len(label_train) != data_train.shape[0]:
    raise ValueError("filtered training label size does not match filtered training data rows: labels=%d, data=%d" % (len(label_train), data_train.shape[0]))
if len(sequence_length) != data_train.shape[0]:
    raise ValueError("filtered training length size does not match filtered training data rows: lengths=%d, data=%d" % (len(sequence_length), data_train.shape[0]))
if len(label_val) != data_val.shape[0]:
    raise ValueError("filtered validation label size does not match filtered validation data rows: labels=%d, data=%d" % (len(label_val), data_val.shape[0]))
if len(length_val) != data_val.shape[0]:
    raise ValueError("filtered validation length size does not match filtered validation data rows: lengths=%d, data=%d" % (len(length_val), data_val.shape[0]))

num_class = args.num_class
hidden_size = [int(args.hidden_size.split(',')[0]), int(args.hidden_size.split(',')[1])]
lr = args.lr
max_length = args.max_length
writer_path = args.writer_path

# Non-invasive checks only. Do not change max_length or sequence lengths.
if data_train.shape[1] != max_length or data_val.shape[1] != max_length:
    print("[Warning] embedding second dimension and --max_length differ: train_dim=%d, val_dim=%d, max_length=%d" % (data_train.shape[1], data_val.shape[1], max_length))
    print("[Warning] conservative version keeps max_length=args.max_length exactly as the original script. If TensorFlow shape error occurs, rerun with --max_length equal to embedding.shape[1].")
if max(sequence_length) > max_length or max(length_val) > max_length:
    print("[Warning] some sequence lengths exceed --max_length. Conservative version does NOT clip lengths, matching the original script.")

metrics_out_dir = args.metrics_out_dir
if metrics_out_dir is None:
    metrics_out_dir = os.path.join(os.path.dirname(args.model_path), 'metrics')
ensure_dir(metrics_out_dir)
val_metadata = load_validation_metadata(args.val_metadata_csv, data_val.shape[0], args.protein_col, args.subtype_col, keep_indices=val_keep_idx, n_original=val_label_diag["n_raw"])

print('training data size', data_train.shape)
print('training label size', len(label_train))
print('validation data size', data_val.shape)
print('validation label size', len(label_val))
print('metrics output dir', metrics_out_dir)

# Save configuration. Does not affect training.
run_config = vars(args).copy()
run_config["timestamp"] = datetime.now().isoformat(timespec="seconds")
run_config["data_train_shape"] = list(data_train.shape)
run_config["data_val_shape"] = list(data_val.shape)
run_config["train_label_filter"] = train_label_diag
run_config["validation_label_filter"] = val_label_diag
with open(os.path.join(metrics_out_dir, "training_run_config.json"), "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2, ensure_ascii=False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

g = build_graph()
print('model build successful')

epoch_summary_rows = []

# =====================================================================
# Training loop: kept as close as possible to the original script.
# =====================================================================
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(args.epoch):
        with tqdm(total=data_train.shape[0]//args.batchsize) as pbar:
            for batch in get_batch(data_train, label_train, args.batchsize, sequence_length, True):
                feed_dict = {g['x']: batch[0], g['y']: batch[1], g['keep_prob']: args.keepprob,
                             g['seq_length']: batch[2]}
                _, loss, accuracy, pred = sess.run([g['train_step'], g['loss'], g['accuracy'], g['prediction']], feed_dict=feed_dict)
                pbar.update(1)

        print('epoch %s accuracy:' % str(i+1), accuracy)
        pred_val = predict_in_batches(sess, g, data_val, label_val, length_val, args.batchsize)
        f1 = metrics.f1_score(label_val, pred_val, average='micro')
        print('F1-score:', f1)

        # Added metrics only. No optimizer call or graph update happens here.
        if ((i + 1) % args.eval_every == 0) or ((i + 1) == args.epoch):
            metric_row = save_metrics_outputs(
                label_val, pred_val, i + 1, metrics_out_dir,
                token_size=args.token_size,
                val_metadata=val_metadata,
                protein_col=args.protein_col,
                subtype_col=args.subtype_col,
                min_group_n=args.min_group_n,
                save_val_predictions=args.save_val_predictions,
                dpi=args.metrics_dpi,
            )
            metric_row['train_last_batch_loss'] = float(loss)
            metric_row['train_last_batch_accuracy'] = float(accuracy)
            epoch_summary_rows.append(metric_row)
            pd.DataFrame(epoch_summary_rows).to_csv(os.path.join(metrics_out_dir, 'training_epoch_summary.csv'), index=False)
            print('Validation accuracy:', '%.6f' % metric_row['accuracy'])
            print('Validation macro-F1:', '%.6f' % metric_row['macro_f1'])
            print('Validation weighted-F1:', '%.6f' % metric_row['weighted_f1'])
            print('Validation balanced accuracy:', '%.6f' % metric_row['balanced_accuracy'])
            for host in HOST_ORDER:
                print('  %s precision=%.6f recall=%.6f f1=%.6f support=%d' % (
                    host,
                    metric_row['%s_precision' % host],
                    metric_row['%s_recall' % host],
                    metric_row['%s_f1' % host],
                    metric_row['%s_support' % host]
                ))

        g['saver'].save(sess, args.model_path, (i+1))

print('Training finished. Metrics saved under:', metrics_out_dir)
