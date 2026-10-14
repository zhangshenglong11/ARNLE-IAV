import os
import math
import sys
import argparse

import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm

tf.disable_v2_behavior()
tf.disable_eager_execution()


def fasta_token_lengths(path, n_split):
    """
    Parse FASTA by records (header starts with '^>'), supports multi-line sequences.
    Return token length per record = floor(total_aa_len / n_split).
    This aligns with: len(re.findall('.{n_split}', seq)) used originally, but without assuming 2-line FASTA.
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
                    lengths.append(cur_len // n_split)
                seen = True
                cur_len = 0
            else:
                if not seen:
                    # ignore any preamble before first header
                    continue
                # remove all whitespace inside sequence lines
                cur_len += len("".join(line.split()))

    if seen:
        lengths.append(cur_len // n_split)

    return lengths


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
parser.add_argument('--data', type=str, required=True, help='file of embedded data to predict (.npy)')
parser.add_argument('--batchsize', type=int, default=256, help='batch size in predicting')
parser.add_argument('--file', type=str, required=True, help='FASTA file (for sequence length)')
parser.add_argument('--n_split', type=int, default=6, help='amount of amino acid in a token')
parser.add_argument('--out_path', type=str, required=True, help='output path: directory or *.npy file')
parser.add_argument('--attn_name', type=str, default='attn_out.npy', help='Custom name for the attention matrix output file')
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
    attn = g.get_operation_by_name('Attention/attn').outputs[0]
    # hidden_size is not defined in this script (it likely existed in the training code).
    # For inference, collect attention outputs per batch and concatenate at the end.
    attn_chunks = []

    # Define softmax ONCE on the Tensor (avoid graph growth inside loop)
    softmax_logits = tf.nn.softmax(logits)

    # Use memmap to avoid loading huge npy into RAM
    data = np.load(args.data, mmap_mode="r")

    # y placeholder is usually not needed for inference; feed safe zeros (int32)
    label = np.zeros((data.shape[0],), dtype=np.int32)

    # Robust FASTA parsing by record boundary
    length_test = fasta_token_lengths(args.file, args.n_split)

    # ===== Quick self-check (before long inference) =====
    header_count = sum(1 for line in open(args.file, "r") if line.startswith(">"))
    print("[CHECK] data.shape:", data.shape)
    print("[CHECK] len(length_test):", len(length_test))
    print("[CHECK] header_count(^>):", header_count)
    if len(length_test) > 0:
        print("[CHECK] length_test min/max:", min(length_test), max(length_test))

    # Hard alignment checks
    if len(length_test) != data.shape[0]:
        raise RuntimeError(
            "FASTA and embedding not aligned: len(length_test)={} != data.shape[0]={}".format(
                len(length_test), data.shape[0]
            )
        )
    if len(length_test) != header_count:
        raise RuntimeError(
            "FASTA parse mismatch: len(length_test)={} != header_count={}".format(
                len(length_test), header_count
            )
        )

    first_bs = min(args.batchsize, data.shape[0])
    print("[CHECK] first batch shapes:", data[:first_bs].shape, np.asarray(length_test[:first_bs]).shape)
    # ==============================================

    prob_list = []
    pred_list = []

    total_batches = int(math.ceil(float(data.shape[0]) / float(args.batchsize)))
    with tqdm(total=total_batches) as pbar:
        for batchx, batchy, batchlen in get_batch(data, label, args.batchsize, length_test, False):
            batchlen = np.asarray(batchlen, dtype=np.int32)

            # Per-batch safety check
            if batchx.shape[0] != batchlen.shape[0]:
                raise RuntimeError(
                    "Batch mismatch: x batch = {}, seq_length = {}".format(batchx.shape[0], batchlen.shape[0])
                )

            feed_dict_test = {x: batchx, y: batchy, keep_prob: 1.0, seq_length: batchlen}
            batch_prob, batch_pred, batch_attn = sess.run(
                [softmax_logits, prediction, attn],
                feed_dict=feed_dict_test,
            )
            print("attn shape:", batch_attn.shape, "dtype:", batch_attn.dtype)
            attn_chunks.append(batch_attn)

            prob_list.append(batch_prob)
            pred_list.append(batch_pred)
            pbar.update(1)

    prob = np.concatenate(prob_list, axis=0) if prob_list else np.zeros((0, args.num_class), dtype=np.float32)
    pred_test = np.concatenate(pred_list, axis=0) if pred_list else np.zeros((0,), dtype=np.int32)
    attn_test = np.concatenate(attn_chunks, axis=0) if attn_chunks else np.zeros((0,), dtype=np.float32)

    # Output handling:
    # - if out_path endswith .npy -> save exactly there
    # - else treat as directory -> save prediction_probability.npy under it
    if args.out_path.endswith(".npy"):
        out_file = args.out_path
        out_dir = os.path.dirname(out_file)
        if out_dir and (not os.path.exists(out_dir)):
            os.makedirs(out_dir)
        np.save(out_file, prob)
        # Save attention next to the probability file.
        out_dir = os.path.dirname(out_file) or "."
        np.save(os.path.join(out_dir, args.attn_name), attn_test)
    else:
        outpath = args.out_path if args.out_path.endswith("/") else (args.out_path + "/")
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        np.save(os.path.join(outpath, "prediction_probability.npy"), prob)
        np.save(os.path.join(outpath, args.attn_name), attn_test)

    print(pred_test[pred_test == 0].shape)
