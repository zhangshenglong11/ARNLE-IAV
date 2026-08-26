#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import logomaker  # type: ignore
except Exception:
    logomaker = None


def df_to_numpy_compat(x, dtype=None):
    if hasattr(x, 'to_numpy'):
        return x.to_numpy(dtype=dtype)
    if hasattr(x, 'values'):
        arr = x.values
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return arr
    return np.asarray(x, dtype=dtype)


AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY-')

HOST_ALIASES = {
    'primate': 'primates',
    'primates': 'primates',
    'human': 'primates',
    'humans': 'primates',
    'homo sapiens': 'primates',

    'artiodactyla': 'artiodactyla',
    'swine': 'artiodactyla',
    'pig': 'artiodactyla',
    'pigs': 'artiodactyla',
    'porcine': 'artiodactyla',
    'sus scrofa': 'artiodactyla',

    'anseriformes': 'aves',
    'galliformes': 'aves',
    'aves': 'aves',
    'avian': 'aves',
    'bird': 'aves',
    'birds': 'aves',
}

HOST_TO_INT = {
    'artiodactyla': 0,
    'primates': 1,
    'aves': 2,
}

HOST_TO_PROB_COL = {
    'artiodactyla': 'Prob_AR',
    'primates': 'Prob_PR',
    'aves': 'Prob_AV',
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_position_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = str(c)
        if s.isdigit():
            cols.append(s)
    return cols


def normalize_h(v: str) -> str:
    v = str(v or '').strip().upper()
    m = re.search(r'(H\d{1,2})', v)
    return m.group(1) if m else ''


def normalize_n(v: str) -> str:
    v = str(v or '').strip().upper()
    m = re.search(r'(N\d{1,2})', v)
    return m.group(1) if m else ''


def normalize_host(v: str) -> str:
    s = str(v or '').strip().lower()
    s = re.sub(r'[_\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return HOST_ALIASES.get(s, s)


def parse_pairs(spec: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in str(spec or '').split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'Invalid compare_pairs entry: {item}; expected source:target')
        src, tgt = item.split(':', 1)
        src2 = normalize_host(src)
        tgt2 = normalize_host(tgt)
        if src2 not in HOST_TO_PROB_COL:
            raise ValueError(f'Unrecognized source host: {src} -> {src2}')
        if tgt2 not in HOST_TO_PROB_COL:
            raise ValueError(f'Unrecognized target host: {tgt} -> {tgt2}')
        if src2 == tgt2:
            raise ValueError(f'Source and target hosts must differ in compare_pairs: {item}')
        out.append((src2, tgt2))
    if not out:
        raise ValueError('No host pairs were parsed.')
    return out


def compute_valid_site_counts(df: pd.DataFrame, pos_cols: List[str]) -> np.ndarray:
    vals = df[pos_cols].fillna('-').astype(str)
    vals_upper = vals.apply(lambda col: col.astype(str).str.upper())
    valid = (vals != '-') & (vals != '') & (vals_upper != 'NAN')
    return df_to_numpy_compat(valid.sum(axis=1).astype(int), dtype=int)


def clean_position_tokens(df: pd.DataFrame, pos_cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    vals = df2[pos_cols].fillna('-').astype(str)
    vals = vals.apply(lambda col: col.str.strip())
    vals = vals.replace('', '-')
    vals = vals.replace('nan', '-')
    vals = vals.replace('NaN', '-')
    vals = vals.replace('NONE', '-')
    vals = vals.replace('None', '-')
    vals = vals.applymap(lambda x: x if len(str(x)) == 1 else '-')
    df2[pos_cols] = vals
    return df2


def build_frequency_tables(source_df: pd.DataFrame,
                           target_df: pd.DataFrame,
                           pos_cols: List[str],
                           min_count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src = clean_position_tokens(source_df, pos_cols)
    tgt = clean_position_tokens(target_df, pos_cols)

    amino_total = sorted({
        aa
        for aa in pd.concat([src[pos_cols], tgt[pos_cols]], axis=0).stack().astype(str)
        if len(aa) == 1
    })
    if not amino_total:
        amino_total = AA_ORDER[:]

    src_rows: Dict[str, List[float]] = {aa: [] for aa in amino_total}
    tgt_rows: Dict[str, List[float]] = {aa: [] for aa in amino_total}

    n_src = len(src)
    n_tgt = len(tgt)

    for col in pos_cols:
        src_vals = src[col].astype(str)
        tgt_vals = tgt[col].astype(str)
        present = set(src_vals).union(set(tgt_vals))
        for aa in amino_total:
            if aa not in present:
                p_src = 0.0
                p_tgt = 0.0
            else:
                c_src = int((src_vals == aa).sum())
                c_tgt = int((tgt_vals == aa).sum())
                p_src = (c_src / n_src) if (n_src > 0 and c_src > min_count) else -1
                p_tgt = (c_tgt / n_tgt) if (n_tgt > 0 and c_tgt > min_count) else -1
            src_rows[aa].append(p_src)
            tgt_rows[aa].append(p_tgt)

    df_src = pd.DataFrame(src_rows, index=pos_cols).T
    df_tgt = pd.DataFrame(tgt_rows, index=pos_cols).T
    return df_src, df_tgt


def compute_sorted_diff(df_source: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """
    Rank sites by the largest positive target-minus-source difference while
    excluding gap or missing-state characters during ranking. This keeps site
    selection consistent with downstream logo filtering.
    """
    src = df_source.replace(-1, 0).copy()
    tgt = df_target.replace(-1, 0).copy()

    # Keep ranking consistent with the drop_cols filter used by make_pairwise_logo_plots.
    # Characters removed before plotting are also excluded from top-site ranking.
    bad_tokens = ['-', '', 'nan', 'NaN', 'NONE', 'None']
    src = src.drop(index=bad_tokens, errors='ignore')
    tgt = tgt.drop(index=bad_tokens, errors='ignore')

    diff = tgt - src
    sort_scores = diff.max(axis=0)

    out = pd.DataFrame({
        'position': sort_scores.index.astype(int),
        'sort': sort_scores.values,
    })
    out = out.sort_values(by='sort', ascending=False).reset_index(drop=True)
    return out


def _prep_logo_df(df_logo: pd.DataFrame) -> pd.DataFrame:
    df_logo = df_logo.copy()

    clean_cols = []
    for c in df_logo.columns:
        c = str(c).strip()
        if c == '' or c.lower() == 'nan' or len(c) != 1:
            c = '-'
        clean_cols.append(c)
    df_logo.columns = clean_cols

    df_logo = df_logo.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df_logo.index = np.arange(df_logo.shape[0], dtype=int)

    if df_logo.shape[1] > 0:
        keep_cols = (df_logo != 0).any(axis=0)
        if keep_cols.any():
            df_logo = df_logo.loc[:, keep_cols]
        else:
            df_logo = df_logo.iloc[:, :0]

    return df_logo


def _draw_logo_on_ax(ax,
                     df_logo: pd.DataFrame,
                     xticks: List[str],
                     signed: bool = False,
                     ymax: Optional[float] = None,
                     show_xticklabels: bool = True,
                     show_bottom_spine: bool = True,
                     show_left_ticks: bool = True) -> None:
    df_logo = _prep_logo_df(df_logo)

    xticks_use = [str(x) for x in xticks[:df_logo.shape[0]]]
    if len(xticks_use) < df_logo.shape[0]:
        xticks_use.extend([str(i) for i in range(len(xticks_use), df_logo.shape[0])])

    arr_check = df_to_numpy_compat(df_logo, dtype=float)

    if logomaker is None or df_logo.shape[0] == 0 or df_logo.shape[1] == 0 or not np.isfinite(arr_check).all():
        arr = df_to_numpy_compat(df_logo, dtype=float).T
        if arr.size == 0:
            arr = np.zeros((1, max(1, len(xticks_use))), dtype=float)

        ax.imshow(arr, aspect='auto', interpolation='nearest')
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.set_xticks(range(len(xticks_use)))
        if show_xticklabels:
            ax.set_xticklabels(xticks_use, rotation=0)
        else:
            ax.set_xticklabels([])

        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)
        if not show_bottom_spine:
            ax.spines['bottom'].set_visible(False)

        return

    logo = logomaker.Logo(
        df_logo,
        ax=ax,
        color_scheme='NajafabadiEtAl2017',
        vpad=0.1,
        width=0.8
    )
    logo.style_spines(visible=False)

    keep_spines = ['left']
    if show_bottom_spine:
        keep_spines.append('bottom')
    logo.style_spines(spines=keep_spines, visible=True)

    ax.set_xticks(np.arange(df_logo.shape[0]))
    if show_xticklabels:
        ax.set_xticklabels(xticks_use, rotation=0)
    else:
        ax.set_xticklabels([])

    ax.set_title('')
    ax.set_ylabel('')

    arr_abs_max = float(np.nanmax(np.abs(arr_check))) if arr_check.size else 1.0
    if not np.isfinite(arr_abs_max) or arr_abs_max <= 0:
        arr_abs_max = 1.0

    vmax = ymax if ymax is not None else arr_abs_max
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    if signed:
        ax.axhline(0.0, color='black', linewidth=1.0)
        ax.set_ylim(-vmax, 0)
    else:
        ax.axhline(0.0, color='black', linewidth=1.0)
        ax.set_ylim(0, vmax)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if not show_bottom_spine:
        ax.spines['bottom'].set_visible(False)

    if show_left_ticks:
        if signed:
            ax.set_yticks(np.linspace(-vmax, 0, 5))
        else:
            ax.set_yticks(np.linspace(0, vmax, 5))
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])


def _save_logo(df_logo: pd.DataFrame,
               xticks: List[str],
               out_png: Path,
               title: str,
               signed: bool = False,
               ylabel: str = '') -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(xticks) * 0.45), 4.5))
    _draw_logo_on_ax(
        ax,
        df_logo,
        xticks,
        signed=signed,
        ymax=None,
        show_xticklabels=True,
        show_bottom_spine=True,
        show_left_ticks=True
    )
    if title:
        ax.set_title(title)
    else:
        ax.set_title('')
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[Info] logo generated successfully: {out_png}')


def make_pairwise_logo_plots(df_source: pd.DataFrame,
                             df_target: pd.DataFrame,
                             sorted_diff: pd.DataFrame,
                             out_dir: Path,
                             prefix: str,
                             source_host: str,
                             target_host: str,
                             top_n: int = 20) -> None:
    safe_mkdir(out_dir)
    if sorted_diff is None or len(sorted_diff) == 0 or 'position' not in sorted_diff.columns:
        return

    top_positions = [str(x) for x in sorted_diff.head(top_n)['position'].tolist()]
    top_positions = [p for p in top_positions if p in df_source.columns and p in df_target.columns]
    if not top_positions:
        return

    src2 = df_source.replace(-1, 0)[top_positions].T.copy()
    tgt2 = df_target.replace(-1, 0)[top_positions].T.copy()

    drop_cols = []
    for c in src2.columns:
        s = str(c).strip()
        if s in ('-', '', 'nan', 'NaN', 'NONE', 'None'):
            drop_cols.append(c)
    if drop_cols:
        src2 = src2.drop(columns=drop_cols, errors='ignore')
        tgt2 = tgt2.drop(columns=drop_cols, errors='ignore')

    keep_cols = []
    for c in src2.columns:
        try:
            has_signal = (src2[c] != 0).any() or (tgt2[c] != 0).any()
        except Exception:
            has_signal = True
        if has_signal:
            keep_cols.append(c)

    if keep_cols:
        src2 = src2[keep_cols]
        tgt2 = tgt2[keep_cols]

    if src2.shape[1] == 0 or tgt2.shape[1] == 0:
        print(f'[Warn] {prefix} logo matrix is empty after gap removal; skipping logo.')
        return

    src2_neg = -src2.copy()

    tgt2.to_csv(out_dir / f'{prefix}_{target_host}_logo_matrix.csv', index=True)
    src2.to_csv(out_dir / f'{prefix}_{source_host}_logo_matrix.csv', index=True)
    src2_neg.to_csv(out_dir / f'{prefix}_{source_host}_negative_logo_matrix.csv', index=True)

    _save_logo(
        tgt2,
        top_positions,
        out_dir / f'{prefix}_{target_host}_logo_plot.png',
        '',
        signed=False,
        ylabel=''
    )

    _save_logo(
        src2_neg,
        top_positions,
        out_dir / f'{prefix}_{source_host}_logo_plot.png',
        '',
        signed=True,
        ylabel=''
    )

    dual_png = out_dir / f'{prefix}_{source_host}_to_{target_host}_dual_panel_logo.png'

    tgt_abs = df_to_numpy_compat(_prep_logo_df(tgt2), dtype=float)
    src_abs = df_to_numpy_compat(_prep_logo_df(src2_neg), dtype=float)

    ymax_tgt = float(np.nanmax(np.abs(tgt_abs))) if tgt_abs.size else 1.0
    ymax_src = float(np.nanmax(np.abs(src_abs))) if src_abs.size else 1.0
    shared_ymax = max(ymax_tgt, ymax_src)
    if not np.isfinite(shared_ymax) or shared_ymax <= 0:
        shared_ymax = 1.0

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(8, len(top_positions) * 0.45), 8),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.0}
    )

    _draw_logo_on_ax(
        axes[0],
        tgt2,
        top_positions,
        signed=False,
        ymax=shared_ymax,
        show_xticklabels=False,
        show_bottom_spine=False,
        show_left_ticks=True
    )

    _draw_logo_on_ax(
        axes[1],
        src2_neg,
        top_positions,
        signed=True,
        ymax=shared_ymax,
        show_xticklabels=True,
        show_bottom_spine=True,
        show_left_ticks=True
    )

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.subplots_adjust(hspace=0.0, top=0.98, bottom=0.08, left=0.08, right=0.995)
    plt.savefig(dual_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print(f'[Info] dual-panel logo generated successfully: {dual_png}')


def calculate_ctsi(monthly_sorted_files: List[Tuple[str, Path]]) -> pd.DataFrame:
    scores = []
    months = []
    for month, path in monthly_sorted_files:
        df = pd.read_csv(path)
        months.append(month)
        if df.empty or 'sort' not in df.columns:
            scores.append(0.0)
            continue
        threshold = df['sort'].quantile(0.95)
        top = df[df['sort'] > threshold]
        if len(top) == 0:
            scores.append(0.0)
        else:
            scores.append(float((top['sort'] ** 2).mean()))
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return pd.DataFrame(columns=['month', 'CTSI'])
    if np.allclose(arr.max(), arr.min()):
        norm = np.zeros_like(arr)
    else:
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    return pd.DataFrame({'month': months, 'CTSI': norm})


def save_ctsi_plot(df_ctsi: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df_ctsi['month'], df_ctsi['CTSI'])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def run_pairwise_subset(df_pair: pd.DataFrame,
                        pos_cols: List[str],
                        out_dir: Path,
                        prefix: str,
                        source_host: str,
                        target_host: str,
                        min_count: int) -> Optional[Path]:
    source_df = df_pair[df_pair['host_norm'] == source_host].copy().reset_index(drop=True)
    target_df = df_pair[df_pair['host_norm'] == target_host].copy().reset_index(drop=True)
    if len(source_df) == 0 or len(target_df) == 0:
        return None

    safe_mkdir(out_dir)
    df_pair.to_csv(out_dir / f'{prefix}_subset.csv', index=False)
    source_df.to_csv(out_dir / f'{prefix}_{source_host}_subset.csv', index=False)
    target_df.to_csv(out_dir / f'{prefix}_{target_host}_subset.csv', index=False)

    df_source, df_target = build_frequency_tables(source_df, target_df, pos_cols, min_count)
    src_path = out_dir / f'{prefix}_{source_host}_bayes.csv'
    tgt_path = out_dir / f'{prefix}_{target_host}_bayes.csv'
    diff_path = out_dir / f'{prefix}_{source_host}_vs_{target_host}_sorted_diff.csv'
    df_source.to_csv(src_path)
    df_target.to_csv(tgt_path)

    sorted_diff = compute_sorted_diff(df_source, df_target)
    sorted_diff.to_csv(diff_path, index=False)

    diff_matrix = df_target.replace(-1, 0) - df_source.replace(-1, 0)
    diff_matrix.to_csv(out_dir / f'{prefix}_{target_host}_minus_{source_host}_diff_matrix.csv')

    make_pairwise_logo_plots(
        df_source=df_source,
        df_target=df_target,
        sorted_diff=sorted_diff,
        out_dir=out_dir,
        prefix=prefix,
        source_host=source_host,
        target_host=target_host,
    )
    return diff_path


def feature_analysis_for_pair(sub_df: pd.DataFrame,
                              pos_cols: List[str],
                              full_attn: np.ndarray,
                              start_rows: np.ndarray,
                              out_dir: Path,
                              prefix: str,
                              source_host: str,
                              target_host: str,
                              threshold: float,
                              subtype_plot_col: str) -> None:
    from sklearn.decomposition import PCA

    safe_mkdir(out_dir)
    sub_df = sub_df.reset_index(drop=True)
    lens = compute_valid_site_counts(sub_df, pos_cols)
    if (lens <= 0).any():
        raise ValueError('At least one sequence has zero valid sites; feature analysis cannot proceed.')

    site_indices = []
    for row_id, L in zip(sub_df['row_id'].astype(int).tolist(), lens.tolist()):
        start = int(start_rows[row_id])
        site_indices.extend(range(start, start + int(L)))
    attn_sub = full_attn[np.asarray(site_indices, dtype=int)]

    labels_seq = df_to_numpy_compat(sub_df['host_norm'].map(HOST_TO_INT).fillna(-1).astype(int), dtype=int)
    if (labels_seq < 0).any():
        bad = sub_df.loc[labels_seq < 0, 'host_norm'].drop_duplicates().tolist()
        raise ValueError(f'Unrecognized values were found in host_norm: {bad}')
    labels_site = np.repeat(labels_seq, lens)

    pca = PCA(n_components=2)
    pcavec = pca.fit_transform(attn_sub)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    pcavec_seq = np.zeros((len(lens), 2), dtype=float)
    label_seq = np.zeros((len(lens),), dtype=int)
    for i, (s, L) in enumerate(zip(starts, lens)):
        pcavec_seq[i] = pcavec[s:s + L].mean(axis=0)
        label_seq[i] = labels_site[s]

    src_code = HOST_TO_INT[source_host]
    tgt_code = HOST_TO_INT[target_host]
    mask_src = label_seq == src_code
    mask_tgt = label_seq == tgt_code
    if mask_src.sum() == 0 or mask_tgt.sum() == 0:
        raise ValueError(f'Feature analysis requires samples from both {source_host} and {target_host}.')

    mean_src = np.mean(pcavec_seq[mask_src], axis=0)
    pcavec_seq = pcavec_seq - mean_src
    mean_src = np.mean(pcavec_seq[mask_src], axis=0)
    mean_tgt = np.mean(pcavec_seq[mask_tgt], axis=0)

    target_prob_col = HOST_TO_PROB_COL[target_host]
    if target_prob_col not in sub_df.columns:
        raise ValueError(f'Missing target-host probability column: {target_prob_col}')

    seq_mask_source = sub_df['host_norm'].eq(source_host).to_numpy()
    seq_mask_adapt = pd.to_numeric(sub_df[target_prob_col], errors='coerce').fillna(0.0).to_numpy() >= threshold
    seq_mask_focus = seq_mask_source & seq_mask_adapt
    if seq_mask_focus.sum() == 0:
        print(f'[Warn] {prefix} No sequences satisfy source={source_host} and {target_prob_col}>={threshold}; skipping feature plot.')
        return

    diff_matrix_path = out_dir / f'{prefix}_{target_host}_minus_{source_host}_diff_matrix.csv'
    diff_matrix = pd.read_csv(diff_matrix_path, index_col=0)

    variants = []
    focus_rows = []
    for _, row in sub_df.loc[seq_mask_focus].iterrows():
        total = 0.0
        for col in pos_cols:
            aa = str(row[col]) if col in row else '-'
            if aa in diff_matrix.index and col in diff_matrix.columns:
                total += float(diff_matrix.loc[aa, col])
        variants.append(total)
        focus_rows.append({
            'host': row.get('host', ''),
            'host_norm': row.get('host_norm', ''),
            subtype_plot_col: row.get(subtype_plot_col, ''),
            'target_prob': float(pd.to_numeric(row.get(target_prob_col, 0.0), errors='coerce')),
        })

    pcavec_focus = pcavec_seq[seq_mask_focus]
    dist_src = np.linalg.norm(pcavec_focus - mean_src, axis=1)
    dist_tgt = np.linalg.norm(pcavec_focus - mean_tgt, axis=1)

    feature_df = pd.DataFrame(focus_rows)
    feature_df['distance_from_source_centroid'] = dist_src
    feature_df['distance_from_target_centroid'] = dist_tgt
    feature_df['variant_score'] = variants
    feature_df.to_csv(out_dir / f'{prefix}_{source_host}_to_{target_host}_feature_summary.csv', index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(pcavec_seq[mask_src, 0], pcavec_seq[mask_src, 1], alpha=0.20, label=f'{source_host} background')
    plt.scatter(pcavec_seq[mask_tgt, 0], pcavec_seq[mask_tgt, 1], alpha=0.20, label=f'{target_host} background')
    sc = plt.scatter(
        pcavec_focus[:, 0],
        pcavec_focus[:, 1],
        c=feature_df['variant_score'],
        alpha=0.85,
        marker='o',
        cmap='seismic',
        label=f'{source_host} with {target_host} adaptivity',
    )
    plt.scatter([mean_src[0]], [mean_src[1]], marker='X', s=120, label=f'{source_host} centroid')
    plt.scatter([mean_tgt[0]], [mean_tgt[1]], marker='X', s=120, label=f'{target_host} centroid')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best')
    plt.colorbar(sc, label='variant_score')
    plt.title(f'{prefix} | {source_host} -> {target_host}')
    plt.tight_layout()
    plt.savefig(out_dir / f'{prefix}_{source_host}_to_{target_host}_feature_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5))
    subtype_unique = [s for s in pd.Series(feature_df[subtype_plot_col]).dropna().astype(str).unique().tolist() if s]
    if not subtype_unique:
        subtype_unique = ['ALL']
        feature_df[subtype_plot_col] = 'ALL'
    markers = ['o', '^', 'D', 'v', '*', '+', 'x', '1', 's', 'P']
    last_scatter = None
    for i, st in enumerate(subtype_unique):
        tmp = feature_df[feature_df[subtype_plot_col] == st]
        last_scatter = plt.scatter(
            tmp['distance_from_source_centroid'],
            tmp['distance_from_target_centroid'],
            c=tmp['variant_score'],
            marker=markers[i % len(markers)],
            alpha=0.70,
            cmap='seismic',
            label=st,
        )
    if last_scatter is not None:
        plt.colorbar(last_scatter, label='variant_score')
    plt.xlabel(f'Distance from {source_host} centroid')
    plt.ylabel(f'Distance from {target_host} centroid')
    plt.legend(loc='best')
    plt.title(f'{prefix} | focus: {source_host} predicted toward {target_host}')
    plt.tight_layout()
    plt.savefig(out_dir / f'{prefix}_{source_host}_to_{target_host}_feature_distance.png', dpi=300, bbox_inches='tight')
    plt.close()


def choose_subtype_settings(df: pd.DataFrame, protein: str, subtype_mode: str, subtype_col: str) -> Tuple[str, str]:
    protein_upper = str(protein or '').strip().upper()

    if subtype_col:
        if subtype_col not in df.columns:
            raise ValueError(f'Specified --subtype_col={subtype_col} is not present. Available columns: {list(df.columns)}')
        resolved_col = subtype_col
    else:
        mode = str(subtype_mode or 'auto').strip().lower()
        if mode == 'h':
            resolved_col = 'H_subtype'
        elif mode == 'n':
            resolved_col = 'N_subtype'
        elif mode == 'auto':
            if protein_upper == 'NA' and 'N_subtype' in df.columns:
                resolved_col = 'N_subtype'
            elif 'H_subtype' in df.columns:
                resolved_col = 'H_subtype'
            elif 'N_subtype' in df.columns:
                resolved_col = 'N_subtype'
            else:
                raise ValueError('No H_subtype or N_subtype column was found in auto mode.')
        else:
            raise ValueError(f'Unsupported --subtype_mode: {subtype_mode}')
        if resolved_col not in df.columns:
            raise ValueError(f'Required subtype column {resolved_col} is missing. Available columns: {list(df.columns)}')

    norm_col = f'{resolved_col}_norm'
    return resolved_col, norm_col


def normalize_subtype_series(df: pd.DataFrame, subtype_col: str, norm_col: str) -> pd.DataFrame:
    df = df.copy()
    if subtype_col == 'H_subtype':
        df[norm_col] = df[subtype_col].map(normalize_h)
    elif subtype_col == 'N_subtype':
        df[norm_col] = df[subtype_col].map(normalize_n)
    else:
        df[norm_col] = df[subtype_col].astype(str).str.strip()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Batch pairwise residue-frequency analysis by protein, subtype, and host pair, with optional feature analysis.'
    )
    ap.add_argument('--master_tables', nargs='+', required=True, help='One or more master-table CSV files')
    ap.add_argument('--out_root', required=True, help='Root output directory')
    ap.add_argument('--compare_pairs', default='aves:primates,artiodactyla:primates', help='Host pairs in source:target,source:target format')
    ap.add_argument('--min_count', type=int, default=50, help='Minimum site-count threshold; default 50')
    ap.add_argument('--run_feature', action='store_true', help='Run optional feature analysis')
    ap.add_argument('--feature_threshold', type=float, default=0.5, help='Probability threshold for treating a source sequence as target-adapted; default 0.5')
    ap.add_argument('--full_attn_npy', default='', help='Full validation-set attention/site-level feature .npy required for feature analysis')
    ap.add_argument('--skip_monthly', action='store_true', help='Skip monthly analysis')
    ap.add_argument('--subtype_mode', default='auto', choices=['auto', 'h', 'n'],
                    help='Subtype grouping mode: auto, h, or n. In auto mode, NA prefers N_subtype and other proteins prefer H_subtype.')
    ap.add_argument('--subtype_col', default='',
                    help='Manually specify the subtype column; overrides --subtype_mode, for example N_subtype.')
    args = ap.parse_args()

    pairs = parse_pairs(args.compare_pairs)
    out_root = Path(args.out_root)
    safe_mkdir(out_root)

    full_attn = None
    if args.run_feature:
        if not args.full_attn_npy:
            raise ValueError('--run_feature requires --full_attn_npy.')
        full_attn = np.load(args.full_attn_npy)

    for table_path in args.master_tables:
        df = pd.read_csv(table_path, dtype=str).fillna('')
        for c in ['Prob_AR', 'Prob_PR', 'Prob_AV']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        if 'row_id' in df.columns:
            df['row_id'] = pd.to_numeric(df['row_id'], errors='coerce').astype(int)
        else:
            df['row_id'] = np.arange(len(df), dtype=int)

        protein = df['protein'].iloc[0] if 'protein' in df.columns and len(df) else Path(table_path).stem.split('_')[0]
        pos_cols = find_position_cols(df)
        if not pos_cols:
            raise ValueError(f'{table_path}  contains no recognized numeric site columns.')

        if 'host' not in df.columns:
            raise ValueError(f'{table_path}  is missing the host column.')
        df['host_norm'] = df['host'].map(normalize_host)

        known_hosts = set(HOST_TO_PROB_COL.keys())
        unknown_hosts = sorted([
            x for x in df['host_norm'].dropna().astype(str).unique().tolist()
            if x and x not in known_hosts
        ])
        if unknown_hosts:
            print(f'[Warn] {protein} contains unmapped host labels: {unknown_hosts}')

        subtype_col_resolved, subtype_norm_col = choose_subtype_settings(
            df=df, protein=protein, subtype_mode=args.subtype_mode, subtype_col=args.subtype_col
        )
        df = normalize_subtype_series(df, subtype_col_resolved, subtype_norm_col)

        subtype_list = sorted([x for x in df[subtype_norm_col].dropna().astype(str).unique().tolist() if x])
        if not subtype_list:
            print(f'[Warn] {protein} has no recognized subtypes in column {subtype_col_resolved}; skipping.')
            continue

        print(f'[Info] {protein} using subtype column {subtype_col_resolved}; recognized subtypes: {subtype_list}')

        start_rows = None
        if args.run_feature:
            lens_all = compute_valid_site_counts(df, pos_cols)
            start_rows = np.concatenate(([0], np.cumsum(lens_all)[:-1]))
            if int(lens_all.sum()) != int(full_attn.shape[0]):
                raise ValueError(
                    f'full_attn rows {full_attn.shape[0]} do not match the total valid-site count {int(lens_all.sum())} in {table_path}.'
                )

        for subtype_value in subtype_list:
            df_sub = df[df[subtype_norm_col] == subtype_value].copy().reset_index(drop=True)
            if len(df_sub) == 0:
                continue

            for source_host, target_host in pairs:
                df_pair = df_sub[df_sub['host_norm'].isin([source_host, target_host])].copy().reset_index(drop=True)
                if len(df_pair) == 0:
                    continue

                n_src = int((df_pair['host_norm'] == source_host).sum())
                n_tgt = int((df_pair['host_norm'] == target_host).sum())
                if n_src == 0 or n_tgt == 0:
                    print(f'[Warn] {protein} {subtype_value} is missing {source_host} or {target_host} samples; skipping this pair.')
                    continue

                pair_name = f'{source_host}_to_{target_host}'
                pair_dir = out_root / protein / subtype_value / pair_name
                safe_mkdir(pair_dir)
                prefix = f'{protein}_{subtype_value}_{pair_name}'

                run_pairwise_subset(
                    df_pair=df_pair,
                    pos_cols=pos_cols,
                    out_dir=pair_dir,
                    prefix=prefix,
                    source_host=source_host,
                    target_host=target_host,
                    min_count=args.min_count,
                )

                if not args.skip_monthly and 'month' in df_pair.columns:
                    month_dir = pair_dir / 'monthly'
                    safe_mkdir(month_dir)
                    month_files: List[Tuple[str, Path]] = []
                    month_list = sorted([m for m in df_pair['month'].astype(str).unique().tolist() if m and m != 'nan'])
                    for month in month_list:
                        df_m = df_pair[df_pair['month'].astype(str) == month].copy().reset_index(drop=True)
                        if int((df_m['host_norm'] == source_host).sum()) == 0 or int((df_m['host_norm'] == target_host).sum()) == 0:
                            continue
                        month_prefix = f'{prefix}_{month}'
                        diff_path = run_pairwise_subset(
                            df_pair=df_m,
                            pos_cols=pos_cols,
                            out_dir=month_dir,
                            prefix=month_prefix,
                            source_host=source_host,
                            target_host=target_host,
                            min_count=args.min_count,
                        )
                        if diff_path is not None:
                            month_files.append((month, diff_path))
                    if month_files:
                        df_ctsi = calculate_ctsi(month_files)
                        df_ctsi.to_csv(pair_dir / f'{prefix}_ctsi_by_month.csv', index=False)
                        save_ctsi_plot(df_ctsi, pair_dir / f'{prefix}_ctsi_trend.png', f'{prefix} CTSI by month')

                if args.run_feature and len(df_pair) > 0:
                    try:
                        feature_analysis_for_pair(
                            sub_df=df_pair,
                            pos_cols=pos_cols,
                            full_attn=full_attn,
                            start_rows=start_rows,
                            out_dir=pair_dir,
                            prefix=prefix,
                            source_host=source_host,
                            target_host=target_host,
                            threshold=args.feature_threshold,
                            subtype_plot_col=subtype_col_resolved,
                        )
                    except Exception as e:
                        print(f'[Warn] feature analysis failed for {prefix}: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
