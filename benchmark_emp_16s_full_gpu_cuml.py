#!/usr/bin/env python3
"""Full GPU EMP 16S HDC + cuML LinearSVC benchmark.

CPU work kept here: BIOM/metadata parsing, deterministic obs hypervector table
creation, train/test split, and label encoding. GPU work: sample x HDC matrix
accumulation, row normalization, LinearSVC training, and prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.svm import LinearSVC as CuLinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

from benchmark_emp_16s_empo import DEFAULT_BIOM, DEFAULT_METADATA, EMPO_COLUMNS, clean_label, load_emp_dataset, log
from benchmark_emp_16s_gpu_hdc import csr_nnz_arrays


DEFAULT_OUTDIR = Path('/home/tsl012/Multi_variant classification/emp_16s_full_gpu_cuml')

ACCUM_KERNEL = cp.RawKernel(r"""
extern "C" __global__ void emp_hdc_accumulate_kernel(
    int nnz,
    int active_dims,
    int dim,
    const int *nnz_sample_idx,
    const int *nnz_obs_idx,
    const float *nnz_values,
    const int *obs_cols,
    const float *obs_signs,
    float *matrix
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nnz * active_dims;
    if (tid >= total) return;
    int nz = tid / active_dims;
    int a = tid - nz * active_dims;
    int sample = nnz_sample_idx[nz];
    int obs = nnz_obs_idx[nz];
    int col = obs_cols[obs * active_dims + a];
    float value = nnz_values[nz] * obs_signs[obs * active_dims + a];
    atomicAdd(matrix + sample * dim + col, value);
}
""", 'emp_hdc_accumulate_kernel')


def obs_hv(sequence: str, dim: int, active_dims: int, seed: int):
    digest = hashlib.blake2b(f'{seed}|asv_sequence:{sequence}'.encode(), digest_size=16).digest()
    token_seed = int.from_bytes(digest[:8], 'little', signed=False)
    rng = np.random.default_rng(token_seed)
    cols = rng.choice(dim, size=active_dims, replace=False).astype(np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=active_dims)
    return cols, signs


def build_obs_hvs(obs_ids: list[str], dim: int, active_dims: int, seed: int):
    cols = np.empty((len(obs_ids), active_dims), dtype=np.int32)
    signs = np.empty((len(obs_ids), active_dims), dtype=np.float32)
    for i, obs in enumerate(obs_ids):
        c, s = obs_hv(obs, dim=dim, active_dims=active_dims, seed=seed)
        cols[i] = c
        signs[i] = s
    return cols.ravel(), signs.ravel()


def build_gpu_hdc_matrix(args, x_counts, obs_ids):
    timings = {}
    started = time.perf_counter()
    obs_cols, obs_signs = build_obs_hvs(obs_ids, args.hdc_dim, args.active_dims_per_sequence, args.random_state)
    timings['cpu_obs_hv_sec'] = time.perf_counter() - started

    started = time.perf_counter()
    sample_idx, obs_idx, values = csr_nnz_arrays(x_counts)
    timings['cpu_csr_nnz_array_sec'] = time.perf_counter() - started

    cp.cuda.Device(args.device).use()
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    d_sample_idx = cp.asarray(sample_idx)
    d_obs_idx = cp.asarray(obs_idx)
    d_values = cp.asarray(values)
    d_obs_cols = cp.asarray(obs_cols)
    d_obs_signs = cp.asarray(obs_signs)
    x_gpu = cp.zeros((x_counts.shape[0], args.hdc_dim), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    timings['h2d_alloc_sec'] = time.perf_counter() - started

    start_evt = cp.cuda.Event()
    stop_evt = cp.cuda.Event()
    total = int(x_counts.nnz * args.active_dims_per_sequence)
    threads = 256
    blocks = (total + threads - 1) // threads
    start_evt.record()
    ACCUM_KERNEL(
        (blocks,),
        (threads,),
        (
            np.int32(x_counts.nnz),
            np.int32(args.active_dims_per_sequence),
            np.int32(args.hdc_dim),
            d_sample_idx,
            d_obs_idx,
            d_values,
            d_obs_cols,
            d_obs_signs,
            x_gpu,
        ),
    )
    stop_evt.record()
    stop_evt.synchronize()
    timings['gpu_hdc_accum_ms'] = cp.cuda.get_elapsed_time(start_evt, stop_evt)

    started = time.perf_counter()
    norms = cp.sqrt(cp.sum(x_gpu * x_gpu, axis=1, keepdims=True))
    x_gpu = x_gpu / cp.maximum(norms, cp.float32(1e-12))
    cp.cuda.Stream.null.synchronize()
    timings['gpu_normalize_sec'] = time.perf_counter() - started
    timings['gpu_matrix_shape'] = [int(x_gpu.shape[0]), int(x_gpu.shape[1])]
    return x_gpu, timings


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        'manifest': cache_dir / 'manifest.json',
        'counts': cache_dir / 'x_counts.npz',
        'arrays': cache_dir / 'hdc_arrays.npz',
        'labels': cache_dir / 'labels_and_splits.npz',
    }


def build_label_split_arrays(metadata, args) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for label_col in EMPO_COLUMNS:
        labels = []
        valid_idx = []
        for i, value in enumerate(metadata[label_col].tolist()):
            label = clean_label(value)
            if label:
                valid_idx.append(i)
                labels.append(label)
        arrays[f'{label_col}_labels'] = np.asarray(labels, dtype=str)
        arrays[f'{label_col}_valid_idx'] = np.asarray(valid_idx, dtype=np.int32)
        if len(Counter(labels)) >= 2:
            train_rel, test_rel = split_for_label(labels, args.test_size, args.random_state)
            arrays[f'{label_col}_train_rel'] = train_rel.astype(np.int32)
            arrays[f'{label_col}_test_rel'] = test_rel.astype(np.int32)
    return arrays


def create_precache(args, cache_dir: Path):
    """Build CPU-derived inputs once; later runs load them unchanged."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    started = time.perf_counter()
    sample_ids, obs_ids, x_counts, metadata, stats = load_emp_dataset(args)
    obs_started = time.perf_counter()
    obs_cols, obs_signs = build_obs_hvs(
        obs_ids, args.hdc_dim, args.active_dims_per_sequence, args.random_state
    )
    stats['cpu_obs_hv_sec'] = time.perf_counter() - obs_started
    csr_started = time.perf_counter()
    sample_idx, obs_idx, values = csr_nnz_arrays(x_counts)
    stats['cpu_csr_nnz_array_sec'] = time.perf_counter() - csr_started
    label_arrays = build_label_split_arrays(metadata, args)

    sparse.save_npz(paths['counts'], x_counts, compressed=True)
    np.savez_compressed(
        paths['arrays'],
        sample_idx=sample_idx,
        obs_idx=obs_idx,
        values=values,
        obs_cols=obs_cols,
        obs_signs=obs_signs,
    )
    np.savez_compressed(paths['labels'], **label_arrays)
    manifest = {
        'hdc_dim': args.hdc_dim,
        'active_dims_per_sequence': args.active_dims_per_sequence,
        'random_state': args.random_state,
        'test_size': args.test_size,
        'min_sample_sum': args.min_sample_sum,
        'samples': int(x_counts.shape[0]),
        'observations': int(x_counts.shape[1]),
        'nnz': int(x_counts.nnz),
        'cache_build_sec': time.perf_counter() - started,
        'source_stats': stats,
    }
    paths['manifest'].write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return x_counts, {k: v for k, v in np.load(paths['arrays']).items()}, label_arrays, stats


def load_precache(args, cache_dir: Path):
    paths = cache_paths(cache_dir)
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(f'Precache is incomplete: {missing}')
    started = time.perf_counter()
    manifest = json.loads(paths['manifest'].read_text())
    expected = {
        'hdc_dim': args.hdc_dim,
        'active_dims_per_sequence': args.active_dims_per_sequence,
        'random_state': args.random_state,
        'test_size': args.test_size,
        'min_sample_sum': args.min_sample_sum,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f'Precache settings do not match this run: {mismatches}')
    x_counts = sparse.load_npz(paths['counts']).tocsr()
    with np.load(paths['arrays']) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(paths['labels']) as data:
        label_arrays = {key: data[key] for key in data.files}
    stats = dict(manifest['source_stats'])
    stats['precache_load_sec'] = time.perf_counter() - started
    return x_counts, arrays, label_arrays, stats


def build_gpu_hdc_from_arrays(args, x_counts, arrays):
    timings = {}
    cp.cuda.Device(args.device).use()
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    d_sample_idx = cp.asarray(arrays['sample_idx'])
    d_obs_idx = cp.asarray(arrays['obs_idx'])
    d_values = cp.asarray(arrays['values'])
    d_obs_cols = cp.asarray(arrays['obs_cols'])
    d_obs_signs = cp.asarray(arrays['obs_signs'])
    x_gpu = cp.zeros((x_counts.shape[0], args.hdc_dim), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    timings['h2d_alloc_sec'] = time.perf_counter() - started

    start_evt = cp.cuda.Event()
    stop_evt = cp.cuda.Event()
    total = int(x_counts.nnz * args.active_dims_per_sequence)
    threads = 256
    blocks = (total + threads - 1) // threads
    start_evt.record()
    ACCUM_KERNEL(
        (blocks,),
        (threads,),
        (
            np.int32(x_counts.nnz),
            np.int32(args.active_dims_per_sequence),
            np.int32(args.hdc_dim),
            d_sample_idx,
            d_obs_idx,
            d_values,
            d_obs_cols,
            d_obs_signs,
            x_gpu,
        ),
    )
    stop_evt.record()
    stop_evt.synchronize()
    timings['gpu_hdc_accum_ms'] = cp.cuda.get_elapsed_time(start_evt, stop_evt)

    started = time.perf_counter()
    norms = cp.sqrt(cp.sum(x_gpu * x_gpu, axis=1, keepdims=True))
    x_gpu = x_gpu / cp.maximum(norms, cp.float32(1e-12))
    cp.cuda.Stream.null.synchronize()
    timings['gpu_normalize_sec'] = time.perf_counter() - started
    timings['gpu_matrix_shape'] = [int(x_gpu.shape[0]), int(x_gpu.shape[1])]
    return x_gpu, timings


def split_for_label(labels, test_size, random_state):
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify)


def train_eval_cuml(x_gpu, labels, valid_idx_np, train_rel, test_rel):
    encoder = LabelEncoder()
    y_cpu = encoder.fit_transform(labels).astype(np.int32)
    y_gpu = cp.asarray(y_cpu)
    valid_gpu = cp.asarray(valid_idx_np.astype(np.int32))
    train_gpu = cp.asarray(train_rel.astype(np.int32))
    test_gpu = cp.asarray(test_rel.astype(np.int32))
    train_abs = valid_gpu[train_gpu]
    test_abs = valid_gpu[test_gpu]

    x_train = x_gpu[train_abs]
    x_test = x_gpu[test_abs]
    y_train = y_gpu[train_gpu]

    model = CuLinearSVC(C=1.0, max_iter=20000, tol=1e-6, linesearch_max_iter=200, fit_intercept=True, verbose=0)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    cp.cuda.Stream.null.synchronize()
    train_sec = time.perf_counter() - started

    started = time.perf_counter()
    pred_gpu = model.predict(x_test)
    cp.cuda.Stream.null.synchronize()
    predict_sec = time.perf_counter() - started

    y_pred = cp.asnumpy(pred_gpu).astype(np.int32)
    y_true = y_cpu[test_rel]
    return encoder, y_true, y_pred, train_sec, predict_sec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--biom', type=Path, default=DEFAULT_BIOM)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument('--min-sample-sum', type=float, default=1000.0)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--hdc-dim', type=int, default=4096)
    parser.add_argument('--active-dims-per-sequence', type=int, default=4)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--precache-dir', type=Path)
    parser.add_argument('--build-precache-only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    cp.cuda.Device(args.device).use()

    if args.precache_dir and args.build_precache_only:
        log(f'Building EMP precache at {args.precache_dir}')
        x_counts, _, _, stats = create_precache(args, args.precache_dir)
        log(f'Precache built for {x_counts.shape[0]} samples x {x_counts.shape[1]} ASVs')
        return

    label_arrays = None
    if args.precache_dir:
        log(f'Loading EMP precache from {args.precache_dir}')
        x_counts, arrays, label_arrays, stats = load_precache(args, args.precache_dir)
        x_gpu, build_stats = build_gpu_hdc_from_arrays(args, x_counts, arrays)
    else:
        log('Loading and filtering EMP BIOM + metadata')
        sample_ids, obs_ids, x_counts, metadata, stats = load_emp_dataset(args)
        x_gpu, build_stats = build_gpu_hdc_matrix(args, x_counts, obs_ids)
    log(f'Loaded filtered matrix {x_counts.shape[0]} samples x {x_counts.shape[1]} ASVs')

    log(f'Building GPU-resident HDC matrix dim={args.hdc_dim}, active_dims={args.active_dims_per_sequence}')
    log(
        'GPU-resident HDC done: '
        f'accum={build_stats["gpu_hdc_accum_ms"]:.3f}ms, '
        f'normalize={build_stats["gpu_normalize_sec"]:.3f}s'
    )

    rows = []
    for label_col in EMPO_COLUMNS:
        log(f'Running {label_col}')
        if label_arrays is not None:
            labels = label_arrays[f'{label_col}_labels'].tolist()
            valid_idx_np = label_arrays[f'{label_col}_valid_idx']
        else:
            labels = []
            valid_idx = []
            for i, value in enumerate(metadata[label_col].tolist()):
                label = clean_label(value)
                if label:
                    valid_idx.append(i)
                    labels.append(label)
            valid_idx_np = np.asarray(valid_idx, dtype=np.int32)
        counts = Counter(labels)
        if len(counts) < 2:
            rows.append({'empo_level': label_col, 'status': 'skipped_one_class', 'samples': len(labels), 'classes': len(counts)})
            continue

        if label_arrays is not None:
            train_rel = label_arrays[f'{label_col}_train_rel']
            test_rel = label_arrays[f'{label_col}_test_rel']
        else:
            train_rel, test_rel = split_for_label(labels, args.test_size, args.random_state)
        encoder, y_true, y_pred, train_sec, predict_sec = train_eval_cuml(
            x_gpu, labels, valid_idx_np, train_rel, test_rel
        )
        acc = accuracy_score(y_true, y_pred)
        level_dir = args.outdir / label_col
        level_dir.mkdir(exist_ok=True)
        report = classification_report(
            encoder.inverse_transform(y_true),
            encoder.inverse_transform(y_pred),
            zero_division=0,
        )
        (level_dir / 'full_gpu_cuml_classification_report.txt').write_text(report)
        rows.append({
            'empo_level': label_col,
            'status': 'ok',
            'samples': len(labels),
            'classes': len(counts),
            'accuracy': acc,
            'train_sec': train_sec,
            'predict_sec': predict_sec,
            'label_counts': json.dumps(dict(counts.most_common()), sort_keys=True),
        })
        log(f'{label_col} full-GPU cuML done: acc={acc:.4f}, train={train_sec:.3f}s, predict={predict_sec:.4f}s')

    stats.update(build_stats)
    stats.update({
        'device': args.device,
        'hdc_dim': args.hdc_dim,
        'active_dims_per_sequence': args.active_dims_per_sequence,
        'total_script_wall_sec': time.perf_counter() - total_started,
    })
    (args.outdir / 'full_gpu_cuml_dataset_stats.json').write_text(json.dumps(stats, indent=2, sort_keys=True))

    fieldnames = sorted({k for row in rows for k in row})
    preferred = ['empo_level', 'status', 'samples', 'classes', 'accuracy', 'train_sec', 'predict_sec', 'label_counts']
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with (args.outdir / 'full_gpu_cuml_empo_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f'Wrote {args.outdir / "full_gpu_cuml_empo_summary.csv"}')
    log(f'Wrote {args.outdir / "full_gpu_cuml_dataset_stats.json"}')


if __name__ == '__main__':
    main()
