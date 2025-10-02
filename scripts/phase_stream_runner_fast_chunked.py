#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_stream_runner_fast_chunked.py

Chunked, memory-safe analyzer for huge angle files (φ = log p mod 2π).
- Streams angles from CSV/CSV.GZ (one column; header 'angle' optional) or NPY.
- Deterministic Train/Test split by running index (no prior n needed).
- Builds circular histograms incrementally for TRAIN and TEST.
- Applies circular Gaussian KDE in Fourier domain: factor exp(-0.5 * (m*sigma)^2) to harmonic m.
- Reports:
    * Peak angle (deg) for TRAIN (and optionally TEST)
    * TEST resultant vector: |V|, θ* (deg)
    * Counts (n_total, n_train, n_test)
    * Optional z_like as (kde - mean)/std across bins (SNR-artig; kein echter Null-Z-Score)

Usage
-----
# Minimal: 1 Checkpoint, 1 Datei, eine Sigma
python phase_stream_runner_fast_chunked.py \
  --checkpoints 100000000000 \
  --angles-files 100000000000:angles_9e10_1e11.csv.gz \
  --sigmas 0.03 \
  --out-csv analysis_9e10_1e11.csv

# Mehrere Sigmas und andere Bins, größere Chunks
python phase_stream_runner_fast_chunked.py \
  --checkpoints 100000000000 \
  --angles-files 100000000000:angles_9e10_1e11.csv.gz \
  --sigmas 0.03 0.05 0.10 \
  --bins 65536 \
  --chunk-rows 1000000 \
  --train-frac 0.6 \
  --out-csv analysis_9e10_1e11_multi_sigma.csv
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

TAU = 2.0 * math.pi


# --------------------------- Circular KDE (Fourier) ---------------------------

def circular_kde_from_hist(hist: np.ndarray, sigma: float) -> np.ndarray:
    """
    Circular KDE via FFT.
    Domain length = 2π, bins = len(hist). Harmonic m gets multiplier exp(-0.5*(m*sigma)^2).
    sigma is in radians (same as your prior σ).
    """
    N = hist.size
    spec = np.fft.rfft(hist.astype(np.float64))
    m = np.arange(spec.size, dtype=np.float64)  # 0..N//2
    spec *= np.exp(-0.5 * (sigma * m) ** 2)
    kde = np.fft.irfft(spec, n=N)
    # keep same total mass as hist (irfft preserves DC)
    return kde


def peak_deg_from_kde(kde: np.ndarray) -> float:
    idx = int(np.argmax(kde))
    return 360.0 * idx / kde.size


def z_like_from_kde(kde: np.ndarray) -> float:
    """
    SNR-artige Größe = (max - mean)/std über Bins.
    Achtung: keine echte Z-Statistik unter Null; dient nur als vergleichbare Skala.
    """
    mu = float(np.mean(kde))
    sd = float(np.std(kde))
    if sd <= 0:
        return float('nan')
    return (float(np.max(kde)) - mu) / sd


# --------------------------- Streaming histogram build -----------------------

def stream_histograms_csv(
    path: Path,
    bins: int,
    chunk_rows: int,
    train_frac: float,
    csv_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, int, int, int, complex]:
    """
    Stream angles from CSV/CSV.GZ and build TRAIN/TEST histograms.
    - angles in radians in [0, 2π)
    - Deterministic split by running index (0..n-1):
        train_mask = (idx % denom) < numer, where numer/denom ≈ train_frac.
    Returns:
      hist_train, hist_test, n_total, n_train, n_test, V_test (complex)
    """
    edges = np.linspace(0.0, TAU, bins + 1)
    h_train = np.zeros(bins, dtype=np.float64)
    h_test = np.zeros(bins, dtype=np.float64)
    n_total = n_train = n_test = 0
    Vt = 0.0 + 0.0j

    # Build rational split with small denominator for determinism
    denom = 10_000
    numer = int(round(train_frac * denom))

    reader = pd.read_csv(
        path,
        compression='infer',
        chunksize=chunk_rows,
        header=0 if csv_col else None
    )

    for chunk_idx, chunk in enumerate(reader):
        if csv_col:
            angles = chunk[csv_col].to_numpy(dtype=np.float64, copy=False)
        else:
            # one-column CSV without header
            col0 = chunk.columns[0]
            angles = chunk[col0].to_numpy(dtype=np.float64, copy=False)

        # sanitize: map into [0, 2π)
        angles = np.mod(angles, TAU)

        k = angles.size
        idx0 = n_total
        idx = np.arange(idx0, idx0 + k, dtype=np.int64)
        train_mask = (idx % denom) < numer
        test_mask = ~train_mask

        # histograms
        if train_mask.any():
            h_train += np.histogram(angles[train_mask], bins=edges)[0].astype(np.float64)
        if test_mask.any():
            a_test = angles[test_mask]
            h_test += np.histogram(a_test, bins=edges)[0].astype(np.float64)
            # accumulate resultant for TEST
            Vt += np.exp(1j * a_test).sum(dtype=np.complex128)

        n_total += k
        n_train += int(train_mask.sum())
        n_test += int(test_mask.sum())

    return h_train, h_test, n_total, n_train, n_test, Vt


def stream_histograms_npy(
    path: Path,
    bins: int,
    train_frac: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, complex]:
    """
    NPY path variant (loads whole file at once; only use for moderate sizes).
    Provided for completeness.
    """
    angles = np.load(path).astype(np.float64, copy=False)
    edges = np.linspace(0.0, TAU, bins + 1)
    h_train = np.zeros(bins, dtype=np.float64)
    h_test = np.zeros(bins, dtype=np.float64)

    angles = np.mod(angles, TAU)

    n_total = angles.size
    denom = 10_000
    numer = int(round(train_frac * denom))
    idx = np.arange(n_total, dtype=np.int64)
    train_mask = (idx % denom) < numer
    test_mask = ~train_mask

    if train_mask.any():
        h_train += np.histogram(angles[train_mask], bins=edges)[0].astype(np.float64)
    if test_mask.any():
        a_test = angles[test_mask]
        h_test += np.histogram(a_test, bins=edges)[0].astype(np.float64)
        Vt = np.exp(1j * a_test).sum(dtype=np.complex128)
    else:
        Vt = 0.0 + 0.0j

    return h_train, h_test, n_total, int(train_mask.sum()), int(test_mask.sum()), Vt


# --------------------------- Main runner -------------------------------------

def parse_angles_mapping(pairs: Iterable[str]) -> Dict[int, Path]:
    """
    Parse mapping like: 100000000000:angles_9e10_1e11.csv.gz
    Returns dict { X: Path }
    """
    out: Dict[int, Path] = {}
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"Bad --angles-files entry: {p}")
        x_str, path_str = p.split(":", 1)
        X = int(float(x_str))
        out[X] = Path(path_str)
    return out


def detect_csv_angle_column(path: Path) -> Optional[str]:
    """
    Detect if CSV has a header with 'angle' or a single column without header.
    Returns column name if present, else None.
    """
    # Read a tiny sample
    try:
        sample = pd.read_csv(path, compression='infer', nrows=5)
    except Exception:
        return None
    cols = list(sample.columns)
    if len(cols) == 1:
        col = cols[0]
        # If the first row parses to a float, treat as 1-col with header or without
        try:
            float(sample.iloc[0, 0])
            # If header name looks like 'angle' use it, otherwise assume single-column numeric (headered)
            return col
        except Exception:
            return None
    # Prefer column literally named 'angle'
    if "angle" in cols:
        return "angle"
    # Else if first column looks numeric, use it
    try:
        float(sample.iloc[0, 0])
        return cols[0]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Chunked phase analyzer for huge prime angle files.")
    ap.add_argument("--checkpoints", type=float, nargs="+", required=True,
                    help="X checkpoints (e.g., 1e10 1e11)")
    ap.add_argument("--angles-files", type=str, nargs="+", required=True,
                    help="Mappings X:path (e.g., 10000000000:angles_5e9_1e10.csv.gz)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.03],
                    help="Smoothing σ in radians (circular Gaussian).")
    ap.add_argument("--bins", type=int, default=65536, help="Histogram bins on [0,2π).")
    ap.add_argument("--chunk-rows", type=int, default=1_000_000,
                    help="CSV chunk size (#rows) per iteration.")
    ap.add_argument("--train-frac", type=float, default=0.6, help="Train fraction [0,1].")
    ap.add_argument("--out-csv", type=str, required=True, help="Output CSV (appends if exists).")
    ap.add_argument("--compute-zlike", action="store_true",
                    help="Also compute SNR-like (max-mean)/std over KDE (not a true z under null).")
    args = ap.parse_args()

    Xs = [int(float(x)) for x in args.checkpoints]
    mapping = parse_angles_mapping(args.angles_files)
    out_path = Path(args.out_csv)
    out_exists = out_path.exists()

    # Prepare output
    import csv
    fout = open(out_path, "a", newline="")
    writer = csv.writer(fout)
    if not out_exists:
        writer.writerow([
            "X", "sigma", "bins",
            "n_total", "n_train", "n_test",
            "peak_deg_train",
            "peak_deg_test",
            "theta_star_deg_test",
            "absV_test",
            "R_test",  # |V|/n
            "z_like_train",  # optional
            "z_like_test",   # optional
            "source_file",
        ])

    for X in Xs:
        if X not in mapping:
            print(f"[warn] No angles file mapping for X={X}", file=sys.stderr)
            continue
        path = mapping[X]
        is_npy = path.suffix.lower() == ".npy"
        print(f"[processing] X={X:,} from {path}")

        if not path.exists():
            print(f"[error] Missing file: {path}", file=sys.stderr)
            continue

        if is_npy:
            h_tr, h_te, n_tot, n_tr, n_te, Vt = stream_histograms_npy(
                path, args.bins, args.train_frac
            )
        else:
            col = detect_csv_angle_column(path)
            h_tr, h_te, n_tot, n_tr, n_te, Vt = stream_histograms_csv(
                path, args.bins, args.chunk_rows, args.train_frac, col
            )

        # Guard against empty splits
        if n_tr == 0 or n_te == 0:
            print(f"[warn] Split produced empty set (n_train={n_tr}, n_test={n_te}); "
                  f"adjust --train-frac or check input.", file=sys.stderr)

        # Precompute TEST θ* and |V|
        absV = float(np.abs(Vt))
        theta_star_deg_test = (math.degrees(math.atan2(Vt.imag, Vt.real)) + 360.0) % 360.0
        R_test = absV / max(1, n_te)

        for sigma in args.sigmas:
            kde_tr = circular_kde_from_hist(h_tr, sigma)
            kde_te = circular_kde_from_hist(h_te, sigma)
            peak_deg_train = peak_deg_from_kde(kde_tr)
            peak_deg_test = peak_deg_from_kde(kde_te)

            if args.compute_zlike:
                zt = z_like_from_kde(kde_tr)
                zq = z_like_from_kde(kde_te)
            else:
                zt = ""
                zq = ""

            writer.writerow([
                X, sigma, args.bins,
                n_tot, n_tr, n_te,
                f"{peak_deg_train:.4f}",
                f"{peak_deg_test:.4f}",
                f"{theta_star_deg_test:.4f}",
                f"{absV:.0f}",
                f"{R_test:.6f}",
                zt, zq,
                str(path),
            ])
            fout.flush()

        print(f"[done] X={X:,} → n={n_tot:,} (train={n_tr:,}, test={n_te:,}); "
              f"|V|_test={absV:.0f}, θ*_test={theta_star_deg_test:.2f}°")

    fout.close()


if __name__ == "__main__":
    main()
