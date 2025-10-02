#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_custom_sane.py

Generate angles φ_p = log(p) mod 2π for primes in (X_min, X_max], using a segmented sieve.
Writes angles *streamingly* to disk to avoid high memory usage.

Key features
------------
- Segmented Sieve of Eratosthenes (base primes up to sqrt(X_max), then segment marking)
- Stream output: gzip-compressed CSV by default (one angle per line, radians)
- Supports multiple X_max checkpoints in one call
- Optional lower bound X_min (default 0): only primes in (X_min, X_max] are processed
- Chunked writing with a small in-RAM buffer
- dtype float32 optional (saves disk)

CLI examples
------------
# angles for X=1e9 to angles_1000000000.csv.gz
python generate_custom_sane.py 1000000000

# angles for multiple checkpoints (each gets own file):
python generate_custom_sane.py 100000000 1000000000

# specify a lower bound and larger segment:
python generate_custom_sane.py 1000000000 --X-min 500000000 --segment-size 2000000

# write .npy instead of CSV (only for moderate sizes; still buffered):
python generate_custom_sane.py 100000000 --format npy

Notes
-----
- For *very* large X (e.g., 1e10), pure-Python sieving will be slow.
  Prefer running several smaller ranges or using a compiled sieve if possible.
- The default output is CSV.GZ (one angle per line), which your validator can read.
- If you really need .npy for big X: the script buffers in chunks and concatenates at end
  (still much lower peak RAM than storing all angles at once, but not zero).
"""

import sys
import math
import argparse
import gzip
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

TAU = 2.0 * math.pi


# ---------- Base sieve up to n (for base primes) ----------
def sieve_upto(n: int) -> np.ndarray:
    """Return all primes <= n using a simple (dense) sieve. n expected ~ sqrt(X_max)."""
    if n < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    limit = int(n ** 0.5)
    for p in range(2, limit + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = False
    return np.nonzero(sieve)[0].astype(np.int64)


# ---------- Segmented sieve iterator ----------
def segmented_primes(lo: int, hi: int, base_primes: np.ndarray, segment_size: int) -> Iterator[int]:
    """
    Yield primes p in (lo, hi], using segmented marking with the provided base_primes.
    """
    if hi <= 2 or hi <= lo:
        return
    # Ensure odd bounds for segment stepping
    start = max(2, lo + 1)
    if start % 2 == 0:
        start += 1

    for seg_lo in range(start, hi + 1, segment_size):
        seg_hi = min(seg_lo + segment_size - 1, hi)
        length = seg_hi - seg_lo + 1
        # Only odds: represent [seg_lo .. seg_hi] but mark only odd indices
        # Map idx -> number: n = seg_lo + idx
        mark = np.ones(length, dtype=bool)

        # Strike evens quickly
        if seg_lo % 2 == 0:
            mark[0::2] = False
        else:
            mark[1::2] = False

        # Mark composites with odd base primes
        for p in base_primes:
            if p == 2:
                continue
            # first multiple of p >= seg_lo
            m = (seg_lo + p - 1) // p * p
            if m < p * p:
                m = p * p
            # m might be even; move to the next multiple with same parity as seg_lo + idx mapping
            step = p
            # strike every p
            for m0 in range(m, seg_hi + 1, step):
                idx = m0 - seg_lo
                if 0 <= idx < length:
                    mark[idx] = False

        # yield primes in this segment
        # ensure 2 if in range
        if seg_lo <= 2 <= seg_hi and 2 > lo:
            yield 2
        # odds that survived
        for i in range(length):
            n = seg_lo + i
            if mark[i] and n >= 2:
                yield n


# ---------- Angle writer helpers ----------
def write_angles_csv_gz(out_path: Path, angles_iter: Iterator[float], buffer_size: int = 200000, dtype="float32"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cast = np.float32 if dtype == "float32" else np.float64
    buf = []
    written = 0
    with gzip.open(out_path, "wt") as f:
        for phi in angles_iter:
            buf.append(cast(phi))
            if len(buf) >= buffer_size:
                # write chunk
                f.write("\n".join(f"{x:.9f}" for x in buf))
                f.write("\n")
                written += len(buf)
                buf.clear()
        if buf:
            f.write("\n".join(f"{x:.9f}" for x in buf))
            f.write("\n")
            written += len(buf)
    return written


def write_angles_npy(out_path: Path, angles_iter: Iterator[float], buffer_size: int = 200000, dtype="float32"):
    """
    Collect in chunks and save a single .npy at the end.
    For huge outputs this still needs disk + some RAM; prefer CSV.GZ for very large X.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cast = np.float32 if dtype == "float32" else np.float64
    chunks: List[np.ndarray] = []
    total = 0
    for phi in angles_iter:
        if len(chunks) == 0 or chunks[-1].size >= buffer_size:
            chunks.append(np.empty(0, dtype=cast))
        # append single value by extending last chunk
        chunks[-1] = np.append(chunks[-1], cast(phi))
        total += 1
    if total == 0:
        np.save(out_path, np.empty(0, dtype=cast))
        return 0
    arr = np.concatenate(chunks, axis=0)
    np.save(out_path, arr)
    return int(arr.size)


# ---------- Main angle generation ----------
def generate_angles(X_max: int, X_min: int, segment_size: int) -> Iterator[float]:
    """
    Yield φ_p = log(p) mod 2π for primes in (X_min, X_max].
    Uses a segmented sieve with base primes up to sqrt(X_max).
    """
    if X_max <= max(2, X_min):
        return iter(())

    limit = int(math.isqrt(X_max)) + 1
    base = sieve_upto(limit)
    # Stream segment by segment
    for p in segmented_primes(X_min, X_max, base, segment_size):
        yield (math.log(p) % TAU)


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate angles φ_p = log(p) mod 2π for primes ≤ X using a segmented sieve.")
    ap.add_argument("X_max", nargs="+", help="One or more X maxima (e.g. 1000000000 or 1e9).")
    ap.add_argument("--X-min", type=float, default=0.0, help="Lower exclusive bound (default: 0).")
    ap.add_argument("--segment-size", type=int, default=1_000_000, help="Segment size for the sieve (default: 1,000,000).")
    ap.add_argument("--format", choices=["csv", "npy"], default="csv", help="Output format (csv -> .csv.gz, or .npy).")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory.")
    ap.add_argument("--prefix", type=str, default="angles_", help="Output filename prefix.")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Angle dtype (float32 saves space).")
    ap.add_argument("--buffer", type=int, default=200_000, help="Write buffer (#angles per flush).")
    return ap.parse_args()


def main():
    args = parse_args()
    X_min = int(float(args["X_min"])) if isinstance(args, dict) else int(float(args.X_min))
    outdir = Path(args.outdir)

    # Normalize X list
    Xs: List[int] = []
    for x in (args.X_max if not isinstance(args, dict) else args["X_max"]):
        Xs.append(int(float(x)))
    Xs = sorted(set(Xs))

    for X in Xs:
        out_path = outdir / f"{args.prefix}{X}"
        if args.format == "csv":
            out_path = out_path.with_suffix(".csv.gz")
        else:
            out_path = out_path.with_suffix(".npy")

        print(f"[info] Generating angles for X={X:,} (X_min={X_min:,}), segment_size={args.segment_size}, out={out_path.name}")
        angles_iter = generate_angles(X_max=X, X_min=X_min, segment_size=args.segment_size)

        if args.format == "csv":
            n = write_angles_csv_gz(out_path, angles_iter, buffer_size=args.buffer, dtype=args.dtype)
        else:
            n = write_angles_npy(out_path, angles_iter, buffer_size=args.buffer, dtype=args.dtype)

        print(f"[ok] Wrote {n:,} angles → {out_path}")

if __name__ == "__main__":
    main()
