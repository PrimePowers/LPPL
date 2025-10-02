#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, subprocess
from typing import List, Tuple

def parse_chunk(spec: str) -> Tuple[float, float, str]:
    try:
        lo_s, hi_s, path = spec.split(":", 2)
        lo = float(lo_s); hi = float(hi_s)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi, path
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid --chunks spec '{spec}': {e}")

def group_checkpoints_by_chunk(checks: List[float], chunks: List[Tuple[float,float,str]]):
    groups = {}
    for x in checks:
        assigned = False
        for lo, hi, path in chunks:
            if lo <= x <= hi:
                groups.setdefault(path, []).append(x)
                assigned = True
                break
        if not assigned:
            raise SystemExit(f"Checkpoint X={x} is not covered by any chunk interval.")
    return groups

def fmtX(x: float) -> str:
    return str(int(round(x)))

def main(argv=None):
    ap = argparse.ArgumentParser(description="Run phase_stream_runner_fast_chunked.py over multiple chunk files for many checkpoints and append to a single analysis CSV.")
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="LO:HI:FILE (e.g., 1e9:3e9:angles_1e9_3e9.csv.gz 3e9:6.5e9:angles_3e9_6_5e9.csv.gz 6.5e9:1e10:angles_6_5e9_1e10.csv.gz)")
    ap.add_argument("--checkpoints", nargs="*", type=float, default=[],
                    help="List of checkpoints X. If omitted, defaults to 1e9..1e10 in steps of 0.5e9.")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.03], help="Smoothing sigmas (radians).")
    ap.add_argument("--bins", type=int, default=65536)
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--out-csv", required=True, help="Target analysis CSV (appends).")
    ap.add_argument("--runner", default="phase_stream_runner_fast_chunked.py",
                    help="Path to phase_stream_runner_fast_chunked.py")
    args = ap.parse_args(argv)

    checks = args.checkpoints
    if not checks:
        base = 1_000_000_000.0
        checks = [base * k for k in [1,1.5,2,2.5,3,3.5,4,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]]

    chunks = [parse_chunk(c) for c in args.chunks]
    groups = group_checkpoints_by_chunk(checks, chunks)

    for path, xs in groups.items():
        xs_sorted = sorted(xs)
        xs_args = [fmtX(x) for x in xs_sorted]
        angle_mappings = [f"{fmtX(x)}:{path}" for x in xs_sorted]
        cmd = [
            sys.executable, args.runner,
            "--checkpoints", *xs_args,
            "--angles-files", *angle_mappings,
            "--sigmas", *[str(s) for s in args.sigmas],
            "--bins", str(args.bins),
            "--chunk-rows", str(args.chunk_rows),
            "--train-frac", str(args.train_frac),
            "--out-csv", args.out_csv,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
