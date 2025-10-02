#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, pandas as pd, gzip

def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge multiple CSV or CSV.GZ files with identical schema into a single CSV.GZ (header written once).")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input CSV/CSV.GZ files in order.")
    ap.add_argument("--output", required=True, help="Output CSV.GZ file.")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    header_written = False
    with gzip.open(args.output, "wt", newline="") as fout:
        for path in args.inputs:
            for chunk in pd.read_csv(path, compression="infer", chunksize=args.chunksize):
                chunk.to_csv(fout, index=False, header=not header_written)
                header_written = True
    print("Merged →", args.output)

if __name__ == "__main__":
    main()
