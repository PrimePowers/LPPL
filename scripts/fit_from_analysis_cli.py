#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def rotation_model(logX, a, b, c): return a + b*logX + c*(logX**2)
def linear_model(logX, a, b): return a + b*logX
def r2(y, yp):
    ssr = np.sum((y-yp)**2); sst = np.sum((y-np.mean(y))**2)
    return 1 - ssr/sst if sst != 0 else float("nan")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit θ(X) from one or more analysis CSVs directly (filter by sigma, choose theta column).")
    ap.add_argument("--analysis", nargs="+", required=True, help="Input analysis CSV files.")
    ap.add_argument("--sigma", type=float, default=0.03, help="Select rows with this sigma.")
    ap.add_argument("--theta-col", default="theta_star_deg_test", help="Theta column to use.")
    ap.add_argument("--save-plot", default=None, help="PNG path for plot.")
    ap.add_argument("--save-json", default=None, help="JSON path for summary.")
    ap.add_argument("--no-unwrap", action="store_true", help="Disable +360 for values ≤ 180.")
    args = ap.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in args.analysis], ignore_index=True)
    df = df[df["sigma"].astype(float).round(8) == round(args.sigma,8)]
    df = df.dropna(subset=["X", args.theta_col]).sort_values("X")
    if df.empty:
        raise SystemExit("No rows after filtering by sigma/theta column.")

    X = df["X"].to_numpy(float)
    theta = df[args.theta_col].to_numpy(float)
    if args.no_unwrap:
        theta_corr = theta.copy()
    else:
        theta_corr = np.array([t if t > 180 else t+360 for t in theta], dtype=float)

    logX = np.log(X)
    quad = {}; lin = {}

    try:
        popt_q, pcov_q = curve_fit(rotation_model, logX, theta_corr, maxfev=100000)
        a,b,c = popt_q; perr = np.sqrt(np.diag(pcov_q))
        pred_q = rotation_model(logX, a,b,c)
        quad = dict(params=dict(a=float(a),b=float(b),c=float(c)),
                    stderr=dict(a=float(perr[0]),b=float(perr[1]),c=float(perr[2])),
                    r2=float(r2(theta_corr, pred_q)),
                    residuals=(theta_corr - pred_q).tolist())
    except Exception as e:
        quad = {"error": str(e)}

    try:
        popt_l, pcov_l = curve_fit(linear_model, logX, theta_corr, maxfev=100000)
        a,b = popt_l; perr = np.sqrt(np.diag(pcov_l))
        pred_l = linear_model(logX, a,b)
        lin = dict(params=dict(a=float(a),b=float(b)),
                   stderr=dict(a=float(perr[0]),b=float(perr[1])),
                   r2=float(r2(theta_corr, pred_l)),
                   residuals=(theta_corr - pred_l).tolist())
    except Exception as e:
        lin = {"error": str(e)}

    plt.figure(figsize=(12,6))
    # left
    plt.subplot(1,2,1)
    x_cont = np.logspace(np.log10(np.min(X)*0.95), np.log10(np.max(X)*1.05), 300)
    logx_cont = np.log(x_cont)
    plt.semilogx(X, theta_corr, "o-", label="Peaks (corrected)" if not args.no_unwrap else "Peaks (raw)")
    if "params" in quad:
        p=quad["params"]; plt.semilogx(x_cont, rotation_model(logx_cont,p["a"],p["b"],p["c"]), "-", label=f"Quad fit: a={p['a']:.1f}, b={p['b']:.2f}, c={p['c']:.3f}")
    if "params" in lin:
        p=lin["params"]; plt.semilogx(x_cont, linear_model(logx_cont,p["a"],p["b"]), "--", label=f"Lin fit: a={p['a']:.1f}, b={p['b']:.2f}")
    plt.xlabel("X"); plt.ylabel("θ (deg)"); plt.title("θ vs X (analysis)"); plt.grid(True,alpha=0.3); plt.legend()

    # right
    plt.subplot(1,2,2)
    if "residuals" in quad:
        plt.semilogx(X, quad["residuals"], "o-", label=f"Residuals (quad), R²={quad.get('r2', float('nan')):.4f}")
    if "residuals" in lin:
        plt.semilogx(X, lin["residuals"], "s--", label=f"Residuals (lin), R²={lin.get('r2', float('nan')):.4f}")
    plt.axhline(0, linestyle="--"); plt.xlabel("X"); plt.ylabel("Residual (deg)"); plt.title("Residuals"); plt.grid(True,alpha=0.3); plt.legend()
    plt.tight_layout()
    if args.save_plot: plt.savefig(args.save_plot, dpi=160, bbox_inches="tight")
    else: plt.show()
    plt.close()

    if args.save_json:
        out = dict(input=dict(analysis=args.analysis, sigma=args.sigma, theta_col=args.theta_col, unwrap=not args.no_unwrap),
                   quadratic=quad, linear=lin)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Saved JSON to:", args.save_json)

if __name__ == "__main__":
    main()
