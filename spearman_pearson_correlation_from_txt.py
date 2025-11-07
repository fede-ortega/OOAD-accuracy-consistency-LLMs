
"""
Reads two TXT files that each contain a JSON array like:
[
  {"explanation":"...", "adherence_score":1, "violation_severity":4},
  ...
]

Aligns entries by index, extracts a metric (default: adherence_score),
computes both Spearman (rank) and Pearson (linear) correlations, and saves
a scatter plot with a y=x reference line. 
"""

import argparse
import json
import sys
from typing import Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr, pearsonr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

VALID_METRICS = {"adherence_score", "violation_severity"}

def _load_list(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.exit(f"ERROR: Could not read JSON array from {path}: {e}")
    if not isinstance(data, list):
        sys.exit(f"ERROR: {path} does not contain a JSON array at the top level.")
    return [x for x in data if isinstance(x, dict)]

def _coerce_num(v: Any):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        try:
            return float(int(v))
        except Exception:
            return None

def _extract_metric(items: List[dict], metric: str) -> List[float]:
    vals: List[float] = []
    for obj in items:
        vals.append(_coerce_num(obj.get(metric)))
    return vals

def _paired(values_a: List[float], values_b: List[float]) -> Tuple[List[float], List[float], int]:
    n = min(len(values_a), len(values_b))
    xs, ys = [], []
    dropped = 0
    for i in range(n):
        a, b = values_a[i], values_b[i]
        if a is None or b is None:
            dropped += 1
            continue
        xs.append(a)
        ys.append(b)
    return xs, ys, dropped

def _corrs(xs: List[float], ys: List[float]):
    if not _HAVE_SCIPY:
        sys.exit("ERROR: scipy is not installed. Please run: pip install scipy")
    if len(xs) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r_s, p_s = spearmanr(xs, ys)
    r_p, p_p = pearsonr(xs, ys)
    return float(r_s), float(p_s), float(r_p), float(p_p)

def _plot(xs: List[float], ys: List[float], out_fig: str, r_s: float, r_p: float):
    plt.figure(figsize=(6.4, 6.0))
    plt.scatter(xs, ys)  # default style; no explicit colors
    # Reference line y=x across the data range
    if xs and ys:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
    else:
        lo, hi = 0, 1
    pad = (hi - lo) * 0.05 if hi > lo else 1
    lo -= pad
    hi += pad
    plt.plot([lo, hi], [lo, hi], linestyle='--')  

    # Grid & labels to match the example vibe
    plt.grid(True, linestyle=':', linewidth=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Spearman correlation={r_s:.2g}\nPearson correlation={r_p:.2g}")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=180)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Spearman & Pearson correlation between two TXT outputs (JSON arrays).")
    parser.add_argument("--file-a", required=True, help="Path to first TXT/JSON array file.")
    parser.add_argument("--file-b", required=True, help="Path to second TXT/JSON array file.")
    parser.add_argument("--out-fig", default="correlation.png", help="Path to output figure file (PNG).")
    parser.add_argument("--metric", default="adherence_score", choices=sorted(VALID_METRICS), help="Metric to correlate.")
    parser.add_argument("--pretty", action="store_true", help="Print a detailed summary to stdout.")
    args = parser.parse_args()

    items_a = _load_list(args.file_a)
    items_b = _load_list(args.file_b)

    vals_a = _extract_metric(items_a, args.metric)
    vals_b = _extract_metric(items_b, args.metric)
    xs, ys, dropped = _paired(vals_a, vals_b)

    r_s, p_s, r_p, p_p = _corrs(xs, ys)
    _plot(xs, ys, args.out_fig, r_s, r_p)

    if args.pretty:
        print("=== Correlation summary ===")
        print(f"Metric: {args.metric}")
        print(f"File A count: {len(vals_a)}  File B count: {len(vals_b)}  Paired used: {len(xs)}  Dropped: {dropped}")
        if len(xs) < 2:
            print("Not enough paired observations to compute correlations (need at least 2).")
        print(f"Spearman r: {r_s:.6f}  p-value: {p_s:.6g}")
        print(f"Pearson  r: {r_p:.6f}  p-value: {p_p:.6g}")
        print(f"Figure:  {args.out_fig}")
    else:
        print(f"{r_s},{p_s},{r_p},{p_p}")

if __name__ == "__main__":
    main()
