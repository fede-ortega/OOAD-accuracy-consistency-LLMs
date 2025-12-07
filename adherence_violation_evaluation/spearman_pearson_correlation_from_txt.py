"""
Reads two TXT files that each contain either:
  a JSON array like: [{"explanation":"...", "adherence_score":1, "violation_severity":4}, ...]
  OR JSON-lines (one object per line)

Aligns entries by index, extracts a metric (default: adherence_score),
computes Spearman (rank) and Pearson (linear) correlations,
and saves a 5×5 agreement matrix (Likert 1–5).

Axis mapping:
  X = Human Annotations  (file_a)
  Y = LLM's Annotations  (file_b)
"""

import argparse
import json
import sys
from typing import Any, List, Tuple
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

VALID_METRICS = {"adherence_score", "violation_severity"}


def _load_list(path: str) -> List[dict]:
    """
    Accepts .txt files that contain either a JSON array or JSON-lines.
    Returns a list of dicts.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        sys.exit(f"ERROR: Could not read {path}: {e}")

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass

    items: List[dict] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            # ignore non-JSON lines
            continue

    if not items:
        sys.exit(f"ERROR: {path} did not contain a JSON array or JSON-lines of objects.")
    return items

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
    return [_coerce_num(obj.get(metric)) for obj in items]

def _paired(values_a: List[float], values_b: List[float]) -> Tuple[List[float], List[float], int]:
    n = min(len(values_a), len(values_b))
    xs, ys = [], []
    dropped = 0
    for i in range(n):
        a, b = values_a[i], values_b[i]
        if a is None or b is None:
            dropped += 1
            continue
        xs.append(a)  # Human annotations
        ys.append(b)  # LLM's annotations
    return xs, ys, dropped


def _corrs(xs: List[float], ys: List[float]):
    if not _HAVE_SCIPY:
        sys.exit("ERROR: scipy is not installed. Please run: pip install scipy")
    if len(xs) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r_s, p_s = spearmanr(xs, ys)
    r_p, p_p = pearsonr(xs, ys)
    return float(r_s), float(p_s), float(r_p), float(p_p)

# Plot: 5×5 agreement matrix

def _bins_from_integers(lo: float, hi: float):
    """
    Build edges centered on integers, e.g., 0.5..5.5 -> bins for 1..5.
    """
    start = math.floor(lo) - 0.5
    end = math.ceil(hi) + 0.5
    if end <= start:
        end = start + 1.0
    return np.arange(start, end + 1.0, 1.0)

def _plot_matrix5(xs: List[float], ys: List[float], out_fig: str, r_s: float, r_p: float):
    if xs and ys:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
    else:
        lo, hi = 1.0, 5.0

    lo = min(lo, 1.0)
    hi = max(hi, 5.0)
    xbins = _bins_from_integers(1.0, 5.0)
    ybins = _bins_from_integers(1.0, 5.0)

    xs_np = np.clip(np.asarray(xs, dtype=float), 1.0, 5.0)
    ys_np = np.clip(np.asarray(ys, dtype=float), 1.0, 5.0)

    M, _, _ = np.histogram2d(xs_np, ys_np, bins=[xbins, ybins])

    plt.figure(figsize=(6.2, 5.8))

    im = plt.imshow(
        M.T,
        origin="lower",
        extent=[0.5, 5.5, 0.5, 5.5],
        aspect="equal",
        interpolation="nearest"
    )

    for i in range(5):
        for j in range(5):
            c = int(M[i, j])
            if c:
                plt.text(i + 1, j + 1, str(c), ha="center", va="center")

    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks([1, 2, 3, 4, 5])
    plt.xlabel("Human Annotations")     # X (file_a)
    plt.ylabel("LLM's Annotations")     # Y (file_b)
    plt.title(f"Agreement Matrix (1–5)\nSpearman={r_s:.2f}   Pearson={r_p:.2f}")
    plt.colorbar(im, label="count")

    ax = plt.gca()
    ax.grid(False)                       
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=180)
    plt.close()

# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate Human (file_a) vs LLM (file_b) annotations and plot a 5×5 agreement matrix."
    )
    parser.add_argument("--file-a", required=True, help="Path to TXT with annotations by Human (JSON array or JSON-lines).")
    parser.add_argument("--file-b", required=True, help="Path to TXT with annotations by LLM (JSON array or JSON-lines).")
    parser.add_argument("--out-fig", default="agreement_matrix.png", help="Output PNG path.")
    parser.add_argument("--metric", default="adherence_score", choices=sorted(VALID_METRICS),
                        help="Metric to correlate (keys inside each JSON object).")
    parser.add_argument("--pretty", action="store_true", help="Print a detailed summary to stdout.")
    args = parser.parse_args()

    items_a = _load_list(args.file_a)
    items_b = _load_list(args.file_b)

    vals_a = _extract_metric(items_a, args.metric)
    vals_b = _extract_metric(items_b, args.metric)
    xs, ys, dropped = _paired(vals_a, vals_b)

    r_s, p_s, r_p, p_p = _corrs(xs, ys)
    _plot_matrix5(xs, ys, args.out_fig, r_s, r_p)

    if args.pretty:
        print("=== Correlation summary ===")
        print(f"Metric: {args.metric}")
        print("X = Human Annotations (file_a)")
        print("Y = LLM's Annotations (file_b)")
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
