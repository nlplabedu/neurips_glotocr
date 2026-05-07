"""
script_sample_stats.py

Reads script_metrics.json from one model (any — sample counts are the same),
computes per-tier statistics on sample counts, and prints a LaTeX-ready table.
"""

import os
import json
import numpy as np
from collections import defaultdict

RESULT_DIR = "../res_v1.0"

HIGH_SCRIPTS = {"Latn"}
MID_SCRIPTS  = {"Arab", "Cyrl", "Deva", "Hani", "Jpan", "Hang", "Grek", "Hebr", "Thai"}

THRESHOLDS = [10, 50, 100, 400]   # "fewer than N samples" counts


def tier(script):
    if script in HIGH_SCRIPTS:
        return "High"
    if script in MID_SCRIPTS:
        return "Mid"
    return "Low"


def load_first_script_metrics(result_dir):
    for d in sorted(os.listdir(result_dir)):
        path = os.path.join(result_dir, d, "script_metrics.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            print(f"Using: {d}")
            return data
    return {}


def stats(counts):
    a = np.array(counts)
    return {
        "n_scripts": len(a),
        "total":     int(a.sum()),
        "min":       int(a.min()),
        "max":       int(a.max()),
        "mean":      float(a.mean()),
        "median":    float(np.median(a)),
        "std":       float(a.std()),
        "p25":       float(np.percentile(a, 25)),
        "p75":       float(np.percentile(a, 75)),
    }


def main():
    metrics = load_first_script_metrics(RESULT_DIR)
    if not metrics:
        print("No script_metrics.json found.")
        return

    tier_counts = defaultdict(list)
    all_counts  = []

    for script, v in metrics.items():
        n = v.get("samples", 0)
        tier_counts[tier(script)].append((script, n))
        all_counts.append(n)

    print("\n── Per-script sample counts by tier ─────────────────────────────\n")

    for tier_name in ["High", "Mid", "Low", "All"]:
        if tier_name == "All":
            pairs  = [(s, n) for t in tier_counts.values() for s, n in t]
            counts = all_counts
        else:
            pairs  = tier_counts[tier_name]
            counts = [n for _, n in pairs]

        if not counts:
            continue

        st = stats(counts)

        # threshold counts
        thresh_counts = {t: sum(1 for n in counts if n < t) for t in THRESHOLDS}

        print(f"{'─'*60}")
        print(f"  Tier: {tier_name}  ({st['n_scripts']} scripts, {st['total']:,} total samples)")
        print(f"  Min: {st['min']}   Max: {st['max']:,}   Mean: {st['mean']:.1f}")
        print(f"  Median: {st['median']:.0f}   Std: {st['std']:.1f}")
        print(f"  Q1: {st['p25']:.0f}   Q3: {st['p75']:.0f}")
        for t in THRESHOLDS:
            pct = thresh_counts[t] / st['n_scripts'] * 100
            print(f"  < {t:4d} samples: {thresh_counts[t]:3d} scripts ({pct:.1f}%)")

        # top 5 and bottom 5
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        print(f"  Top-5:    {', '.join(f'{s}({n})' for s,n in pairs_sorted[:5])}")
        print(f"  Bottom-5: {', '.join(f'{s}({n})' for s,n in pairs_sorted[-5:])}")

    print(f"\n{'─'*60}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("\n── LaTeX table ──────────────────────────────────────────────────\n")

    rows = []
    for tier_name in ["High", "Mid", "Low", "All"]:
        if tier_name == "All":
            counts = all_counts
            n_scripts = len(all_counts)
            label = "\\textbf{All}"
        else:
            counts    = [n for _, n in tier_counts[tier_name]]
            n_scripts = len(counts)
            label     = tier_name
        if not counts:
            continue
        st = stats(counts)
        thresh_str = " / ".join(
            str(sum(1 for n in counts if n < t)) for t in THRESHOLDS
        )
        rows.append((label, n_scripts, st['min'], st['max'],
                     st['mean'], st['median'], st['p25'], st['p75'], thresh_str))

    thresh_header = " / ".join(f"$<${t}" for t in THRESHOLDS)

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Per-script sample count statistics by resource tier. "
          r"Q1/Q3 = first/third quartile. "
          r"Last column: number of scripts below each threshold (" + thresh_header + r").}")
    print(r"\label{tab:sample_stats}")
    print(r"\begin{tabular}{l r r r r r r r l}")
    print(r"\toprule")
    print(r"Tier & Scripts & Min & Max & Mean & Median & Q1 & Q3 & \# scripts below thresholds \\")
    print(r"\midrule")
    for label, ns, mn, mx, mean, med, q1, q3, thresh_str in rows:
        if label == "\\textbf{All}":
            print(r"\midrule")
        print(f"  {label} & {ns} & {mn} & {mx:,} & {mean:.1f} & {med:.0f} & {q1:.0f} & {q3:.0f} & {thresh_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()