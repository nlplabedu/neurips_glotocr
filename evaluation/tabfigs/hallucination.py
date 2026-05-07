#!/usr/bin/env python3
"""
Script Analysis Tool for OCR Evaluation Results
================================================
Uses GlotScript to detect when OCR model predictions
produce text in the wrong writing system (script hallucinations,
silences, or artifacts).

Aggregation: all per-model rates are MACRO-AVERAGED over scripts,
so every script contributes equally regardless of sentence count.

Usage:
    python analyze_script_errors.py --root /path/to/res_v1.0

Folder structure expected:
    res_v1.0/
        <model>+<image_type>/
            scripts/
                Arab.csv
                Latn.csv
                ...
"""

import os
import csv
import argparse
import re
from collections import defaultdict
from GlotScript import get_script_predictor

sp = get_script_predictor(replace_punctuation=False, replace_digits=False)

# Some model outputs are enormous repeated strings; raise the limit to 10 MB.
csv.field_size_limit(10_000_000)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

IGNORE_SCRIPTS = {"Zyyy", "Zinh", "Zzzz"}   # common / inherited / unknown


def detect_script(text: str):
    """Return the dominant ISO-15924 script code of `text`, or None if empty."""
    if not text or not text.strip():
        return None
    try:
        code, conf, _ = sp(text)
        if conf == 0 or code in IGNORE_SCRIPTS:
            return None
        return code
    except Exception:
        return None


def is_empty_pred(pred: str) -> bool:
    """True when the model returned nothing meaningful."""
    return not pred or not pred.strip()


def classify_row(expected_script: str, gt: str, pred: str):
    """
    Returns one of:
        'correct'       - pred script matches expected
        'hallucination' - pred has a different recognisable script
        'silent'        - pred is empty / whitespace
        'artifact'      - pred has only Zyyy/Zinh/Zzzz or digits/punctuation
    """
    if is_empty_pred(pred):
        return "silent", None

    pred_script = detect_script(pred)

    if pred_script is None:
        return "artifact", None

    if pred_script == expected_script:
        return "correct", pred_script

    return "hallucination", pred_script


def scrub_raw_pred(raw: str) -> str:
    """
    Strip model-specific wrapper tokens (e.g. olmocr <|ref|>...<|/det|>)
    before analysis, keeping only the actual predicted text.
    """
    cleaned = re.sub(r"<\|[^|]+\|>.*?<\|/[^|]+\|>", "", raw, flags=re.DOTALL)
    return cleaned.strip()


def macro_rates(script_dict: dict) -> dict:
    """
    Compute macro-average rates over scripts.
    Each script contributes equally: rate per script = count/total for that script.
    Returns dict with keys: correct, hallucination, silent, artifact (all in %).
    """
    keys = ["correct", "hallucination", "silent", "artifact"]
    rates = {k: [] for k in keys}
    for counts in script_dict.values():
        n = max(counts.get("total", 0), 1)
        for k in keys:
            rates[k].append(100 * counts.get(k, 0) / n)
    return {k: sum(v) / len(v) for k, v in rates.items()}


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────

def analyze_folder(root: str):
    results = {}          # model -> script -> {correct, hallucination, silent, artifact, total, hall_targets}
    error_examples = defaultdict(list)

    model_folders = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and 'old' not in d
    ])

    for model_folder in model_folders:
        scripts_dir = os.path.join(root, model_folder, "scripts")
        if not os.path.isdir(scripts_dir):
            continue

        model_results = {}
        csv_files = [f for f in os.listdir(scripts_dir) if f.endswith(".csv")]

        for csv_file in sorted(csv_files):
            expected_script = csv_file.replace(".csv", "")
            filepath = os.path.join(scripts_dir, csv_file)

            counts = defaultdict(int)
            hall_targets = defaultdict(int)

            try:
                with open(filepath, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gt = (row.get("gt_norm") or "").strip()
                        pred_raw = (row.get("pred_norm") or "").strip()
                        pred = scrub_raw_pred(pred_raw)

                        kind, pred_script = classify_row(expected_script, gt, pred)
                        counts["total"] += 1
                        counts[kind] += 1

                        if kind == "hallucination" and pred_script:
                            hall_targets[pred_script] += 1

                        if kind != "correct" and counts["total"] <= 1000:
                            error_examples[model_folder].append({
                                "script_file": csv_file,
                                "expected_script": expected_script,
                                "error_type": kind,
                                "pred_script": pred_script,
                                "gt_snippet": gt[:80],
                                "pred_snippet": pred[:80],
                            })

            except Exception as e:
                print(f"  [WARN] Could not read {filepath}: {e}")
                continue

            model_results[expected_script] = dict(counts)
            model_results[expected_script]["hall_targets"] = dict(hall_targets)

        results[model_folder] = model_results

    return results, error_examples


def print_report(results, error_examples):
    print("\n" + "=" * 80)
    print("OCR MODEL SCRIPT ANALYSIS REPORT")
    print("(rates are MACRO-AVERAGED over scripts: each script contributes equally)")
    print("=" * 80)

    # ── per-model macro-averaged summary ──────────────────────────────
    model_macro = {}
    for model, scripts in results.items():
        model_macro[model] = macro_rates(scripts)

    # Sort by hallucination rate ascending
    ranked = sorted(model_macro.items(), key=lambda kv: kv[1]["hallucination"])

    print(f"\n{'Model':<45} {'Correct%':>9} {'Hall%':>7} {'Silent%':>8} {'Artifact%':>10} {'Sum':>6}")
    print("-" * 90)
    for model, mr in ranked:
        s = sum(mr.values())
        print(f"{model:<45} {mr['correct']:>8.1f}% {mr['hallucination']:>6.1f}% "
              f"{mr['silent']:>7.1f}% {mr['artifact']:>9.1f}% {s:>5.1f}%")

    # Average of model macro-averages (macro over models too)
    keys = ["correct", "hallucination", "silent", "artifact"]
    avg = {k: sum(mr[k] for mr in model_macro.values()) / len(model_macro) for k in keys}
    print("-" * 90)
    print(f"{'Average (macro over scripts x models)':<45} {avg['correct']:>8.1f}% "
          f"{avg['hallucination']:>6.1f}% {avg['silent']:>7.1f}% "
          f"{avg['artifact']:>9.1f}% {sum(avg.values()):>5.1f}%")

    # ── per-model, per-script breakdown ───────────────────────────────
    print("\n" + "=" * 80)
    print("BREAKDOWN BY MODEL x EXPECTED SCRIPT  (per-script rates)")
    print("=" * 80)
    for model, scripts in sorted(results.items()):
        mr = model_macro[model]
        print(f"\n  Model: {model}")
        print(f"  Macro avg -> correct={mr['correct']:.1f}%  hall={mr['hallucination']:.1f}%  "
              f"silent={mr['silent']:.1f}%  artifact={mr['artifact']:.1f}%")
        print(f"  {'Script':<10} {'Total':>6} {'Correct%':>9} {'Hall%':>6} {'Silent%':>8} {'Artifact%':>10} | Top hallucinated scripts")
        print("  " + "-" * 95)
        for scr, counts in sorted(scripts.items()):
            n = max(counts.get("total", 0), 1)
            top_hall = sorted(counts.get("hall_targets", {}).items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{s}({c})" for s, c in top_hall) if top_hall else "-"
            print(f"  {scr:<10} {counts.get('total', 0):>6} "
                  f"{100*counts.get('correct',0)/n:>8.1f}% "
                  f"{100*counts.get('hallucination',0)/n:>5.1f}% "
                  f"{100*counts.get('silent',0)/n:>7.1f}% "
                  f"{100*counts.get('artifact',0)/n:>9.1f}% | {top_str}")

    # ── error examples ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAMPLE ERROR EXAMPLES (up to 5 per model)")
    print("=" * 80)
    for model, examples in sorted(error_examples.items()):
        shown = set()
        count = 0
        print(f"\n  -- {model} --")
        for ex in examples:
            key = (ex["error_type"], ex["expected_script"], ex.get("pred_script"))
            if key in shown:
                continue
            shown.add(key)
            print(f"    [{ex['error_type'].upper()}] "
                  f"Expected: {ex['expected_script']} | "
                  f"Got script: {ex['pred_script'] or '-'}")
            print(f"      GT:   {ex['gt_snippet']}")
            print(f"      Pred: {ex['pred_snippet']}")
            count += 1
            if count >= 5:
                break

    # ── hallucination leaderboard ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("HALLUCINATION RANKING  (macro-avg over scripts, ascending)")
    print("=" * 80)
    for i, (model, mr) in enumerate(ranked, 1):
        bar = "=" * int(mr['hallucination'] / 2)
        print(f"  {i:>2}. {model:<45} {mr['hallucination']:5.1f}%  {bar}")

    print("\n" + "=" * 80)
    print("SILENCE RANKING  (macro-avg over scripts, descending)")
    print("=" * 80)
    silent_ranked = sorted(model_macro.items(), key=lambda kv: kv[1]["silent"], reverse=True)
    for i, (model, mr) in enumerate(silent_ranked, 1):
        bar = "=" * int(mr['silent'] / 2)
        print(f"  {i:>2}. {model:<45} {mr['silent']:5.1f}%  {bar}")

    print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze OCR script errors with GlotScript")
    parser.add_argument(
        "--root", required=True,
        help="Path to the root folder containing model sub-folders (e.g. res_v1.0)"
    )
    parser.add_argument(
        "--csv-out", default=None,
        help="Optional: write a flat CSV summary to this path"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: '{args.root}' is not a directory.")
        return

    print(f"Scanning: {args.root}")
    print("This may take a few minutes for large datasets...\n")

    results, error_examples = analyze_folder(args.root)
    print_report(results, error_examples)

    if args.csv_out:
        rows = []
        for model, scripts in results.items():
            for scr, counts in scripts.items():
                n = max(counts.get("total", 0), 1)
                rows.append({
                    "model": model,
                    "expected_script": scr,
                    "total": counts.get("total", 0),
                    "correct": counts.get("correct", 0),
                    "hallucination": counts.get("hallucination", 0),
                    "silent": counts.get("silent", 0),
                    "artifact": counts.get("artifact", 0),
                    "hall_rate_pct": round(100 * counts.get("hallucination", 0) / n, 2),
                    "silent_rate_pct": round(100 * counts.get("silent", 0) / n, 2),
                    "top_hall_scripts": "; ".join(
                        f"{s}:{c}" for s, c in
                        sorted(counts.get("hall_targets", {}).items(), key=lambda x: -x[1])[:5]
                    ),
                })
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to: {args.csv_out}")


if __name__ == "__main__":
    main()