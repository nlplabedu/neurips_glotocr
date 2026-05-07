import os
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from configs import MODELS, DATASET_ROOT, RESULT_DIR, EXTRACTORS
from utils import normalize_text, cer, detect_script, save_json


# ── thresholds that drive every acc metric ────────────────────────────────────
ACC_THRESHOLDS = [0.00, 0.02, 0.05, 0.10]
MAJOR_SCRIPTS  = {"Latn", "Arab", "Cyrl", "Deva"}

# ── resource tiers ────────────────────────────────────────────────────────────
HIGH_SCRIPTS = {"Latn"}
MID_SCRIPTS  = {"Arab", "Cyrl", "Deva", "Hani", "Jpan", "Hang", "Grek", "Hebr", "Thai"}
MIN_CHAR = 0


# ── helpers ───────────────────────────────────────────────────────────────────
def _acc_keys():
    return [f"Acc@{t:.2f}" for t in ACC_THRESHOLDS]


def _make_accumulator():
    return {"samples": 0, "cer_sum": 0.0, "script_correct": 0,
            **{k: 0 for k in _acc_keys()}}


def _update(acc, c, pred_script=None, gt_script=None):
    acc["samples"]  += 1
    acc["cer_sum"]  += c
    for t, k in zip(ACC_THRESHOLDS, _acc_keys()):
        if c <= t:
            acc[k] += 1
    if pred_script is not None:
        acc["script_correct"] += int(pred_script == gt_script)


def _finalize(acc, include_script_acc=True):
    n = acc["samples"]
    result = {
        "samples":    n,
        "CER":        acc["cer_sum"] / n,
        **{k: acc[k] / n for k in _acc_keys()},
    }
    if include_script_acc:
        result["ScriptAcc"] = acc["script_correct"] / n
    return result


def _macro(results):
    keys = ["CER"] + _acc_keys() + ["ScriptAcc"]
    n = len(results)
    return {k: sum(v[k] for v in results.values()) / n for k in keys}


def _tier_avg(script_results, tier_scripts):
    """Average metrics over scripts present in both tier_scripts and script_results."""
    subset = {s: v for s, v in script_results.items() if s in tier_scripts}
    return _macro(subset) if subset else {}


# ── data loading ──────────────────────────────────────────────────────────────
def load_model_dataset(model_dir):
    print("  • Searching parquet files")
    parquet_files = glob.glob(os.path.join(model_dir, "*.parquet"))
    print(f"  • Found {len(parquet_files)} parquet files")
    dfs = [pd.read_parquet(p) for p in tqdm(parquet_files, desc="  • Loading parquet")]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  • Loaded {len(df)} rows")
    df  = df[df["text"].str.len() >= MIN_CHAR]
    print(f"  • After filtering short texts: {len(df)} rows")
    return df


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model):
    print(f"\n{'='*30}\nEvaluating model: {model}\n{'='*30}")

    df       = load_model_dataset(os.path.join(DATASET_ROOT, model))
    extractor = EXTRACTORS[model]

    script_acc   = defaultdict(_make_accumulator)
    language_acc = defaultdict(_make_accumulator)          # key: (script, lang)
    script_conf  = defaultdict(lambda: defaultdict(int))
    script_exs   = defaultdict(list)
    total_acc    = _make_accumulator()

    print("Stage 2: Running evaluation")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  • Evaluating samples"):
        gt       = normalize_text(row["text"])
        pred     = extractor(row["markdown"])
        script   = row["script"]
        language = row["language"]
        c        = cer(pred, gt)
        pred_script = detect_script(pred)

        _update(script_acc[script], c, pred_script, script)
        _update(total_acc,          c, pred_script, script)
        script_conf[script][pred_script] += 1

        if script in MAJOR_SCRIPTS:
            _update(language_acc[(script, language)], c)

        script_exs[script].append(
            {"gt_raw": row["text"], "pred_raw": row["markdown"],
             "gt_norm": gt, "pred_norm": pred, "cer": c}
        )

    print("Stage 3-4: Finalising per-script and language metrics")
    script_results   = {s: _finalize(a)              for s, a in script_acc.items()}
    language_results = defaultdict(dict)
    for (script, lang), a in language_acc.items():
        language_results[script][lang] = _finalize(a, include_script_acc=False)

    print("Stage 5-6: Computing macro / micro / tier metrics")
    low_scripts = set(script_results) - HIGH_SCRIPTS - MID_SCRIPTS
    summary = {
        "model":              model,
        "samples":            total_acc["samples"],
        "macro":              _macro(script_results),
        "micro":              _finalize(total_acc),
        "high_resource":      _tier_avg(script_results, HIGH_SCRIPTS),
        "mid_resource":       _tier_avg(script_results, MID_SCRIPTS),
        "low_resource":       _tier_avg(script_results, low_scripts),
    }

    print("Stage 7: Saving results")
    model_result_dir = os.path.join(RESULT_DIR, model)
    os.makedirs(model_result_dir, exist_ok=True)

    save_json(os.path.join(model_result_dir, "summary.json"),        summary)
    save_json(os.path.join(model_result_dir, "script_metrics.json"), script_results)
    save_json(os.path.join(model_result_dir, "script_confusion.json"), dict(script_conf))

    for script, langs in language_results.items():
        if script in MAJOR_SCRIPTS:
            save_json(
                os.path.join(model_result_dir, f"language_metrics_{script}.json"),
                langs,
            )

    print("Stage 8: Saving script debug CSVs")
    scripts_dir = os.path.join(model_result_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    for script, rows in script_exs.items():
        pd.DataFrame(rows).to_csv(
            os.path.join(scripts_dir, f"{script}.csv"), index=False
        )

    print("✔ Finished model:", model)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    print("Starting UniOCR evaluation")
    os.makedirs(RESULT_DIR, exist_ok=True)
    for model in MODELS:
        evaluate_model(model)


if __name__ == "__main__":
    main()