"""
generate_script_table.py

Generates a LaTeX longtable:
  Rows    : all scripts, sorted by sample count desc, then avg Acc@5 desc
  Columns : Script | Samples | Top-2 confusion (colored) | Acc@5 per top-K model
  Models  : top TOP_K by combined #1+#2 Acc@5 rankings; shown as auto-abbreviations
  Confusion column: script name only, blue=correct (intensity=%), red=wrong (intensity=%)
"""

import os
import json
import numpy as np
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
RESULT_DIR  = "../res_v1.0"
OUTPUT_FILE = "table_script_long.tex"
TOP_K       = 5

UNKNOWN_LABELS = {"Zyyy", "Zzzz", "Zinh", "null", "None", "Unk", "Unknown"}

MODEL_NAMES = {
    "dots-ocr-1.5+img_plain":                  "dots.ocr-1.5",
    "paddleocr-vl-1.5+img_plain":              "PaddleOCR-VL-1.5",
    "gemini-3.1-flash-lite-preview+img_plain":  "Gemini-Flash-Lite",
    "gpt-4.1+img_plain":                        "GPT-4.1",
    "dots-ocr+img_plain":                       "dots.ocr",
    "glm-ocr-v2+img_plain":                     "GLM-OCR",
    "deepseek-ocr2-vllm+img_plain":             "DeepSeek-OCR-2",
    "olmocr2-vllm+img_plain":                   "olmOCR-2",
    "nanonets-ocr2+img_plain":                  "Nanonets-OCR2",
    "firered-ocr+img_plain":                    "FireRed-OCR",
    "lighton-ocr2+img_plain":                   "LightOnOCR-2",
    "rolm-ocr+img_plain":                       "RolmOCR",
    "hunyuan-ocr+img_plain":                    "HunyuanOCR",
    "qwen3-vl-8b+img_plain":                    "Qwen3-VL-8B",
}


# ── abbreviations ─────────────────────────────────────────────────────────────

MODEL_ABBREVS = {
    "dots-ocr-1.5+img_plain":                  "dots-1.5",
    "paddleocr-vl-1.5+img_plain":              "Paddle-1.5",
    "gemini-3.1-flash-lite-preview+img_plain":  "Gemini-FL",
    "gpt-4.1+img_plain":                        "GPT-4.1",
    "dots-ocr+img_plain":                       "dots",
    "glm-ocr-v2+img_plain":                     "GLM-v2",
    "deepseek-ocr2-vllm+img_plain":             "DS-OCR2",
    "olmocr2-vllm+img_plain":                   "olmOCR2",
    "nanonets-ocr2+img_plain":                  "Nano-2",
    "firered-ocr+img_plain":                    "FireRed",
    "lighton-ocr2+img_plain":                   "LightOn2",
    "rolm-ocr+img_plain":                       "RolmOCR",
    "hunyuan-ocr+img_plain":                    "Hunyuan",
    "qwen3-vl-8b+img_plain":                    "Qwen3-8B",
}


# ── script aliases (treated as correct predictions) ──────────────────────────
# key = true script, value = set of predicted scripts considered correct
SCRIPT_ALIASES = {
    "Jpan": {"Kana", "Hira", "Hani"},
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def list_models(result_dir):
    return [
        d for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d))
    ]


# ── data collection ───────────────────────────────────────────────────────────

def collect(result_dir):
    models = list_models(result_dir)
    script_metrics  = {}
    script_samples  = {}
    conf_accum      = defaultdict(lambda: defaultdict(list))
    per_model_conf  = defaultdict(dict)   # model -> {true_s -> {pred_s -> fraction}}

    for model in models:
        sm = load_json(os.path.join(result_dir, model, "script_metrics.json"))
        cf = load_json(os.path.join(result_dir, model, "script_confusion.json"))

        if sm:
            script_metrics[model] = sm
            for script, v in sm.items():
                if script not in script_samples:
                    script_samples[script] = v.get("samples", 0)

        if cf:
            for true_s, preds in cf.items():
                total = sum(preds.values())
                if total == 0:
                    continue
                aliases = SCRIPT_ALIASES.get(true_s, set())
                per_model_conf[model][true_s] = {}
                for pred_s, cnt in preds.items():
                    if pred_s in UNKNOWN_LABELS:
                        continue
                    norm_pred = true_s if pred_s in aliases else pred_s
                    conf_accum[true_s][norm_pred].append(cnt / total)
                    per_model_conf[model][true_s][norm_pred] = (
                        per_model_conf[model][true_s].get(norm_pred, 0) + cnt / total
                    )

    confusion_avg = {
        s: {p: np.mean(fracs) for p, fracs in preds.items()}
        for s, preds in conf_accum.items()
    }
    return script_metrics, script_samples, confusion_avg, conf_accum, per_model_conf


def build_group_confusion(per_model_conf, models):
    """Average confusion fractions over a specific subset of models."""
    accum = defaultdict(lambda: defaultdict(list))
    for m in models:
        for true_s, preds in per_model_conf.get(m, {}).items():
            for pred_s, frac in preds.items():
                accum[true_s][pred_s].append(frac)
    return {
        s: {p: np.mean(fracs) for p, fracs in preds.items()}
        for s, preds in accum.items()
    }


# ── select top-K models ───────────────────────────────────────────────────────

def select_top_models(script_metrics, top_k):
    """
    Rank models per script by Acc@0.05.
    First fill slots by #1+#2 finishes, then expand to #3, #4, ...
    until top_k slots are filled. Final tiebreaker: overall avg Acc@0.05.
    """
    all_scripts = set()
    for sm in script_metrics.values():
        all_scripts.update(sm.keys())

    # pre-compute per-script ranked lists
    script_rankings = {}
    for script in all_scripts:
        scores = sorted(
            [(m, sm[script].get("Acc@0.05", 0.0))
             for m, sm in script_metrics.items() if script in sm],
            key=lambda x: x[1], reverse=True
        )
        script_rankings[script] = [m for m, _ in scores]

    all_models = list(script_metrics.keys())
    selected   = []

    max_rank = max((len(v) for v in script_rankings.values()), default=1)

    for rank_window in range(2, max_rank + 1):
        if len(selected) >= top_k:
            break
        rank_counts = defaultdict(int)
        for ranked in script_rankings.values():
            for m in ranked[:rank_window]:
                rank_counts[m] += 1
        # add models not yet selected, ordered by count desc
        candidates = sorted(
            [m for m in all_models if m not in selected],
            key=lambda m: rank_counts.get(m, 0), reverse=True
        )
        for m in candidates:
            if len(selected) >= top_k:
                break
            if rank_counts.get(m, 0) > 0:
                selected.append(m)

    # final fallback: fill by overall avg Acc@0.05
    if len(selected) < top_k:
        avg_score = {
            m: np.mean([
                sm.get(s, {}).get("Acc@0.05", 0.0)
                for s in all_scripts
            ])
            for m, sm in script_metrics.items()
        }
        remaining = sorted(
            [m for m in all_models if m not in selected],
            key=lambda m: avg_score.get(m, 0.0), reverse=True
        )
        selected.extend(remaining[:top_k - len(selected)])

    return selected[:top_k]


# ── colour helpers ────────────────────────────────────────────────────────────

def _hex(r, g, b):
    return f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def blue_color(v, lo, hi):
    if np.isnan(v) or hi <= lo:
        return "FFFFFF"
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return _hex(1.0 - 0.478 * t, 1.0 - 0.243 * t, 1.0 - 0.086 * t)


# ── confusion column ──────────────────────────────────────────────────────────

def conf_cells(conf_avg, script, top_n=2, models=None):
    """Return plain 'ScriptA, ScriptB' — top-2 predicted, unknowns excluded.
    If models is provided, confusion is already pre-averaged for that group."""
    preds    = conf_avg.get(script, {})
    filtered = {p: v for p, v in preds.items() if p not in UNKNOWN_LABELS}
    top      = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]
    parts    = [p for p, v in top if v > 0.005]
    return ", ".join(parts) if parts else "---"


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def fmt(v):
    return "---" if np.isnan(v) else f"{v * 100:.1f}"


def colored_cell(v, lo, hi):
    if np.isnan(v):
        return "---"
    return f"\\cellcolor[HTML]{{{blue_color(v, lo, hi)}}}{fmt(v)}"


# ── table layout ──────────────────────────────────────────────────────────────
ROWS_PER_CHUNK   = 30   # scripts per pair of tables
MODELS_PER_TABLE = 7    # how many models per table (top-7 then bottom-7)
MAX_TABLES       = 10   # max number of paired table chunks to emit (None = no limit)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    script_metrics, script_samples, confusion_avg, conf_accum, per_model_conf = collect(RESULT_DIR)
    all_models = select_top_models(script_metrics, 14)

    print("All models ranked:")
    for i, m in enumerate(all_models, 1):
        group = "top-7" if i <= MODELS_PER_TABLE else "bot-7"
        print(f"  {i:2d} [{group}] {MODEL_ABBREVS.get(m, MODEL_NAMES.get(m, m)):12s} = {MODEL_NAMES.get(m, m)}")

    top_models = all_models[:MODELS_PER_TABLE]
    bot_models = all_models[MODELS_PER_TABLE:]

    # group-specific confusion averages for Top-2 pred column
    top_conf = build_group_confusion(per_model_conf, top_models)
    bot_conf = build_group_confusion(per_model_conf, bot_models)

    # ── fix ScriptAcc for alias scripts using corrected confusion ────────────
    for m in all_models:
        if m not in script_metrics:
            continue
        cf = load_json(os.path.join(RESULT_DIR, m, "script_confusion.json"))
        if cf is None:
            continue
        for true_s, aliases in SCRIPT_ALIASES.items():
            if true_s not in script_metrics.get(m, {}):
                continue
            preds   = cf.get(true_s, {})
            total   = sum(preds.values())
            if total == 0:
                continue
            correct = preds.get(true_s, 0) + sum(
                cnt for pred_s, cnt in preds.items() if pred_s in aliases
            )
            script_metrics[m][true_s]["ScriptAcc"] = correct / total

    # ── detect zero-SA scripts (all models ScriptAcc == 0) ────────────────────
    def all_zero_sa(s):
        return all(
            script_metrics.get(m, {}).get(s, {}).get("ScriptAcc", 0.0) == 0.0
            for m in all_models
            if m in script_metrics and s in script_metrics[m]
        )

    zero_sa_scripts  = []
    normal_scripts_unsorted = []
    for s in script_samples:
        (zero_sa_scripts if all_zero_sa(s) else normal_scripts_unsorted).append(s)

    print(f"\nZero-SA scripts: {len(zero_sa_scripts)}, Normal scripts: {len(normal_scripts_unsorted)}")

    # sort normal scripts by top-7 models
    normal_scripts = sorted(
        normal_scripts_unsorted,
        key=lambda s: (
            -script_samples.get(s, 0),
            -np.mean([script_metrics[m][s].get("Acc@0.05", 0.0)
                      for m in top_models if m in script_metrics and s in script_metrics[m]
                      ] or [0.0]),
            -np.mean([script_metrics[m][s].get("ScriptAcc", 0.0)
                      for m in top_models if m in script_metrics and s in script_metrics[m]
                      ] or [0.0]),
        )
    )

    # sort zero-SA scripts by sample count desc
    zero_sa_scripts.sort(key=lambda s: -script_samples.get(s, 0))

    def col_stats(models, scripts, metric_key):
        vals = {
            m: [script_metrics[m][s].get(metric_key, np.nan)
                for s in scripts
                if m in script_metrics and s in script_metrics.get(m, {})]
            for m in models
        }
        return (
            {m: np.nanmin(v) if v else 0.0 for m, v in vals.items()},
            {m: np.nanmax(v) if v else 1.0 for m, v in vals.items()},
        )

    top_acc_min, top_acc_max = col_stats(top_models, normal_scripts, "Acc@0.05")
    bot_acc_min, bot_acc_max = col_stats(bot_models, normal_scripts, "Acc@0.05")

    def col_spec(models):
        return "l r p{2.2cm} " + " ".join(["r"] * len(models))

    def model_header_row(models):
        return " & ".join(
            "\\makecell{\\textbf{" + MODEL_ABBREVS.get(m, MODEL_NAMES.get(m, m))
            + "}\\\\A5/SA$\\uparrow$}" for m in models
        )

    def make_abbrev_list(models):
        return "; ".join(
            "\\textbf{" + MODEL_ABBREVS.get(m, MODEL_NAMES.get(m, m))
            + "}=" + MODEL_NAMES.get(m, m) for m in models
        )

    def fixed_header(models):
        return (
            "\\textbf{Script} & \\textbf{$n$} & \\textbf{Top-2 pred.} & "
            + model_header_row(models) + " \\\\\n\\midrule\n"
        )

    def build_row(s, models, acc_min, acc_max, conf_dict):
        n      = script_samples.get(s, 0)
        cf_col = conf_cells(conf_dict, s)
        cells  = []
        for m in models:
            mv   = script_metrics.get(m, {}).get(s, {})
            acc  = mv.get("Acc@0.05",  np.nan)
            sacc = mv.get("ScriptAcc", np.nan)
            color = blue_color(acc, acc_min[m], acc_max[m])
            cells.append("\\cellcolor[HTML]{" + color + "}" + fmt(acc) + "/" + fmt(sacc))
        return "  " + s + " & " + str(n) + " & " + cf_col + " & " + " & ".join(cells) + " \\\\\n"

    top_abbrev_list = make_abbrev_list(top_models)
    bot_abbrev_list = make_abbrev_list(bot_models)

    # ── document preamble ─────────────────────────────────────────────────────
    lines = [
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{colortbl}\n"
        "\\usepackage{makecell}\n"
        "\\usepackage{array}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage[margin=0.7in]{geometry}\n\n"
        "\\begin{document}\n\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n\n"
    ]

    # ── zero-SA compact multi-column table ───────────────────────────────────
    N_ZERO_COLS = 3   # how many script entries side by side per row

    if zero_sa_scripts:
        # pad to multiple of N_ZERO_COLS
        padded = zero_sa_scripts + [""] * ((-len(zero_sa_scripts)) % N_ZERO_COLS)
        rows   = [padded[i:i+N_ZERO_COLS] for i in range(0, len(padded), N_ZERO_COLS)]

        # col spec: N_ZERO_COLS groups of (l r p{2cm}), separated by a small gap col
        single  = "l r p{2cm}"
        gap     = "@{\\hspace{8pt}}"
        tabspec = gap.join([single] * N_ZERO_COLS)

        lines.append(
            "\\begin{table}[ht]\n\\centering\n"
            "\\caption{Scripts where all models achieve zero ScriptAcc. "
            "These scripts are not identifiable by any model. "
            "Top-2 pred.\\ shows where samples are misclassified (avg over all models).}\n"
            "\\label{tab:zero_sa}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{" + tabspec + "}\n"
            "\\toprule\n"
        )

        # header: repeat for each group
        hcell  = "\\textbf{Script} & \\textbf{$n$} & \\textbf{Top-2 pred.}"
        lines.append(" & ".join([hcell] * N_ZERO_COLS) + " \\\\\n\\midrule\n")

        for row in rows:
            cells = []
            for s in row:
                if s == "":
                    cells.append(" & & ")
                else:
                    n      = script_samples.get(s, 0)
                    cf_col = conf_cells(confusion_avg, s)
                    cells.append(s + " & " + str(n) + " & " + cf_col)
            lines.append(" & ".join(cells) + " \\\\\n")

        lines.append(
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}%resizebox\n"
            "\\end{table}\n\n"
        )

    # ── normal paired tables ──────────────────────────────────────────────────
    chunks = [normal_scripts[i:i+ROWS_PER_CHUNK]
              for i in range(0, len(normal_scripts), ROWS_PER_CHUNK)]

    # apply MAX_TABLES cap
    if MAX_TABLES is not None:
        chunks = chunks[:MAX_TABLES]

    for ci, chunk in enumerate(chunks):
        script_range = chunk[0] + "--" + chunk[-1]

        for gi, (models, acc_min, acc_max, abbrev_list, group_label, conf_dict) in enumerate([
            (top_models, top_acc_min, top_acc_max, top_abbrev_list, "Top-7 models", top_conf),
            (bot_models, bot_acc_min, bot_acc_max, bot_abbrev_list, "Bottom-7 models", bot_conf),
        ]):
            tab_label = "tab:script_" + str(ci + 1) + ("a" if gi == 0 else "b")
            if ci == 0 and gi == 0:
                caption = (
                    "\\caption{Acc@5/ScriptAcc (\\%) per script --- "
                    + group_label + " (sorted by top-7 avg). "
                    "Each cell: A5/SA; blue $\\propto$ A5 per column. "
                    "Abbreviations: " + abbrev_list + ".}"
                )
            else:
                caption = (
                    "\\caption[]{"
                    + group_label + ", scripts " + script_range
                    + ". Abbreviations: " + abbrev_list + ".}"
                )

            lines.append("\\begin{table}[ht]\n\\centering\n")
            lines.append(caption + "\n\\label{" + tab_label + "}\n")
            lines.append(
                "\\resizebox{\\textwidth}{!}{%\n"
                "\\begin{tabular}{" + col_spec(models) + "}\n"
                "\\toprule\n"
            )
            lines.append(fixed_header(models))
            for s in chunk:
                lines.append(build_row(s, models, acc_min, acc_max, conf_dict))
            lines.append(
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "}%resizebox\n"
                "\\end{table}\n\n"
            )

    lines.append("\\end{document}\n")

    with open(OUTPUT_FILE, "w", encoding="utf8") as f:
        f.writelines(lines)

    print(f"\nSaved: {OUTPUT_FILE}  "
          f"({len(normal_scripts)} normal scripts, {len(zero_sa_scripts)} zero-SA, "
          f"{len(chunks)} chunks x2 tables, MAX_TABLES={MAX_TABLES})")


if __name__ == "__main__":
    main()