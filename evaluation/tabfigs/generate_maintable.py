"""
generate_table.py
Reads summary.json for every model and writes a filled LaTeX table.
"""

import os
import json
import numpy as np

RESULT_DIR  = "../res_v1.0"
OUTPUT_FILE = "main_tab.tex"

MODEL_NAMES = {
    "dots-ocr-1.5+img_plain":                  "dots.ocr-1.5",
    "paddleocr-vl-1.5+img_plain":              "PaddleOCR-VL-1.5",
    "gemini-3.1-flash-lite-preview+img_plain":  "Gemini 3.1 Flash-Lite",
    "gpt-4.1+img_plain":                        "GPT4.1",
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

TIERS   = ["high_resource", "mid_resource", "low_resource"]
METRICS = ["CER", "Acc@0.00", "Acc@0.05", "Acc@0.10"]
CER_KEY = "CER"   # lower = better


# ── helpers ───────────────────────────────────────────────────────────────────

def load_summary(model_dir):
    path = os.path.join(model_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def tier_values(summary, tier):
    block = summary.get(tier, {})
    return tuple(block.get(m, np.nan) for m in METRICS)


def mean_tier(v_high, v_mid, v_low):
    result = []
    for h, m, l in zip(v_high, v_mid, v_low):
        vals = [x for x in [h, m, l] if not np.isnan(x)]
        result.append(np.mean(vals) if vals else np.nan)
    return tuple(result)


def fmt(v):
    return "---" if np.isnan(v) else f"{v * 100:.1f}"


def _norm(v, lo, hi):
    if np.isnan(v) or hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def _hex(r, g, b):
    return f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def heatmap_color(v, lo, hi, is_cer):
    t = _norm(v, lo, hi)
    if is_cer:
        r = 1.0 - 0.043 * t
        g = 1.0 - 0.373 * t
        b = 1.0 - 0.341 * t
    else:
        r = 1.0 - 0.478 * t
        g = 1.0 - 0.243 * t
        b = 1.0 - 0.086 * t
    return _hex(r, g, b)


def cell(values, col_min, col_max, best, second_best):
    cells = []
    for i, (v, m) in enumerate(zip(values, METRICS)):
        s    = fmt(v)
        is_c = (m == CER_KEY)
        if not np.isnan(v):
            color = heatmap_color(v, col_min[i], col_max[i], is_c)
            bg    = f"\\cellcolor[HTML]{{{color}}}"
            if abs(v - best[i]) < 1e-9:
                s = f"\\mathbf{{{s}}}"
            elif not np.isnan(second_best[i]) and abs(v - second_best[i]) < 1e-9:
                s = f"\\underline{{{s}}}"
            cells.append(f"{bg}${s}$")
        else:
            cells.append("---")
    return " & ".join(cells)


# ── main ──────────────────────────────────────────────────────────────────────

def main():

    # --- collect data ---------------------------------------------------------
    rows = []
    for key, name in MODEL_NAMES.items():
        summary = load_summary(os.path.join(RESULT_DIR, key))
        if summary is None:
            continue
        v_high = tier_values(summary, "high_resource")
        v_mid  = tier_values(summary, "mid_resource")
        v_low  = tier_values(summary, "low_resource")
        v_mean = mean_tier(v_high, v_mid, v_low)
        rows.append((name, v_high, v_mid, v_low, v_mean))

    # sort by Overall-Mean Acc@0.05 descending
    rows.sort(key=lambda r: r[4][2] if not np.isnan(r[4][2]) else -1, reverse=True)

    # --- compute per-column best/second_best/min/max --------------------------
    all_tier_blocks = [
        [r[1] for r in rows],   # high
        [r[2] for r in rows],   # mid
        [r[3] for r in rows],   # low
        [r[4] for r in rows],   # mean
    ]

    def col_stats(blocks):
        best, second_best, col_min, col_max = [], [], [], []
        for mi, m in enumerate(METRICS):
            vals = sorted(
                [b[mi] for b in blocks if not np.isnan(b[mi])],
                reverse=(m != CER_KEY),
            )
            if not vals:
                best.append(np.nan); second_best.append(np.nan)
                col_min.append(np.nan); col_max.append(np.nan)
            else:
                best.append(vals[0])
                second_best.append(vals[1] if len(vals) > 1 else np.nan)
                col_min.append(min(vals)); col_max.append(max(vals))
        return best, second_best, col_min, col_max

    bw = [col_stats(blk) for blk in all_tier_blocks]

    # --- write LaTeX ----------------------------------------------------------
    header = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{adjustbox}
\usepackage[margin=0.6in, landscape]{geometry}

\definecolor{tierHigh}{HTML}{FFFFFF}
\definecolor{tierMid} {HTML}{FFFFFF}
\definecolor{tierLow} {HTML}{FFFFFF}
\definecolor{tierMean}{HTML}{FFFFFF}

\begin{document}

\begin{table}[ht]
\centering
\caption{%
  UniOCR results across resource tiers.
  \textbf{High}: Latin (1 script).
  \textbf{Mid}: 9 scripts (Arab, Cyrl, Deva, Hani, Jpan, Hang, Grek, Hebr, Thai).
  \textbf{Low}: 148 remaining scripts.
  \textbf{Mean}: equal-weight mean of the three tiers.
  CER$\downarrow$; Acc$\uparrow$ (\%).
  \textbf{Bold} = best; \underline{underline} = 2nd best.
}
\label{tab:uniocr}
\vspace{4pt}
\adjustbox{max width=\linewidth}{%
\begin{tabular}{l | cccc | cccc | cccc | cccc}
\toprule
\multirow{2}{*}{\textbf{Model}}
  & \multicolumn{4}{c|}{\cellcolor{tierHigh}\textbf{High} (1 script)}
  & \multicolumn{4}{c|}{\cellcolor{tierMid}\textbf{Mid} (9 scripts)}
  & \multicolumn{4}{c|}{\cellcolor{tierLow}\textbf{Low} (148 scripts)}
  & \multicolumn{4}{c} {\cellcolor{tierMean}\textbf{Overall (mean across tiers)}}
\\
"""

    subheader = (
        "  & "
        + " & ".join([r"\makecell{CER\\$\downarrow$}", r"\makecell{A@0\\$\uparrow$}",
                       r"\makecell{A@5\\$\uparrow$}", r"\makecell{A@10\\$\uparrow$}"] * 4)
        + r" \\"
        + "\n\\midrule\n"
    )

    footer = r"""\bottomrule
\end{tabular}
}% end adjustbox
\end{table}
% A@0 = Acc@0.00, A@5 = Acc@0.05, A@10 = Acc@0.10. Values are percentages.
\end{document}
"""

    lines = [header, subheader]
    for name, v_high, v_mid, v_low, v_mean in rows:
        parts = [
            cell(v_high, bw[0][2], bw[0][3], bw[0][0], bw[0][1]),
            cell(v_mid,  bw[1][2], bw[1][3], bw[1][0], bw[1][1]),
            cell(v_low,  bw[2][2], bw[2][3], bw[2][0], bw[2][1]),
            cell(v_mean, bw[3][2], bw[3][3], bw[3][0], bw[3][1]),
        ]
        lines.append(f"  {name} & " + " & ".join(parts) + r" \\" + "\n")

    lines.append(footer)

    with open(OUTPUT_FILE, "w", encoding="utf8") as f:
        f.writelines(lines)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()