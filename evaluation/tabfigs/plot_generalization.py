import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULT_DIR  = "../res_v1.0"
OUTPUT_NAME = "ocr_generalization"

MODEL_NAMES = {
    "dots-mocr+img_plain":                      "dots.mocr",
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

COLORS = [
    "#E07A5F", "#4D908E", "#E09F3E", "#577590", "#8E7DBE",
    "#43AA8B", "#F94144", "#277DA1", "#F4A261", "#90BE6D",
    "#C77DFF", "#E76F51", "#2A9D8F", "#E9C46A",
]

ACC_KEY = "Acc@0.05"


def load_summary(model_dir):
    path = os.path.join(model_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def main():

    models = [
        d for d in os.listdir(RESULT_DIR)
        if os.path.isdir(os.path.join(RESULT_DIR, d))
    ]

    model_data = []
    for model in models:
        summary = load_summary(os.path.join(RESULT_DIR, model))
        if summary is None:
            continue
        high = summary.get("high_resource", {}).get(ACC_KEY, np.nan)
        mid  = summary.get("mid_resource",  {}).get(ACC_KEY, np.nan)
        low  = summary.get("low_resource",  {}).get(ACC_KEY, np.nan)
        scores = [s for s in [high, mid, low] if not np.isnan(s)]
        avg    = np.mean(scores) if scores else np.nan
        model_data.append((model, high, mid, low, avg))

    # sort by avg descending
    model_data.sort(key=lambda x: x[4] if not np.isnan(x[4]) else -1, reverse=True)

    color_map = {m[0]: COLORS[i % len(COLORS)] for i, m in enumerate(model_data)}

    fig, ax = plt.subplots(figsize=(9, 6))
    # bg = "#f6f6f6"
    bg = "white"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    x      = [0, 1, 2]
    labels = [
        "High-resource\n(1 script  · Latin)",
        "Mid-resource\n(9 scripts · Arab, Cyrl, Deva…)",
        "Low-resource\n(148 scripts)",
    ]

    from adjustText import adjust_text

    # find top-2 and bottom-2 at each x position for labeling
    label_set = set()
    for xi, key in enumerate(["high_resource", "mid_resource", "low_resource"]):
        col = [(m[xi + 1] * 100, m[0]) for m in model_data if not np.isnan(m[xi + 1])]
        col.sort(key=lambda v: v[0])
        for val, mod in col[-2:]:   # top 2 only
            label_set.add((mod, xi))

    texts = []
    for model, high, mid, low, avg in model_data:
        y     = [high * 100, mid * 100, low * 100]
        color = color_map[model]

        ax.plot(x, y, color=color, linewidth=1.6, alpha=0.5, zorder=2)

        ax.scatter([0, 2], [high * 100, low * 100],
                   s=90, color=color, edgecolors="white",
                   linewidths=1.2, zorder=4)

        ax.scatter([1], [mid * 100],
                   s=55, color=color, edgecolors="white",
                   linewidths=1.0, zorder=4)

        for xi, yi in zip(x, y):
            if not np.isnan(yi) and (model, xi) in label_set:
                texts.append(ax.text(
                    xi, yi, f"{yi:.1f}",
                    fontsize=9.6, color=color, weight="bold", zorder=6,
                ))

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
        expand=(1.4, 1.6),
        force_text=(0.3, 0.5),
        force_points=(0.2, 0.3),
        verbose=False,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Acc@5 (%)", fontsize=14)
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(0, 108)

    ax.grid(axis="y", color="#d9d9d9", linewidth=1, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    legend_handles = []
    for model, high, mid, low, avg in model_data:
        if 'old' in model or 'hint' in model:
            continue
        color = color_map[model]
        nice  = MODEL_NAMES.get(model, model)
        legend_handles.append(
            mpatches.Patch(color=color, label=f"{nice}  ({avg*100:.1f}%)")
        )

    legend = ax.legend(
        handles=legend_handles,
        frameon=True, fancybox=True,
        fontsize=8.8, loc="upper right",
        title="Model  (mean of High / Mid / Low)",
        title_fontsize=9.8,
    )
    # legend.get_frame().set_facecolor("#f0eee9")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#dddddd")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    print("Saved:")
    print(f"  {OUTPUT_NAME}.png")
    print(f"  {OUTPUT_NAME}.pdf")


if __name__ == "__main__":
    main()