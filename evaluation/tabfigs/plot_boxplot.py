import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR  = "../res_v1.0"
SCRIPTS     = ["Latn", "Deva", "Arab", "Cyrl"]
OUTPUT_NAME = "ocr_boxplot_grid"

MODEL_NAMES = {
    "dots-mocr+img_plain":                     "dots.mocr",
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


def load_lang_metrics(model_dir, script):
    path = os.path.join(model_dir, f"language_metrics_{script}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def main():
    all_models = sorted([
        d for d in os.listdir(RESULT_DIR)
        if os.path.isdir(os.path.join(RESULT_DIR, d))
           and "old" not in d and "hint" not in d
    ])

    data = {m: {} for m in all_models}
    for model in all_models:
        for script in SCRIPTS:
            metrics = load_lang_metrics(os.path.join(RESULT_DIR, model), script)
            if metrics is None:
                continue
            scores = [v["Acc@0.05"] * 100 for v in metrics.values() if "Acc@0.05" in v]
            if scores:
                data[model][script] = scores

    valid_models = [m for m in all_models if data[m]]

    def overall_mean(model):
        all_scores = [s for script in SCRIPTS
                      for s in data[model].get(script, [])]
        return np.mean(all_scores) if all_scores else -1

    valid_models.sort(key=overall_mean, reverse=True)
    nice_names = [MODEL_NAMES.get(m, m) for m in valid_models]
    positions  = list(range(len(valid_models)))

    color_map = {m: COLORS[i % len(COLORS)] for i, m in enumerate(valid_models)}

    n_cols = 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(10, len(valid_models) * 1.1) * n_cols / 1.4,
                                      6 * n_rows),
                             sharex=True)

    background = "white"
    fig.patch.set_facecolor(background)

    for idx, script in enumerate(SCRIPTS):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]
        ax.set_facecolor(background)

        for i, model in enumerate(valid_models):
            scores = data[model].get(script)
            if not scores:
                continue
            color = color_map[model]
            ax.boxplot(
                scores,
                positions=[i],
                widths=0.55,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="#111111", linewidth=2.5),
                boxprops=dict(facecolor=color, color=color, alpha=0.85),
                whiskerprops=dict(color=color, linewidth=1.5),
                capprops=dict(color=color, linewidth=1.5),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=4,
                    alpha=0.6,
                    linewidth=0.5,
                ),
            )

        ax.set_title(script, fontsize=18, fontweight="bold", pad=6)
        ax.set_ylabel("Acc@5 (%)", fontsize=16)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlim(-0.6, len(valid_models) - 0.4)
        ax.set_ylim(0, 105)

        ax.set_xticks(positions)
        if row == n_rows - 1:
            ax.set_xticklabels(nice_names, rotation=30, ha="right", fontsize=15)
        else:
            ax.set_xticklabels([])

        ax.grid(axis="y", color="#d9d9d9", linewidth=1, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    print(f"Saved: {OUTPUT_NAME}.png / .pdf")


if __name__ == "__main__":
    main()