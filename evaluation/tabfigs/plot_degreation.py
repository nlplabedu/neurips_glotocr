"""
plot_degradation.py

Dumbbell chart faceted by tier (High / Mid / Low).
For each model that has both img_plain and img_old_document results,
shows Acc@5 on plain vs old_document and the absolute drop.
Sorted by avg plain score across tiers (same order in all panels).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── config ────────────────────────────────────────────────────────────────────
RESULT_DIR  = "../res_v1.0"
OUTPUT_NAME = "degradation_tiers"
ACC_KEY     = "Acc@0.05"

TIERS = [
    ("high_resource", "High-resource\n(Latin)"),
    ("mid_resource",  "Mid-resource\n(9 scripts)"),
    ("low_resource",  "Low-resource\n(148 scripts)"),
]

MODEL_NAMES = {
    "dots-mocr":  "dots.mocr",
    "dots-ocr":      "dots.ocr",
    "glm-ocr-v2":    "GLM-OCR",
    "gpt-4.1":       "GPT-4.1",
    "hunyuan-ocr":   "HunyuanOCR",
    "olmocr2-vllm":  "olmOCR-2",
}

PLAIN_COLOR = "#4D908E"   # teal
OLD_COLOR   = "#E07A5F"   # coral
LINE_COLOR  = "#BBBBBB"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def get_tier_score(summary, tier):
    if summary is None:
        return np.nan
    return summary.get(tier, {}).get(ACC_KEY, np.nan)


# ── collect paired data ───────────────────────────────────────────────────────

def collect(result_dir):
    all_dirs = {
        d: os.path.join(result_dir, d)
        for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d))
    }

    plain_dirs = {k.replace("+img_plain", ""): v
                  for k, v in all_dirs.items() if k.endswith("+img_plain")}
    old_dirs   = {k.replace("+img_old_document", ""): v
                  for k, v in all_dirs.items() if k.endswith("+img_old_document")}

    paired_bases = sorted(set(plain_dirs) & set(old_dirs))

    data = {}
    for base in paired_bases:
        plain_s = load_summary(os.path.join(plain_dirs[base], "summary.json"))
        old_s   = load_summary(os.path.join(old_dirs[base],   "summary.json"))
        if plain_s is None or old_s is None:
            continue

        data[base] = {}
        for tier_key, _ in TIERS:
            p = get_tier_score(plain_s, tier_key)
            o = get_tier_score(old_s,   tier_key)
            data[base][tier_key] = {
                "plain": p,
                "old":   o,
                "drop":  (p - o) if not (np.isnan(p) or np.isnan(o)) else np.nan,
            }

    return data


# ── plot ──────────────────────────────────────────────────────────────────────

def main():
    data = collect(RESULT_DIR)
    if not data:
        print("No paired data found.")
        return

    def avg_plain(base):
        scores = [
            data[base][tier_key]["plain"]
            for tier_key, _ in TIERS
            if not np.isnan(data[base][tier_key]["plain"])
        ]
        return np.mean(scores) if scores else -1

    sorted_bases = sorted(data.keys(), key=avg_plain, reverse=True)
    nice_names   = [MODEL_NAMES.get(b, b) for b in sorted_bases]
    y_pos        = list(range(len(sorted_bases)))

    n_tiers = len(TIERS)
    fig, axes = plt.subplots(1, n_tiers,
                             figsize=(5 * n_tiers, max(4, len(data) * 0.55 + 1.5)),
                             sharey=True)

    bg = "white"
    fig.patch.set_facecolor(bg)

    for ti, (ax, (tier_key, tier_label)) in enumerate(zip(axes, TIERS)):
        ax.set_facecolor(bg)

        for yi, base in enumerate(sorted_bases):
            td   = data[base][tier_key]
            p    = td["plain"] * 100
            o    = td["old"]   * 100
            drop = td["drop"]  * 100 if not np.isnan(td["drop"]) else np.nan

            if np.isnan(p) or np.isnan(o):
                continue

            ax.plot([o, p], [yi, yi], color=LINE_COLOR, linewidth=1.8, zorder=2)
            ax.scatter(p, yi, color=PLAIN_COLOR, s=80, zorder=4,
                       edgecolors="white", linewidths=1.0)
            ax.scatter(o, yi, color=OLD_COLOR,   s=80, zorder=4,
                       edgecolors="white", linewidths=1.0)

            if not np.isnan(drop):
                ax.text(max(p, o) + 0.8, yi, f"−{drop:.1f}",
                        va="center", ha="left", fontsize=11,
                        color="#E07A5F" if drop > 0 else "#4D908E",
                        weight="bold")

        # y labels only on leftmost panel
        ax.set_yticks(y_pos)
        if ti == 0:
            ax.set_yticklabels(nice_names, fontsize=13.5)
            ax.tick_params(labelleft=True, left=True, length=4, width=1.5)
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_linewidth(2.0)
            ax.spines["left"].set_color("#555555")
        else:
            ax.tick_params(labelleft=False, left=False)
            ax.spines["left"].set_visible(False)

        ax.set_xlabel("Acc@5 (%)", fontsize=14)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_title(tier_label, fontsize=15, pad=8)
        ax.set_xlim(0, 108)
        ax.invert_yaxis()

        ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["bottom"].set_color("#555555")

    handles = [
        mlines.Line2D([], [], color=PLAIN_COLOR, marker="o", linestyle="None",
                      markersize=8, label="Clean (img_plain)"),
        mlines.Line2D([], [], color=OLD_COLOR,   marker="o", linestyle="None",
                      markersize=8, label="Degraded (img_old_document)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=12, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTPUT_NAME}.pdf", bbox_inches="tight")
    print(f"Saved: {OUTPUT_NAME}.png / .pdf")


if __name__ == "__main__":
    main()