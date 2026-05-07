"""
plot_script_bubble.py

Bubble chart: X = mean Acc@0.05 across models,
              Y = mean ScriptAcc across models,
              size = log(languages),
              color = resource tier (High / Mid / Low).

Jpan fix: Hani, Hira, Kana predicted for Jpan are treated as correct.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.lines as mlines

# ── config ────────────────────────────────────────────────────────────────────
RESULT_DIR  = "../res_v1.0"
OUTPUT_NAME = "script_bubble"
LANGS_FILE  = "langs_list.txt"

HIGH_SCRIPTS = {"Latn"}
MID_SCRIPTS  = {"Arab", "Cyrl", "Deva", "Hani", "Jpan", "Hang", "Grek", "Hebr", "Thai"}

LABEL_SA_MIN = 0.10   # skip labels for scripts with ScriptAcc < 10%

TIER_COLORS  = {"High": "#E07A5F", "Mid": "#4D908E", "Low": "#B0BEC5"}
TIER_ZORDER  = {"High": 4, "Mid": 3, "Low": 2}

# Scripts whose sub-scripts count as correct identification
SCRIPT_ALIASES = {
    "Jpan": {"Hani", "Hira", "Kana"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_lang_counts(path):
    counts = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("_")
            if len(parts) >= 2:
                counts[parts[1]] = counts.get(parts[1], 0) + 1
    return counts


def load_script_metrics(model_dir):
    path = os.path.join(model_dir, "script_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def load_script_confusion(model_dir):
    path = os.path.join(model_dir, "script_confusion.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def tier(script):
    if script in HIGH_SCRIPTS:
        return "High"
    if script in MID_SCRIPTS:
        return "Mid"
    return "Low"


def corrected_script_acc(true_s, confusion):
    """ScriptAcc treating SCRIPT_ALIASES[true_s] as correct."""
    preds   = confusion.get(true_s, {})
    total   = sum(preds.values())
    if total == 0:
        return np.nan
    aliases = SCRIPT_ALIASES.get(true_s, set())
    correct = preds.get(true_s, 0) + sum(
        cnt for pred_s, cnt in preds.items() if pred_s in aliases
    )
    return correct / total


# ── aggregate across models ───────────────────────────────────────────────────

def aggregate(result_dir):
    acc_accum = {}
    sa_accum  = {}

    models = [
        d for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d))
    ]

    for model in models:
        metrics   = load_script_metrics(os.path.join(result_dir, model))
        confusion = load_script_confusion(os.path.join(result_dir, model))
        if metrics is None:
            continue

        for script, v in metrics.items():
            acc_accum.setdefault(script, []).append(v.get("Acc@0.05", np.nan))

            if script in SCRIPT_ALIASES and confusion is not None:
                sa = corrected_script_acc(script, confusion)
            else:
                sa = v.get("ScriptAcc", np.nan)

            sa_accum.setdefault(script, []).append(sa)

    result = {}
    for script in acc_accum:
        a  = [x for x in acc_accum[script] if not np.isnan(x)]
        sa = [x for x in sa_accum[script]  if not np.isnan(x)]
        if not a or not sa:
            continue
        result[script] = {"acc05": np.mean(a), "script_acc": np.mean(sa)}
    return result


# ── plot ──────────────────────────────────────────────────────────────────────

def main():
    data = aggregate(RESULT_DIR)
    if not data:
        print("No data found — check RESULT_DIR.")
        return

    lang_counts = load_lang_counts(LANGS_FILE)
    log_counts  = {s: np.log1p(lang_counts.get(s, 1)) / np.log1p(2) for s in data}

    fig, ax = plt.subplots(figsize=(10, 7))
    # bg = "#f6f6f6"
    bg = "white"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    texts = []

    for tier_name in ["Low", "Mid", "High"]:
        scripts = [s for s, v in data.items() if tier(s) == tier_name]
        if not scripts:
            continue

        xs    = np.array([data[s]["acc05"]      for s in scripts])
        ys    = np.array([data[s]["script_acc"] for s in scripts])
        sizes = np.array([log_counts[s] ** 2 * 15 + 20 for s in scripts])
        color = TIER_COLORS[tier_name]

        ax.scatter(
            xs * 100, ys * 100,
            s=sizes, c=color, alpha=0.75,
            edgecolors="white", linewidths=0.6,
            zorder=TIER_ZORDER[tier_name],
            label=f"{tier_name}-resource",
        )

        for s, x, y in zip(scripts, xs, ys):
            if y < LABEL_SA_MIN:
                continue
            texts.append(ax.text(
                x * 100, y * 100, s,
                fontsize=10,
                color=color if tier_name != "Low" else "#546E7A",
                weight="bold" if tier_name != "Low" else "normal",
                zorder=5,
            ))

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="#cccccc", linewidth=1, linestyle="--", zorder=1)

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                verbose=False)

    ax.set_xlabel("Acc@5 (%)",     fontsize=15)
    ax.set_ylabel("ScriptAcc (%)", fontsize=15)
    # ax.set_title(
    #     "Script-level recognition accuracy vs. OCR accuracy (Acc@5)\n"
    #     "(averaged across models; bubble size $\\propto$ log number of languages using the script)",
    #     fontsize=11,
    # )

    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        mlines.Line2D([], [], color=TIER_COLORS[t], marker="o", linestyle="None",
                      markersize=8, label=f"{t}-resource")
        for t in ["Low", "Mid", "High"]
    ]
    ax.legend(handles=handles, frameon=True, fancybox=True, fontsize=12,
              loc="lower right", title="Resource tier", title_fontsize=12,
              borderpad=1.2, labelspacing=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    print(f"Saved: {OUTPUT_NAME}.png / .pdf")


if __name__ == "__main__":
    main()