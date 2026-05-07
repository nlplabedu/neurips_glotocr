"""
plot_hint_gain.py

Scatter plot: X = Acc@5 on img_plain, Y = gain from hint (hint - plain).
One dot per script. Color = resource tier.
Shows whether hints help more on hard or easy scripts.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from adjustText import adjust_text

# ── config ────────────────────────────────────────────────────────────────────
RESULT_DIR  = "../res_v1.0_hint"
OUTPUT_NAME = "hint_gain_scatter"
ACC_KEY     = "Acc@0.05"

PLAIN_DIR = "gpt-4.1+img_plain"
HINT_DIR  = "gpt-4.1+img_plain_hint"

HIGH_SCRIPTS = {"Latn"}
MID_SCRIPTS  = {"Arab", "Cyrl", "Deva", "Hani", "Jpan", "Hang", "Grek", "Hebr", "Thai"}

TIER_COLORS = {"High": "#E07A5F", "Mid": "#4D908E", "Low": "#B0BEC5"}
TIER_ZORDER = {"High": 4, "Mid": 3, "Low": 2}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    plain_sm = load_json(os.path.join(RESULT_DIR, PLAIN_DIR, "script_metrics.json"))
    hint_sm  = load_json(os.path.join(RESULT_DIR, HINT_DIR,  "script_metrics.json"))

    if plain_sm is None or hint_sm is None:
        print("Could not load script_metrics.json — check PLAIN_DIR / HINT_DIR.")
        return

    scripts = sorted(set(plain_sm) & set(hint_sm))

    points = []
    for s in scripts:
        p = plain_sm[s].get(ACC_KEY, np.nan)
        h = hint_sm[s].get(ACC_KEY,  np.nan)
        if np.isnan(p) or np.isnan(h):
            continue
        points.append({
            "script": s,
            "plain":  p * 100,
            "hint":   h * 100,
            "gain":   (h - p) * 100,
            "tier":   tier(s),
        })

    if not points:
        print("No paired script data found.")
        return

    print(f"{len(points)} scripts in plot.")

    fig, ax = plt.subplots(figsize=(9, 7))
    bg = "white"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    texts = []

    for tier_name in ["Low", "Mid", "High"]:
        pts   = [p for p in points if p["tier"] == tier_name]
        xs    = [p["plain"] for p in pts]
        ys    = [p["gain"]  for p in pts]
        color = TIER_COLORS[tier_name]

        ax.scatter(xs, ys, color=color, s=70, alpha=0.8,
                   edgecolors="white", linewidths=0.8,
                   zorder=TIER_ZORDER[tier_name],
                   label=f"{tier_name}-resource")

        for p in pts:
            # skip label only if plain < 15 AND |gain| < 15
            if p["plain"] < 3 and abs(p["gain"]) < 3:
                continue
            texts.append(ax.text(
                p["plain"], p["gain"], p["script"],
                fontsize=11, color=color,
                weight="bold" if tier_name != "Low" else "normal",
                zorder=5,
            ))

    # zero gain reference line
    ax.axhline(0, color="#aaaaaa", linewidth=1.2, linestyle="--", zorder=1)

    # trend line
    all_x = np.array([p["plain"] for p in points])
    all_y = np.array([p["gain"]  for p in points])
    if len(all_x) > 2:
        z    = np.polyfit(all_x, all_y, 1)
        xfit = np.linspace(all_x.min(), all_x.max(), 200)
        ax.plot(xfit, np.polyval(z, xfit),
                color="#999999", linewidth=1.2, linestyle=":",
                zorder=1, label=f"Trend (slope {z[0]:+.2f})")

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5),
        verbose=False,
    )

    ax.set_xlabel("Acc@5 (%s)", fontsize=15)
    ax.set_ylabel("Gain from hint (pp)", fontsize=15)
    # ax.set_title(
    #     "GPT-4.1: does script/language hinting help?\n"
    #     "X = baseline accuracy, Y = hint − plain (positive = improvement)",
    #     fontsize=11,
    # )

    ax.tick_params(labelsize=11)
    ax.grid(color="#eeeeee", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)

    handles = [
        mlines.Line2D([], [], color=TIER_COLORS[t], marker="o", linestyle="None",
                      markersize=8, label=f"{t}-resource")
        for t in ["Low", "Mid", "High"]
    ]
    handles.append(
        mlines.Line2D([], [], color="#999999", linestyle=":",
                      linewidth=1.5, label="Trend line")
    )
    ax.legend(handles=handles, fontsize=10, frameon=True, fancybox=True,
              loc="upper right", title="Resource tier", title_fontsize=11)

    gains  = np.array([p["gain"] for p in points])
    n_pos  = int((gains > 0).sum())
    n_neg  = int((gains < 0).sum())
    n_zero = int((gains == 0).sum())
    ax.text(0.02, 0.02,
            f"Improved: {n_pos}  Hurt: {n_neg}  No change: {n_zero}  "
            f"Mean gain: {gains.mean():+.1f} pp",
            transform=ax.transAxes, fontsize=11, color="#555555",
            va="bottom")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=300)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    print(f"Saved: {OUTPUT_NAME}.png / .pdf")
    print(f"\nSummary: {n_pos} improved, {n_neg} hurt, {n_zero} unchanged")
    print(f"Mean gain: {gains.mean():+.2f} pp  |  Max: {gains.max():+.2f}  |  Min: {gains.min():+.2f}")


if __name__ == "__main__":
    main()