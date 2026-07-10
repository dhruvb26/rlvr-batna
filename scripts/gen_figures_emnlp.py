import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THRESHOLD = "#EF4444"
SURPLUS = "#3B82F6"
BASE = "#D1D5DB"

TEXT = "#1A1A1A"

COL_W = 3.25
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.handlelength": 1.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#999999",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

ROOT = Path(__file__).resolve().parent.parent
SURPLUS_METRICS = ROOT / "logs/negotiation/run-20260509-115751/metrics.jsonl"
THRESHOLD_METRICS = ROOT / "logs/negotiation/run-20260509-142749/metrics.jsonl"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _style(ax, ylabel=None, xlabel="Training Step"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def load_metrics(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def smooth(vals: list[float], w: int = 5) -> np.ndarray:
    if len(vals) <= w:
        return np.array(vals)
    k = np.ones(w) / w
    p = np.pad(vals, (w // 2, w // 2), mode="edge")
    return np.convolve(p, k, mode="valid")[: len(vals)]


def fig1_reward(surplus, batna):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ss = np.array([m["step"] for m in surplus])
    sr = np.array([m["env/all/reward/total"] for m in surplus])
    bs = np.array([m["step"] for m in batna])
    br = np.array([m["env/all/reward/total"] for m in batna])

    sr_sm = smooth(sr)
    br_sm = smooth(br)

    ax.fill_between(ss, sr, sr_sm, color=SURPLUS, alpha=0.08)
    ax.fill_between(bs, br, br_sm, color=THRESHOLD, alpha=0.08)

    ax.plot(ss, sr_sm, color=SURPLUS, lw=1.2, ls="--", label="Surplus")
    ax.plot(bs, br_sm, color=THRESHOLD, lw=1.6, ls="-", label="Threshold (Ours)")

    _style(ax, "Mean Reward")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, max(bs))
    ax.set_ylim(0, 0.72)
    fig.savefig(FIG_DIR / "reward_curves.png")
    plt.close(fig)
    print("  reward_curves.png")


def fig2_first_bid(surplus, batna):
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    ss = np.array([m["step"] for m in surplus])
    sf = np.array([m.get("env/all/first_bid_ratio", float("nan")) for m in surplus])
    bs = np.array([m["step"] for m in batna])
    bf = np.array([m.get("env/all/first_bid_ratio", float("nan")) for m in batna])

    sf_sm = smooth(sf)
    bf_sm = smooth(bf)

    marker_every = 10
    ax.plot(ss, sf_sm, color=SURPLUS, lw=1.0, ls="--", label="Surplus")
    ax.plot(
        bs,
        bf_sm,
        color=THRESHOLD,
        lw=1.4,
        ls="-",
        marker="o",
        markersize=2.5,
        markevery=marker_every,
        markerfacecolor=THRESHOLD,
        markeredgecolor="white",
        markeredgewidth=0.3,
        label="Threshold",
    )
    ax.axhline(0.89, color="#4B5563", ls=":", lw=0.9, label="Base", alpha=0.9)

    _style(ax, "First Bid / Budget")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, max(bs))
    ax.set_ylim(0.15, 1.0)
    fig.savefig(FIG_DIR / "first_bid_ratio.png")
    plt.close(fig)
    print("  first_bid_ratio.png")


def fig3_deal_walk(batna):
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    steps = np.array([m["step"] for m in batna])
    deal = np.array([m["env/all/outcome_deal"] for m in batna])
    walk = np.array([m["env/all/outcome_walk_away"] for m in batna])

    deal_sm = smooth(deal)
    walk_sm = smooth(walk)

    ax.fill_between(steps, deal_sm, walk_sm, color=THRESHOLD, alpha=0.07)
    ax.plot(steps, deal_sm, color=THRESHOLD, lw=1.4, ls="-", label="Deal rate")
    ax.plot(steps, walk_sm, color=SURPLUS, lw=0.9, ls="-.", label="Walk-away rate")

    _style(ax, "Rate")
    ax.legend(frameon=False, loc="center right")
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, 1.0)
    fig.savefig(FIG_DIR / "deal_walk_rate.png")
    plt.close(fig)
    print("  deal_walk_rate.png")


def fig4_per_dataset(batna):
    steps = np.array([m["step"] for m in batna])
    panels = [
        (
            "Price",
            "per_dataset_br_price.png",
            [
                ("AHP", "env/all/bargained_ratio/amazon", THRESHOLD, "-"),
                ("CRA", "env/all/bargained_ratio/craigslist", SURPLUS, "--"),
            ],
        ),
        (
            "Multi-item",
            "per_dataset_br_multi.png",
            [
                ("DnD", "env/all/bargained_ratio/dnd", THRESHOLD, "-"),
                ("CA", "env/all/bargained_ratio/casino", SURPLUS, "--"),
            ],
        ),
    ]
    for title, fname, lines in panels:
        fig, ax = plt.subplots(figsize=(COL_W, 2.0))
        for name, key, c, ls in lines:
            v = np.array([m.get(key, float("nan")) for m in batna])
            v_sm = smooth(v)
            ax.plot(steps, v_sm, color=c, lw=0.9, ls=ls, label=name)
        ax.axhline(0, color="#D1D5DB", lw=0.6)
        ax.set_title(title, fontsize=9)
        _style(ax, "Bargained Ratio")
        ax.set_xlim(0, max(steps))
        ax.legend(
            frameon=False,
            ncol=2,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
        )
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)
        print(f"  {fname}")


def fig_reward_design():
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    tau = 0.4
    rho = np.linspace(0, 1, 500)
    surplus_r = rho.copy()
    thresh_r = np.where(rho >= tau, rho, -0.5)

    ax.axvspan(0, tau, color=THRESHOLD, alpha=0.04, zorder=0)
    ax.axhline(0, color="#666666", ls="-", lw=0.6, zorder=2)

    ax.plot(rho, surplus_r, color=SURPLUS, lw=1.4, ls="--", label="Surplus", zorder=3)
    ax.plot(
        rho[rho < tau],
        thresh_r[rho < tau],
        color=THRESHOLD,
        lw=1.8,
        ls="-",
        label="Threshold (Ours)",
        zorder=4,
    )
    ax.plot(
        rho[rho >= tau], thresh_r[rho >= tau], color=THRESHOLD, lw=1.8, ls="-", zorder=4
    )
    ax.plot(tau, -0.5, "o", color=THRESHOLD, ms=4, mfc="white", mew=1.2, zorder=5)
    ax.plot(tau, tau, "o", color=THRESHOLD, ms=4, mfc=THRESHOLD, mew=0.6, zorder=5)

    ax.axvline(tau, color=THRESHOLD, ls=":", lw=0.7, alpha=0.5)
    ax.text(tau + 0.04, -0.55, f"τ = {tau}", fontsize=8, color=THRESHOLD, ha="left")

    _style(ax, ylabel="Reward", xlabel="Deal Quality (ρ)")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.65, 1.08)
    fig.savefig(FIG_DIR / "reward_design.png")
    plt.close(fig)
    print("  reward_design.png")


def fig_bar_results():
    names = ["Amazon", "CaSiNo", "Craigslist", "Deal or No Deal", "Job Interview"]
    base_v = [-0.054, 0.519, 0.013, 0.668, 0.711]
    surplus_v = [0.635, 0.582, 0.607, 0.550, 0.625]
    batna_v = [0.800, 0.581, 0.646, 0.701, 0.703]

    x = np.arange(len(names))
    w = 0.24
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    ax.bar(
        x - w,
        base_v,
        w,
        label="Qwen3-30B-A3B (Base)",
        color=BASE,
        edgecolor=BASE,
        lw=0.4,
        hatch="//",
        alpha=0.80,
    )
    ax.bar(
        x,
        surplus_v,
        w,
        label="Surplus",
        color=SURPLUS,
        edgecolor=SURPLUS,
        lw=0.4,
    )
    bars_t = ax.bar(
        x + w,
        batna_v,
        w,
        label="Threshold (Ours)",
        color=THRESHOLD,
        edgecolor=THRESHOLD,
        lw=0.4,
    )
    for bar in bars_t:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(h, 0) + 0.02,
            f".{int(abs(h) * 100):02d}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
            color=TEXT,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    _style(ax, ylabel="Bargained Ratio", xlabel=None)
    ax.set_ylim(-0.65, 0.95)
    ax.axhline(0, color=TEXT, lw=0.25)
    gpt_avg = np.mean([0.517, 0.519, 0.268, 0.687, 0.731])
    ax.axhline(
        gpt_avg,
        color="#4B5563",
        ls=":",
        lw=0.9,
        alpha=0.9,
        zorder=1,
        label="GPT-5.4 (avg)",
    )
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.savefig(FIG_DIR / "bar_results.png")
    plt.close(fig)
    print("  bar_results.png")


if __name__ == "__main__":
    print("Loading metrics...")
    surplus = load_metrics(SURPLUS_METRICS)
    batna = load_metrics(THRESHOLD_METRICS)
    print(f"  Surplus: {len(surplus)} steps, THRESHOLD: {len(batna)} steps")

    print("Generating figures...")
    fig_reward_design()
    fig_bar_results()
    fig1_reward(surplus, batna)
    fig2_first_bid(surplus, batna)
    fig3_deal_walk(batna)
    fig4_per_dataset(batna)
    print("Done.")
