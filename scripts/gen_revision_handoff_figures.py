"""Rebuild nine IJPR revision plots from saved local training/evaluation results.

Run from the repository root: python -m scripts.gen_revision_handoff_figures.
Writes PNG/PDF pairs and figure_data.json only inside the output directory.
No inference, source-result edits, or handoff/document updates are performed.
The three overview/sensitivity plots in the snapshot are separate artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts import gen_figures_ijpr as style

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = Path("ijpr-submission/inventory-2026-10-04/inventory.json")
DEFAULT_ANALYSIS = Path(
    "ijpr-submission/analysis-quality-mechanism-2026-10-04/results.json"
)
DEFAULT_OUTPUT = Path("ijpr-submission/figures/revision-2026-10-04")


def source_path(value, root):
    """Resolve saved repository paths against this checkout, including old roots."""
    path = Path(value)
    if not path.is_absolute():
        return root / path
    # Rebase historical run/manifest references instead of reading an old clone.
    for marker in ("logs", "ijpr-submission"):
        if marker in path.parts:
            return root.joinpath(*path.parts[path.parts.index(marker) :])
    return path


def relative_source(value, root):
    path = source_path(value, root)
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def generate(
    root=DEFAULT_ROOT,
    *,
    inventory=DEFAULT_INVENTORY,
    analysis=DEFAULT_ANALYSIS,
    output=DEFAULT_OUTPUT,
):
    """Generate plots explicitly; importing this module creates no output files."""
    root = Path(root).resolve()
    inventory = Path(inventory)
    analysis = Path(analysis)
    output = Path(output)
    inventory = (root / inventory).resolve()
    analysis = (root / analysis).resolve()
    output_dir = (root / output).resolve()
    for path in (inventory, analysis):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required saved analysis input not found: {path}. "
                "Restore the local evidence package or supply --inventory/--analysis."
            )
    inventory_data = json.loads(inventory.read_text())
    analysis_data = json.loads(analysis.read_text())
    red = style.THRESHOLD
    blue = style.SURPLUS
    gray = style.BASE
    slate = style.WALK
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    train = {}
    inputs = {}
    for t in inventory_data["training"]:
        rows_path = source_path(t["run"], root) / "metrics.jsonl"
        if not rows_path.is_file():
            raise FileNotFoundError(
                f"Training metrics not found: {rows_path}. Restore saved logs under --root/logs/."
            )
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
        if len(rows) != 80 or [row["step"] for row in rows] != list(range(80)):
            raise ValueError(
                f"Expected exactly 80 ordered training updates: {rows_path}"
            )
        train[(t["method"], t["tau"], t["seed"])] = rows
    output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(output_dir / f"{name}.png", dpi=300)
        fig.savefig(output_dir / f"{name}.pdf")
        plt.close(fig)

    def curve(ax, cond, metric, color, label, ls="-", marker=None):
        raw = np.array(
            [[r[metric] for r in train[(*cond, seed)]] for seed in (42, 43, 44)]
        )
        assert np.isfinite(raw).all()
        smooth = np.array([style.smooth(v, 5) for v in raw])
        mean = smooth.mean(axis=0)
        sd = smooth.std(axis=0, ddof=1)
        ax.plot(
            range(80),
            mean,
            color=color,
            lw=1.4 if ls == "-" else 1.1,
            ls=ls,
            label=label,
            marker=marker,
            markersize=2.5,
            markevery=10,
            markeredgecolor="white",
            markeredgewidth=0.3,
        )
        ax.fill_between(range(80), mean - sd, mean + sd, color=color, alpha=0.10, lw=0)
        inputs[str((cond, metric))] = {
            "raw_seed42_43_44": raw.tolist(),
            "smoothed_mean": mean.tolist(),
            "smoothed_sample_sd": sd.tolist(),
        }
        ax.set_xlim(0, 79)

    def comparison(name, metric, ylabel, marker=False):
        fig, ax = plt.subplots(figsize=(style.COL_W, 2))
        curve(ax, ("surplus", 0), metric, blue, "Surplus", "--")
        curve(
            ax,
            ("selective", 0.4),
            metric,
            red,
            "Selective Threshold",
            marker="o" if marker else None,
        )
        style._style(ax, ylabel)
        ax.legend(frameon=False, loc="best", fontsize=7)
        if name == "reward_curves":
            ax.set_ylim(0, 0.9)
        if name == "first_bid_ratio":
            ax.set_ylim(0.15, 1)
        save(fig, name)

    comparison("reward_curves", "env/all/reward/total", "Mean Episode Reward")
    comparison(
        "first_bid_ratio", "env/all/first_bid_ratio", "Opening Bid / Budget", True
    )
    for name, title, pairs in [
        (
            "per_dataset_br_price",
            "Price",
            [("AHP", "amazon", red, "-"), ("CRA", "craigslist", blue, "--")],
        ),
        (
            "per_dataset_br_multi",
            "Multi-item",
            [("DnD", "dnd", red, "-"), ("CA", "casino", blue, "--")],
        ),
    ]:
        fig, ax = plt.subplots(figsize=(style.COL_W, 2))
        for label, ds, color, ls in pairs:
            curve(
                ax,
                ("selective", 0.4),
                "env/all/bargained_ratio/" + ds,
                color,
                label,
                ls,
            )
        style._style(ax, r"Bargained Ratio ($\rho$)")
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color=gray, lw=0.6)
        ax.legend(
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            fontsize=7,
        )
        if name.endswith("multi"):
            ax.set_ylim(0, 1)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(style.COL_W, 1.8))
    curve(ax, ("selective", 0.4), "env/all/outcome_deal", red, "Deal rate")
    curve(
        ax,
        ("selective", 0.4),
        "env/all/outcome_walk_away",
        slate,
        "Walk-away rate",
        "-.",
    )
    style._style(ax, "Episode Rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="center right", fontsize=7)
    save(fig, "deal_walk_rate")
    # Same reward-design illustration, retaining the original color and line conventions.
    fig, ax = plt.subplots(figsize=(style.COL_W, 2))
    tau = 0.4
    rho = np.linspace(0, 1, 501)
    ax.axvspan(0, tau, color=red, alpha=0.04)
    ax.axhline(0, color=slate, lw=0.6)
    ax.plot(rho, rho, color=blue, lw=1.4, ls="--", label="Surplus")
    ax.plot([0, tau], [-0.5, -0.5], color=red, lw=1.8, label="Threshold")
    ax.plot([tau, 1], [tau, 1], color=red, lw=1.8)
    ax.plot(tau, -0.5, "o", color=red, ms=4, mfc="white", mew=1.2)
    ax.plot(tau, tau, "o", color=red, ms=4)
    ax.axvline(tau, color=red, ls=":", lw=0.7, alpha=0.5)
    ax.text(tau - 0.05, -0.43, r"$\tau=0.4$", color=red, fontsize=8, ha="right")
    style._style(ax, "Terminal Reward", r"Deal Quality $\rho$")
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.65, 1.08))
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    save(fig, "reward_design")
    # Original grouped-bar format; uniform appears as an additional hatched red series.
    dsorder = ["amazon", "casino", "craigslist", "dnd", "ji"]
    ticklabels = ["Amazon", "CaSiNo", "Craigslist", "DnD", "JI"]
    for name, metric, ylabel in [
        ("bar_results", "conditional_br", r"Bargained Ratio ($\rho$)"),
        ("bar_all_episode_utility", "all_episode_br", "All-episode Utility"),
        ("bar_deal_rate", "deal_rate", "Deal Rate (%)"),
    ]:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        x = np.arange(5)
        w = 0.19
        specs = [
            ("base", 0, "base", "Untrained", gray, "//"),
            ("surplus", 0, "main", "Surplus", blue, None),
            ("selective", 0.4, "main", "Selective Threshold", red, None),
            ("uniform", 0.4, "main", "Uniform Threshold", red, "xx"),
        ]
        for j, (method, tau, panel, label, color, hatch) in enumerate(specs):
            samples = [
                [
                    s[metric]
                    for s in analysis_data["dataset_statistics"]
                    if s["panel"] == panel
                    and s["opponent"] == "GPT-5.4"
                    and s["dataset"] == ds
                    and s["method"] == method
                    and s["tau"] == tau
                ]
                for ds in dsorder
            ]
            assert all(len(v) == (1 if method == "base" else 3) for v in samples)
            factor = 100 if metric == "deal_rate" else 1
            m = np.array([np.mean(v) for v in samples]) * factor
            sd = (
                np.array([np.std(v, ddof=1) for v in samples]) * factor
                if method != "base"
                else None
            )
            ax.bar(
                x + (j - 1.5) * w,
                m,
                w,
                label=label,
                color=("white" if method == "uniform" else color),
                edgecolor=color,
                lw=0.6,
                hatch=hatch,
                yerr=sd,
                capsize=1.5,
                error_kw={"elinewidth": 0.55, "ecolor": "#555555"},
            )
            inputs[name + " " + label] = {
                "datasets": dsorder,
                "mean": m.tolist(),
                "sample_sd": sd.tolist() if sd is not None else None,
            }
        ax.set_xticks(x, ticklabels, rotation=20, ha="right")
        style._style(ax, ylabel, None)
        ax.set_ylim(0, 108 if metric == "deal_rate" else 1.05)
        ax.legend(
            frameon=False,
            ncol=2,
            fontsize=6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
        )
        save(fig, name)

    data = {
        "inputs": inputs,
        "training_source_manifests": [
            relative_source(t["manifest"], root) for t in inventory_data["training"]
        ],
        "evaluation_source": str(analysis.relative_to(root))
        if analysis.is_relative_to(root)
        else str(analysis),
        "smoothing": "Centered five-step moving average per seed, edge-padded as in original IJPR plotting code. Mean and sample SD across seeds 42/43/44.",
    }
    (output_dir / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository/data root used to resolve historical source paths",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="Saved training inventory; relative paths resolve under --root",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=DEFAULT_ANALYSIS,
        help="Saved evaluation aggregates; relative paths resolve under --root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Plot output directory; relative paths resolve under --root",
    )
    args = parser.parse_args()
    output = generate(
        args.root, inventory=args.inventory, analysis=args.analysis, output=args.output
    )
    print(f"Generated nine PNG/PDF pairs and figure_data.json in {output}")


if __name__ == "__main__":
    main()
