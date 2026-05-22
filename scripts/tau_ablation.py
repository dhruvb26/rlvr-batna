"""Post-hoc τ ablation: reclassify deals from a single trained checkpoint
at varying quality thresholds to show smooth trade-offs."""

import json
import sys
from pathlib import Path

THRESHOLDS = [0.2, 0.4, 0.6]
MULTI_ITEM = {"casino", "dnd", "ji"}

MATCHUP = "qwen3-30b-a3b-instruct-2507-batna_vs_qwen3-30b-a3b-instruct-2507"
DATASETS = ["amazon", "craigslist", "casino", "dnd", "ji"]


def load_episodes(run_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for ds in DATASETS:
        path = run_dir / ds / MATCHUP / "episodes.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            out[ds] = json.load(f)
    return out


def ablate(episodes: dict[str, list[dict]]):
    # Per-dataset, per-threshold stats
    for tau in THRESHOLDS:
        print(f"\n{'=' * 70}")
        print(f"  τ = {tau}")
        print(f"{'=' * 70}")
        print(
            f"{'Dataset':<12} {'Deals':>6} {'Acceptable':>11} {'Eff.Deal%':>10} "
            f"{'Avg BR(acc)':>12} {'Avg BR(all)':>12} {'Walk%':>7}"
        )
        print("-" * 70)

        all_acc_br = []
        all_deal_br = []
        all_total = 0
        all_deals = 0
        all_acceptable = 0
        all_walks = 0

        for ds in DATASETS:
            if ds not in episodes:
                continue
            eps = episodes[ds]
            total = len(eps)
            deals = [
                e
                for e in eps
                if e["outcome"] == "deal" and e["bargained_ratio"] is not None
            ]
            walks = [e for e in eps if e["outcome"] == "walk_away"]

            # τ only applies to multi-item; price deals are always "acceptable"
            if ds in MULTI_ITEM:
                acceptable = [d for d in deals if d["bargained_ratio"] >= tau]
            else:
                acceptable = deals

            acc_brs = [d["bargained_ratio"] for d in acceptable]
            deal_brs = [d["bargained_ratio"] for d in deals]

            eff_deal_rate = len(acceptable) / total if total > 0 else 0
            avg_br_acc = sum(acc_brs) / len(acc_brs) if acc_brs else 0
            avg_br_all = sum(deal_brs) / len(deal_brs) if deal_brs else 0
            walk_rate = len(walks) / total if total > 0 else 0

            print(
                f"{ds:<12} {len(deals):>6} {len(acceptable):>11} "
                f"{eff_deal_rate:>9.0%} {avg_br_acc:>12.4f} "
                f"{avg_br_all:>12.4f} {walk_rate:>6.0%}"
            )

            all_acc_br.extend(acc_brs)
            all_deal_br.extend(deal_brs)
            all_total += total
            all_deals += len(deals)
            all_acceptable += len(acceptable)
            all_walks += len(walks)

        print("-" * 70)
        eff = all_acceptable / all_total if all_total > 0 else 0
        avg_acc = sum(all_acc_br) / len(all_acc_br) if all_acc_br else 0
        avg_all = sum(all_deal_br) / len(all_deal_br) if all_deal_br else 0
        walk = all_walks / all_total if all_total > 0 else 0
        print(
            f"{'OVERALL':<12} {all_deals:>6} {all_acceptable:>11} "
            f"{eff:>9.0%} {avg_acc:>12.4f} {avg_all:>12.4f} {walk:>6.0%}"
        )


def main():
    run_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("logs/baseline_qwen/run_20260519_022034")
    )
    episodes = load_episodes(run_dir)
    print(f"Run: {run_dir}")
    print(f"Loaded: {', '.join(f'{k}({len(v)})' for k, v in episodes.items())}")
    ablate(episodes)


if __name__ == "__main__":
    main()
