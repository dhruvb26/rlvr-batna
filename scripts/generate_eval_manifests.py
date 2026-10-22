"""Generate revision evaluation schedules offline, without creating API clients.

Main: AHP 93, CaSiNo 124, DnD/CRA/JI 100 each. Baseline: a balanced
50-scenario subset per dataset, preserving main-panel assignments exactly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from scripts.audit_splits import PATHS, audit, has_blocking_issues
from src.envs import get_env
from src.eval.manifest import build_manifest, load_manifest
from src.scenarios import digest

MAIN_COUNTS = {"amazon": 93, "casino": 124, "dnd": 100, "craigslist": 100, "ji": 100}


def subset_manifest(parent, count, *, seed):
    """Retain assignments and record parent episode IDs; balance role/first mover."""
    if parent["repeats"] != 1:
        raise ValueError("Revision subsets currently require one trial per scenario")
    if count % 2 or count > parent["selected_unique_scenarios"]:
        raise ValueError("Subset must have an even size within the parent pool")
    rng = random.Random(digest([seed, parent["dataset"], "baseline-subset-v1"]))
    groups = defaultdict(list)
    for entry in parent["episodes"]:
        groups[(entry["learner_agent_id"], entry["learner_goes_first"])].append(entry)
    roles = sorted({key[0] for key in groups})
    half = count // 2
    if len(roles) == 1:
        quotas = {(roles[0], first): half for first in [False, True]}
    elif len(roles) == 2:
        a, b = roles
        # Each role and each first-mover status receives half of the subset.
        feasible = [
            x
            for x in range(half + 1)
            if x <= len(groups[(a, True)])
            and half - x <= len(groups[(a, False)])
            and half - x <= len(groups[(b, True)])
            and x <= len(groups[(b, False)])
        ]
        if not feasible:
            raise ValueError("Parent cannot support a balanced subset of this size")
        x = min(feasible, key=lambda x: abs(x - half / 2))
        quotas = {
            (a, True): x,
            (a, False): half - x,
            (b, True): half - x,
            (b, False): x,
        }
    else:
        raise ValueError("Expected one fixed role or two varying roles")
    chosen = []
    for key, n in sorted(quotas.items()):
        if n > len(groups[key]):
            raise ValueError("Not enough parent episodes for a balanced subset")
        chosen.extend(rng.sample(groups[key], n))
    entries = [
        dict(e, episode_id=i, parent_episode_id=e["episode_id"])
        for i, e in enumerate(sorted(chosen, key=lambda e: e["episode_id"]))
    ]
    return {
        **parent,
        "requested_unique_scenarios": count,
        "selected_unique_scenarios": count,
        "parent_manifest_sha256": digest(parent),
        "episodes": entries,
    }


def generate(root, *, seed=42, persona_weights=None):
    report = audit(root)
    if has_blocking_issues(report):
        raise ValueError(
            "Split audit failed; repair data before freezing evaluation manifests"
        )
    base = root / "data/evaluation_manifests/revision"
    records = []
    for dataset, count in MAIN_COUNTS.items():
        data_path = PATHS[dataset]["test"]
        env = get_env(dataset)
        scenarios = env.load_scenarios(str(root / data_path))
        main = build_manifest(
            dataset,
            env,
            scenarios,
            seed=seed,
            num_episodes=count,
            repeats=1,
            persona_weights=persona_weights,
        )
        if main["selected_unique_scenarios"] != count:
            raise ValueError(
                f"{dataset}: required {count} unique scenarios are unavailable"
            )
        baseline = subset_manifest(main, 50, seed=seed)
        for panel, manifest in [("main", main), ("baseline", baseline)]:
            path = base / panel / f"{dataset}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            load_manifest(path, dataset, env, scenarios)
            records.append(
                {
                    "panel": panel,
                    "dataset": dataset,
                    "data_path": data_path,
                    "manifest_path": str(path.relative_to(root)),
                    "manifest_sha256": digest(manifest),
                    "episodes": len(manifest["episodes"]),
                }
            )
    index = {
        "schema_version": 1,
        "seed": seed,
        "persona_weights": persona_weights or {"none": 1},
        "panels": records,
        "note": "Schedules only. No inference performed. Join panels by scenario_id and repeat; episode_id is panel-local.",
    }
    (base / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Read persona_weights only; panel counts and seed are explicit",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    result = generate(
        args.root.resolve(), seed=args.seed, persona_weights=cfg.get("persona_weights")
    )
    for panel in ["main", "baseline"]:
        print(
            panel,
            sum(r["episodes"] for r in result["panels"] if r["panel"] == panel),
            "episodes per matchup",
        )
