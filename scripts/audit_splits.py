"""Read-only split audit. Run: python -m scripts.audit_splits --output PATH.

Identity overlap is reported at separate source and loaded-context levels;
repeated preference structures alone do not prove repeated source dialogues.
No split files are rewritten. --strict exits 1 for overlap or failed AHP/DnD invariants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.craigslist_data import listing_content_id
from src.envs import get_env
from src.scenarios import digest, scenario_id

PATHS = {
    "amazon": {"train": "data/ahp/train", "test": "data/ahp/test"},
    "casino": {"train": "data/casino/ca.train.csv", "test": "data/casino/ca.test.csv"},
    "dnd": {"train": "data/dnd/dnd.train.csv", "test": "data/dnd/dnd.test.csv"},
    "craigslist": {
        s: f"data/craigslist/revision/{s}.json" for s in ["train", "validation", "test"]
    },
    "ji": {"train": "data/ji/ji.train.json", "test": "data/ji/ji.test.json"},
}


def asin(product):
    for key in ["amazon_link", "link"]:
        match = re.search(
            r"/(?:dp|product)/([A-Z0-9]{10})(?:[/?]|$)",
            product.get(key, ""),
            re.IGNORECASE,
        )
        if match:
            return match[1].upper()
    return None


def read_rows(path):
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    rows = []
    for file in files:
        if file.suffix == ".csv":
            with file.open() as f:
                rows.extend(csv.DictReader(f))
        else:
            rows.extend(json.loads(file.read_text()))
    return rows, {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def dnd_pair_key(row):
    sides = [json.loads(row["input"]), json.loads(row["partner_input"])]
    return digest(sorted(sides, key=lambda v: json.dumps(v, sort_keys=True)))


def loaded_dnd_pair_key(scenario):
    items = ("book", "hat", "ball")
    sides = [
        {
            "count": [scenario["counts"][item] for item in items],
            "value": [scenario["agents"][agent]["value"][item] for item in items],
        }
        for agent in scenario["agent_ids"]
    ]
    return digest(sorted(sides, key=lambda v: json.dumps(v, sort_keys=True)))


def invalid_dnd_pairs(rows):
    return [
        i
        for i in range(0, len(rows), 2)
        if i + 1 == len(rows) or dnd_pair_key(rows[i]) != dnd_pair_key(rows[i + 1])
    ]


def source_levels(dataset, rows):
    if dataset == "amazon":
        return {"product_asin": [asin(r) for r in rows]}
    if dataset == "craigslist":
        return {
            "dialogue_uuid": [r.get("uuid") for r in rows],
            "scenario_uuid": [r["scenario"].get("uuid") for r in rows],
            "listing_content": [listing_content_id(r) for r in rows],
            "listing_post_id": [
                str(r["scenario"]["post_id"])
                if r["scenario"].get("post_id") is not None
                else None
                for r in rows
            ],
        }
    if dataset == "ji":
        return {"negotiation_id": [r.get("id") for r in rows]}
    if dataset == "casino":
        return {"dialogue_content": [digest(json.loads(r["chat_logs"])) for r in rows]}

    # DnD has two seat views per dialogue; normalize the speaker swap.
    def dialogue(r):
        text = r["dialogue"]
        flipped = (
            text.replace("YOU:", "_SELF_:")
            .replace("THEM:", "YOU:")
            .replace("_SELF_:", "THEM:")
        )
        return digest(min(text, flipped))

    return {
        "dialogue_text_ignoring_seat": [dialogue(r) for r in rows],
        "raw_allocation_context": [dnd_pair_key(r) for r in rows],
    }


def audit(root, *, original_craigslist=False):
    report = {
        "schema_version": 1,
        "datasets": {},
        "notes": [
            "Read-only audit; no split changes or model calls.",
            "Craigslist scenario UUID reuse is diagnostic, not blocking: unrelated listings reuse these IDs.",
            "Counts reflect usable loader outputs as well as raw sources; source overlaps may include filtered rows.",
            "DnD dialogue text matches can be generic identical conversations, not necessarily the same source record.",
            "Loaded scenario IDs normalize DnD seat swaps; prompt pairs ignore seat order and metadata not shown to agents.",
            "JI train/test files are audited for provenance only; this does not authorize JI training.",
        ],
    }
    for dataset, splits in PATHS.items():
        if dataset == "craigslist" and original_craigslist:
            splits = {s: f"data/craigslist/{s}.json" for s in splits}
        env = get_env(dataset)
        levels_by_split = {}
        summaries = {}
        for split, rel in splits.items():
            path = root / rel
            rows, hashes = read_rows(path)
            scenarios = env.load_scenarios(str(path))
            levels = source_levels(dataset, rows)
            levels["loaded_scenario"] = [scenario_id(dataset, s) for s in scenarios]
            levels["model_prompt_pair"] = [
                digest(sorted(env.build_system_prompt(s, a) for a in s["agent_ids"]))
                for s in scenarios
            ]
            levels_by_split[split] = levels
            summaries[split] = {
                "raw_rows": len(rows),
                "loaded_scenarios": len(scenarios),
                "files": {str(Path(k).relative_to(root)): v for k, v in hashes.items()},
                "identity_counts": {
                    key: {
                        "unique": len({v for v in vals if v is not None}),
                        "duplicate_rows": sum(
                            n - 1
                            for n in Counter(v for v in vals if v is not None).values()
                        ),
                        "missing": sum(v is None for v in vals),
                    }
                    for key, vals in levels.items()
                },
            }
            if dataset == "dnd":
                invalid = invalid_dnd_pairs(rows)
                # Historical diagnostic only: the loader no longer skips rows.
                summaries[split]["legacy_nonmatching_adjacent_pairs"] = len(invalid)
                all_keys = {dnd_pair_key(row) for row in rows}
                retained_keys = {loaded_dnd_pair_key(s) for s in scenarios}
                summaries[split]["unique_raw_contexts_omitted_by_loader"] = len(
                    all_keys - retained_keys
                )
                summaries[split]["unexpected_loaded_contexts"] = len(
                    retained_keys - all_keys
                )
                summaries[split]["duplicate_loaded_contexts"] = len(scenarios) - len(
                    retained_keys
                )
        overlaps = {}
        for a, b in combinations(splits, 2):
            overlap = {}
            for key in levels_by_split[a]:
                left = Counter(x for x in levels_by_split[a][key] if x is not None)
                right = Counter(x for x in levels_by_split[b][key] if x is not None)
                shared = sorted(left.keys() & right.keys())
                overlap[key] = {
                    "blocking": not (
                        dataset == "craigslist" and key == "scenario_uuid"
                    ),
                    "shared_identities": len(shared),
                    "left_rows": sum(left[k] for k in shared),
                    "right_rows": sum(right[k] for k in shared),
                    "examples": shared[:10],
                }
            overlaps[f"{a}__{b}"] = overlap
        report["datasets"][dataset] = {"splits": summaries, "overlaps": overlaps}
    # Reproduce existing AHP split in memory and verify source coverage.
    from scripts.split_ahp import SEED, TEST_FRACTION

    rng = random.Random(SEED)
    expected = {"train": [], "test": []}
    original = []
    for f in sorted((root / "data/ahp").glob("*.json")):
        rows = json.loads(f.read_text())
        original.extend(rows)
        rng.shuffle(rows)
        n = max(1, round(len(rows) * TEST_FRACTION))
        expected["test"].extend(rows[:n])
        expected["train"].extend(rows[n:])
    actual = {s: read_rows(root / rel)[0] for s, rel in PATHS["amazon"].items()}
    report["ahp_verification"] = {
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "matches_split_script": {s: actual[s] == expected[s] for s in actual},
        "exact_source_coverage": Counter(map(digest, original))
        == Counter(map(digest, actual["train"] + actual["test"])),
        "source_products": len(original),
        "source_unique_asins": len({asin(r) for r in original}),
        "cross_split_asins": sorted(
            {asin(r) for r in actual["train"]}
            & {asin(r) for r in actual["test"]} - {None}
        ),
    }
    return report


def has_blocking_issues(result):
    overlaps = sum(
        v["shared_identities"]
        for ds in result["datasets"].values()
        for pair in ds["overlaps"].values()
        for v in pair.values()
        if v.get("blocking", True)
    )
    ahp = result["ahp_verification"]
    invalid_dnd = any(
        s["unique_raw_contexts_omitted_by_loader"]
        or s["unexpected_loaded_contexts"]
        or s["duplicate_loaded_contexts"]
        for s in result["datasets"]["dnd"]["splits"].values()
    )
    invalid_ahp = not ahp["exact_source_coverage"] or not all(
        ahp["matches_split_script"].values()
    )
    return bool(overlaps or invalid_dnd or invalid_ahp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--original-craigslist", action="store_true")
    args = parser.parse_args()
    result = audit(args.root.resolve(), original_craigslist=args.original_craigslist)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output)
    if args.strict and has_blocking_issues(result):
        raise SystemExit(1)
