"""Create reproducible disjoint revision splits, preserving original source files.

Run: python -m scripts.repair_craigslist_splits
No network or model calls. Test has priority over validation, then training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.craigslist_data import identity_keys

SPLITS = ("train", "validation", "test")


def repair(rows_by_split):
    records = [
        (split, i, row)
        for split in SPLITS
        for i, row in enumerate(rows_by_split[split])
    ]
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    row_keys = []
    for i, (_, _, row) in enumerate(records):
        keys = identity_keys(row)
        row_keys.append(keys)
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    priority = {s: i for i, s in enumerate(SPLITS)}
    clean = {s: [] for s in SPLITS}
    excluded = []
    keep = set()
    for members in groups.values():
        target = max((records[i][0] for i in members), key=priority.get)
        for i in members:
            split, row_index, row = records[i]
            if split == target:
                keep.add(i)
            else:
                excluded.append(
                    {
                        "source_split": split,
                        "source_row_index": row_index,
                        "dialogue_uuid": row.get("uuid"),
                        "retained_split": target,
                        "reason": "Connected listing/context group occurs in higher-priority split",
                        "identity_keys": row_keys[i],
                        "retained_rows": [
                            {"split": target, "row_index": records[j][1]}
                            for j in members
                            if records[j][0] == target
                        ],
                    }
                )
    for i, (split, _, row) in enumerate(records):
        if i in keep:
            clean[split].append(row)
    excluded.sort(key=lambda e: (e["source_split"], e["source_row_index"]))
    return clean, excluded


def write_repair(root):
    source_dir = root / "data/craigslist"
    output_dir = source_dir / "revision"
    sources = {s: source_dir / f"{s}.json" for s in SPLITS}
    rows = {s: json.loads(p.read_text()) for s, p in sources.items()}
    clean, excluded = repair(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in SPLITS:
        path = output_dir / f"{split}.json"
        # Preserve the test file byte-for-byte, and all retained rows in original order.
        if split == "test":
            path.write_bytes(sources[split].read_bytes())
        else:
            path.write_text(
                json.dumps(clean[split], ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        outputs[split] = {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_rows": len(clean[split]),
            "removed_rows": len(rows[split]) - len(clean[split]),
        }
    report = {
        "schema_version": 1,
        "policy": "test > validation > train; retain source row order; no scenario-UUID grouping",
        "sources": {
            s: {
                "path": str(p.relative_to(root)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "raw_rows": len(rows[s]),
            }
            for s, p in sources.items()
        },
        "outputs": outputs,
        "excluded_rows": excluded,
    }
    (output_dir / "repair-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    result = write_repair(args.root.resolve())
    print(json.dumps(result["outputs"], indent=2))
