"""Score saved negotiations on a fixed shared panel; never invoke model APIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..scenarios import digest
from .runner import AggregateMetrics, EpisodeResult

ASSIGNMENTS = ("persona", "learner_agent_id", "opponent_agent_id", "learner_goes_first")


def episode_key(entry):
    sid = entry.get("scenario_id")
    repeat = entry.get("repeat")
    if not isinstance(sid, str) or not sid or type(repeat) is not int or repeat < 0:
        raise ValueError(
            "Shared scoring requires scenario_id and repeat; legacy logs cannot be paired safely"
        )
    return sid, repeat


def indexed(entries):
    result = {}
    for entry in entries:
        key = episode_key(entry)
        if key in result:
            raise ValueError(f"Duplicate scenario/trial: {key}")
        result[key] = entry
    return result


def validate_subset(manifest, comparison):
    if (
        manifest.get("schema_version") != 1
        or comparison.get("schema_version") != 1
        or manifest["dataset"] != comparison["dataset"]
        or manifest["pool_sha256"] != comparison["pool_sha256"]
    ):
        raise ValueError(
            "Shared and run manifests must describe the same dataset and pool"
        )
    scheduled = indexed(manifest["episodes"])
    requested = indexed(comparison["episodes"])
    if not requested:
        raise ValueError("Shared comparison manifest cannot be empty")
    for key, entry in requested.items():
        if key not in scheduled:
            raise ValueError("Shared panel is not a subset of the scheduled run")
        if any(entry.get(k) != scheduled[key].get(k) for k in ASSIGNMENTS):
            raise ValueError("Shared panel role/persona/first-mover assignments differ")
    return requested


def episode_from_dict(raw, *, learner="", opponent=""):
    fields = {k: v for k, v in raw.items() if k in EpisodeResult.__dataclass_fields__}
    fields.setdefault("learner_label", learner)
    fields.setdefault("opponent_label", opponent)
    return EpisodeResult(**fields)


def summarize_shared(episodes, manifest, comparison, metadata):
    requested = validate_subset(manifest, comparison)
    available = indexed(episodes)
    scheduled = indexed(manifest["episodes"])
    mode = metadata.get("economic_regulation")
    for key, raw in available.items():
        if key not in scheduled or any(
            raw.get(k) != scheduled[key].get(k) for k in ASSIGNMENTS
        ):
            raise ValueError("Saved episode does not match its scheduled assignments")
        if raw.get("economic_regulation") != mode:
            raise ValueError(
                "Saved episodes have missing or inconsistent regulation metadata"
            )
    missing = requested.keys() - available.keys()
    summary = {
        k: metadata.get(k)
        for k in [
            "matchup",
            "dataset",
            "learner",
            "opponent",
            "economic_regulation",
            "comparison_group",
        ]
    }
    summary.update(
        panel="shared",
        comparison_manifest_sha256=digest(comparison),
        requested_episodes=len(requested),
        completed_episodes=len(requested) - len(missing),
        status="incomplete" if missing else "complete",
        missing=[{"scenario_id": s, "repeat": r} for s, r in sorted(missing)],
        overall=None,
    )
    # Do not silently compare different surviving subsets after API failures.
    if not missing:
        aggregate = AggregateMetrics()
        for key in requested:
            aggregate.add(
                episode_from_dict(
                    available[key],
                    learner=metadata.get("learner", ""),
                    opponent=metadata.get("opponent", ""),
                )
            )
        summary["overall"] = aggregate.summary()
    return summary


def write_shared_reports(output_dir, summaries):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shared_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    fields = [
        "result",
        "dataset",
        "learner",
        "opponent",
        "economic_regulation",
        "comparison_group",
        "comparison_manifest_sha256",
        "status",
        "requested_episodes",
        "completed_episodes",
        "deal_rate",
        "avg_bargained_ratio",
        "avg_learner_points",
        "regulation_interventions",
        "regulation_episode_rate",
    ]
    with (output_dir / "shared_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, summary in summaries.items():
            row = {k: summary.get(k) for k in fields}
            row["result"] = key
            row.update(
                {
                    k: (summary.get("overall") or {}).get(k)
                    for k in fields
                    if k
                    in [
                        "deal_rate",
                        "avg_bargained_ratio",
                        "avg_learner_points",
                        "regulation_interventions",
                        "regulation_episode_rate",
                    ]
                }
            )
            writer.writerow(row)


def score_shared_logs(log_dirs, manifest_dir, output_dir):
    summaries = {}
    seen = set()
    for log_dir in log_dirs:
        root = Path(log_dir)
        if not root.is_dir():
            raise ValueError(f"Log directory does not exist: {root}")
        for path in sorted(root.rglob("episodes.json")):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            metadata = json.loads((path.parent / "metrics.json").read_text())
            dataset = metadata["dataset"]
            manifest = json.loads((path.parent.parent / "manifest.json").read_text())
            comparison = json.loads(
                (Path(manifest_dir) / f"{dataset}.json").read_text()
            )
            if metadata.get("manifest_sha256") != digest(manifest):
                raise ValueError("Saved run manifest does not match matchup metadata")
            key = str(path.parent)
            summaries[key] = summarize_shared(
                json.loads(path.read_text()), manifest, comparison, metadata
            )
    if not summaries:
        raise ValueError("No saved episodes found")
    write_shared_reports(output_dir, summaries)
    return summaries


def refresh_shared_run(root):
    """Refresh automatic shared reports from the manifests copied into a run."""
    root = Path(root)
    summaries = {}
    for path in sorted(root.glob("*/*/episodes.json")):
        directory = path.parent.parent
        comparison_path = directory / "comparison_manifest.json"
        if not comparison_path.exists():
            continue
        metadata = json.loads((path.parent / "metrics.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        comparison = json.loads(comparison_path.read_text())
        if metadata.get("manifest_sha256") != digest(manifest):
            raise ValueError("Saved run manifest does not match matchup metadata")
        key = f"{metadata['dataset']}/{path.parent.name}"
        summaries[key] = summarize_shared(
            json.loads(path.read_text()), manifest, comparison, metadata
        )
        (path.parent / "shared_metrics.json").write_text(
            json.dumps(summaries[key], indent=2) + "\n"
        )
    if summaries:
        write_shared_reports(root, summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dirs", nargs="+", required=True)
    parser.add_argument(
        "--manifest-dir", default="data/evaluation_manifests/revision/baseline"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    score_shared_logs(args.log_dirs, args.manifest_dir, args.output_dir)
