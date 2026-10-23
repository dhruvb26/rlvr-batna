"""Count logged economic-regulation replacements; no model calls.

Run from the repository root:
    uv run --locked python ijpr-submission/analysis-regulation-2026-10-22/audit_regulation.py

Writes and prints two tables: Training configs and Eval configs.
Counts pool the three training seeds; regulation-on controls use seed42.
Only distinct completed evaluation configs are included; reused controls are not added twice.
Non-price records are validated but excluded from intervention-rate denominators.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

PRICE = {"amazon", "craigslist"}
COUNT_FIELDS = ("episodes", "replacements", "affected_episodes")



def read(path):
    return loads(path.read_bytes())


def condition(method, tau):
    return "Surplus" if method == "surplus" else f"{method.title()} tau={tau:g}"


def counts(episodes):
    return dict(episodes=len(episodes),
                replacements=sum(e["replacements"] for e in episodes),
                affected_episodes=sum(bool(e["replacements"]) for e in episodes))


def mean_utility(episodes):
    return statistics.mean(e["utility"] for e in episodes) if episodes else None


def combine(rows):
    result = {k: sum(r[k] for r in rows) for k in COUNT_FIELDS}
    result["affected_percent"] = 100 * result["affected_episodes"] / result["episodes"]
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("report.md"))
    args = parser.parse_args()
    if not __debug__:
        parser.error("Do not use python -O; audit validation checks must be enabled")
    if args.output.suffix.lower() != ".md":
        parser.error("--output must name a .md file")
    root = args.root.resolve()

    def resolve(value):
        path = Path(value)
        if path.is_absolute() and "/rlvr-batna/" in str(path):
            return root / str(path).split("/rlvr-batna/", 1)[1]
        return path if path.is_absolute() else root / path

    inventory = read(root / "ijpr-submission/inventory-2026-10-04/inventory.json")
    behavior = read(root / "ijpr-submission/analysis-quality-mechanism-2026-10-04/results.json")
    tau02 = read(root / "ijpr-submission/analysis-tau02-2026-10-05/results.json")
    cells = [dict(c, panel="main" if c["tau"] in (0, .4) else "additional_threshold")
             for c in inventory["cells"]]
    cells += [dict(c, method="selective", tau=.2, panel="additional_threshold",
                   opponent="Qwen3.6 native" if c["opponent"] == "Qwen3.6" else c["opponent"],
                   episodes_file=c["episodes"]) for c in tau02["cells"]]
    cells += [c for c in behavior["cells"] if c["panel"] in ("base", "regulation_off", "regulation_on_fresh")]
    assert len(cells) == 516
    manifests = {panel: {p.stem: read(p) for p in (root / f"data/evaluation_manifests/revision/{panel}").glob("*.json")}
                 for panel in ("main", "baseline")}
    episodes, rows, seen_paths = [], [], set()
    for c in cells:
        path = resolve(c["episodes_file"])
        assert path not in seen_paths, path
        seen_paths.add(path)
        saved = read(path)
        metrics = read(resolve(c["metrics"]))
        overall = metrics["overall"]
        panel = "baseline" if c["panel"] in ("regulation_off", "regulation_on_fresh") else "main"
        expected = {(e["scenario_id"], e["repeat"]): e for e in manifests[panel][c["dataset"]]["episodes"]}
        assert len(saved) == len(expected) == overall["total_episodes"] and metrics["failed_episodes"] == 0, path
        assert {(e["scenario_id"], e["repeat"]) for e in saved} == set(expected), path
        cell_episodes = []
        for e in saved:
            events = e["regulation_interventions"]  # Missing metadata must fail, never count as zero.
            assert isinstance(events, list) and e["economic_regulation"] is (c["panel"] != "regulation_off"), path
            assignment = expected[e["scenario_id"], e["repeat"]]
            assert all(e[k] == assignment[k] for k in ("learner_agent_id", "opponent_agent_id", "learner_goes_first", "persona")), path
            assert all(event["opponent_utility"] < 0 and event.get("original_response") and event.get("replacement_response") for event in events), path
            cell_episodes.append(dict(panel=c["panel"], dataset=c["dataset"], opponent=c["opponent"],
                                      condition=condition(c["method"], c["tau"]), seed=c["seed"],
                                      scenario_id=e["scenario_id"], repeat=e["repeat"], replacements=len(events),
                                      utility=max(-1., min(1., e["bargained_ratio"])) if e["outcome"] == "deal" else 0.))
        total = counts(cell_episodes)
        assert overall["regulation_logged_episodes"] == total["episodes"]
        assert overall["regulation_interventions"] == total["replacements"]
        assert abs(overall["regulation_episode_rate"] - total["affected_episodes"] / total["episodes"]) < .000051
        episodes.extend(cell_episodes)
        if c["dataset"] in PRICE:
            rows.append(dict(phase="evaluation", panel=c["panel"], dataset=c["dataset"], opponent=c["opponent"],
                             condition=condition(c["method"], c["tau"]), seed=c["seed"], **total,
                             full_utility=mean_utility(cell_episodes), source=str(path.relative_to(root))))
    assert len(episodes) == 52500
    primary = [e for e in episodes if e["panel"] == "main"]
    assert len(primary) == 41877
    assert not any(e["replacements"] for e in episodes if e["dataset"] not in PRICE)

    # Exclude the same scenario/repeat identities for every condition and seed.
    removed, groups = defaultdict(set), defaultdict(list)
    for e in primary:
        if e["dataset"] in PRICE:
            if e["replacements"]:
                removed[e["dataset"], e["opponent"]].add((e["scenario_id"], e["repeat"]))
            groups[e["dataset"], e["opponent"], e["condition"], e["seed"]].append(e)
    for row in rows:
        if row["panel"] == "main":
            group = groups[row["dataset"], row["opponent"], row["condition"], row["seed"]]
            clean = [e for e in group if (e["scenario_id"], e["repeat"]) not in removed[e["dataset"], e["opponent"]]]
            row.update(common_clean_episodes=len(clean), common_clean_utility=mean_utility(clean))

    # E5 ON reuses 400 main controls and adds 200 seed42 episodes.
    off = [e for e in episodes if e["panel"] == "regulation_off"]
    off_keys = {(e["condition"], e["seed"], e["dataset"], e["scenario_id"], e["repeat"]) for e in off}
    matched_on = [e for e in episodes if
                  ((e["panel"] == "main" and e["seed"] in (43, 44) and e["condition"] in ("Surplus", "Selective tau=0.4") and e["opponent"] == "GPT-5.4") or e["panel"] == "regulation_on_fresh")
                  and (e["condition"], e["seed"], e["dataset"], e["scenario_id"], e["repeat"]) in off_keys]
    assert len(off) == len(matched_on) == 600
    for mode, selected in (("on", matched_on), ("off", off)):
        grouped = defaultdict(list)
        for e in selected:
            grouped[e["dataset"], e["condition"], e["seed"]].append(e)
        for (ds, treatment, seed), group in sorted(grouped.items()):
            rows.append(dict(phase="evaluation", panel=f"e5_matched_{mode}", dataset=ds,
                             opponent="GPT-5.4", condition=treatment, seed=seed, **counts(group),
                             full_utility=mean_utility(group)))

    training = list(inventory["training"]) + [dict(t, method="selective", tau=.2, run=str(resolve(t["manifest"]).parent)) for t in tau02["training"]]
    assert len(training) == len({resolve(t["run"]) for t in training}) == 15
    full_training_episodes = 0
    for index, t in enumerate(training, 1):
        run = resolve(t["run"])
        cfg = read(run / "training_manifest.json")["config"]
        assert cfg["economic_regulation"] is True and cfg["max_steps"] == 80
        assert cfg["batch_size"] == 64 and cfg["group_size"] == 8 and t["checkpoint_record"]["batch"] == 80
        files = sorted(run.glob("iteration_*/train_rollout_summaries.jsonl"))
        assert [int(p.parent.name.split("_")[-1]) for p in files] == list(range(80)), run
        totals = defaultdict(lambda: dict.fromkeys(COUNT_FIELDS, 0))
        for path in files:
            iteration = int(path.parent.name.split("_")[-1])
            seen = set()
            with path.open("rb") as file:
                for raw in file:
                    if not raw.strip():
                        continue
                    record = loads(raw)
                    key = record["group_idx"], record["traj_idx"]
                    assert record["iteration"] == iteration and record["split"] == "train" and key not in seen, path
                    seen.add(key)
                    assert record["steps"][-1]["episode_done"] is True, path
                    log = loads(record["steps"][-1]["logs"]["negotiation"])
                    events = log["regulation_interventions"]
                    assert isinstance(events, list) and log["economic_regulation"] is True, path
                    assert log["dataset"] in ("amazon", "craigslist", "casino", "dnd"), path
                    assert all(event["opponent_utility"] < 0 and event.get("original_response") and event.get("replacement_response") for event in events), path
                    assert log["dataset"] in PRICE or not events, path
                    counter = totals[log["dataset"]]
                    counter["episodes"] += 1
                    counter["replacements"] += len(events)
                    counter["affected_episodes"] += bool(events)
            assert seen == {(g, j) for g in range(64) for j in range(8)}, path
        assert sum(r["episodes"] for r in totals.values()) == 40960
        full_training_episodes += 40960
        for ds, total in sorted(totals.items()):
            if ds in PRICE:
                rows.append(dict(phase="training", panel="retained_training", dataset=ds,
                                 opponent="Qwen3.6 native", condition=condition(t["method"], t["tau"]),
                                 seed=t["seed"], **total, source=str(run.relative_to(root))))
    assert full_training_episodes == 614400

    # Pooled seed rows reproduce the requested dataset/opponent/condition table.
    pooled = defaultdict(list)
    for row in rows:
        pooled[tuple(row[k] for k in ("phase", "panel", "dataset", "opponent", "condition"))].append(row)
    for key, group in sorted(pooled.items()):
        row = dict(zip(("phase", "panel", "dataset", "opponent", "condition"), key), seed="all", **combine(group))
        for field in ("full_utility", "common_clean_utility"):
            values = [r.get(field) for r in group]
            if all(v is not None for v in values):
                row[field] = statistics.mean(values)
        if all("common_clean_episodes" in r for r in group):
            row["common_clean_episodes"] = sum(r["common_clean_episodes"] for r in group)
        rows.append(row)
    for row in rows:
        row["affected_percent"] = 100 * row["affected_episodes"] / row["episodes"]
    price_main = [r for r in rows if r["panel"] == "main" and r["seed"] == "all"]
    macro = []
    for opponent in sorted({r["opponent"] for r in price_main}):
        for treatment in sorted({r["condition"] for r in price_main}):
            selected = [r for r in price_main if r["opponent"] == opponent and r["condition"] == treatment]
            assert len(selected) == 2
            macro.append(dict(phase="evaluation", panel="main_price_macro", dataset="amazon+craigslist",
                              opponent=opponent, condition=treatment, seed="all", **combine(selected),
                              common_clean_episodes=sum(r["common_clean_episodes"] for r in selected),
                              full_utility=statistics.mean(r["full_utility"] for r in selected),
                              common_clean_utility=statistics.mean(r["common_clean_utility"] for r in selected)))
    rows += macro

    # Publish one Markdown report after all raw-record validation has succeeded.
    report = report_markdown(rows)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".regulation-", delete=False) as file:
            temp = Path(file.name)
            file.write(report)
        os.replace(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)

    print(report, end="")


DATASET_NAMES = {"amazon": "Amazon", "craigslist": "Craigslist"}
OPPONENT_NAMES = {
    "GPT-5.4": "GPT-5.4",
    "GPT-5.4-mini": "GPT-5.4-mini",
    "Qwen3.6 native": "Qwen3.6-35B-A3B (Tinker)",
    "qwen3-30b-a3b-instruct-2507": "Qwen3-30B Instruct",
    "qwen3-30b-a3b-thinking-2507": "Qwen3-30B Thinking",
    "qwen3-235b-a22b-2507": "Qwen3-235B Instruct",
    "qwen3-235b-a22b-thinking-2507": "Qwen3-235B Thinking",
    "llama-4-maverick": "Llama 4 Maverick",
    "kimi-k2-thinking": "Kimi K2 Thinking",
}
CONDITION_NAMES = {"Surplus": "Surplus", "Selective tau=0.2": "Selective τ=.2",
                   "Selective tau=0.4": "Selective τ=.4", "Uniform tau=0.4": "Uniform τ=.4",
                   "Selective tau=0.6": "Selective τ=.6", "Base tau=0": "Untrained Qwen3.6"}


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]])


def report_markdown(rows):
    training = [r for r in rows if r["seed"] == "all" and r["panel"] == "retained_training"]
    evaluation = [r for r in rows if r["seed"] == "all" and r["panel"] in
                  ("main", "additional_threshold", "base", "regulation_off", "regulation_on_fresh")]
    opponents = list(OPPONENT_NAMES)
    conditions = list(CONDITION_NAMES)
    panels = ["retained_training", "main", "additional_threshold", "base", "regulation_off", "regulation_on_fresh"]
    def table(group):
        ordered = sorted(group, key=lambda r: (r["dataset"], opponents.index(r["opponent"]),
                                               panels.index(r["panel"]), conditions.index(r["condition"])))
        values = []
        for row in ordered:
            label = CONDITION_NAMES[row["condition"]]
            if row["panel"] == "regulation_off":
                label += " — regulation Off"
            elif row["panel"] == "regulation_on_fresh":
                label += " — regulation On (seed42)"
            values.append([DATASET_NAMES[row["dataset"]], OPPONENT_NAMES[row["opponent"]], label,
                           f"{row['replacements']:,}", f"{row['episodes']:,}"])
        return markdown_table(["Dataset", "Opponent", "Configuration", "Replacements", "Episodes"], values)
    return "## Training configs\n\n" + table(training) + "\n\n## Eval configs\n\n" + table(evaluation) + "\n"


if __name__ == "__main__":
    main()
