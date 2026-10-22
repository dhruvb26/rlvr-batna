from __future__ import annotations

import json
import time
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from ..envs import get_env
from ..scenarios import deserialize_scenario, digest, serialize_scenario
from .display import print_comparison_table, print_matchup_report
from .generator import OPENAI_BASE_URL, make_generator
from .manifest import build_manifest, load_manifest
from .runner import (
    EVAL_PROTOCOL_VERSION as PROTOCOL_VERSION,
)
from .runner import (
    AggregateMetrics,
    EpisodeResult,
    compute_episode_metrics,
    run_episode,
)
from .shared import episode_from_dict, refresh_shared_run, validate_subset

DEFAULT_DATA_PATHS: dict[str, str] = {
    "casino": "data/casino/ca.test.csv",
    "dnd": "data/dnd/dnd.test.csv",
    "amazon": "data/ahp/test",
    "craigslist": "data/craigslist/revision/test.json",
    "ji": "data/ji/ji.test.json",
}


def _resolve_datasets(cfg: dict) -> list[dict]:
    """Resolve the datasets list from config, supporting both single and multi formats.

    Args:
        cfg: Evaluation config dict.

    Returns:
        List of {name, data_path} dicts.
    """
    if "datasets" in cfg:
        resolved = []
        for entry in cfg["datasets"]:
            if isinstance(entry, str):
                entry = {"name": entry}
            name = entry["name"] if isinstance(entry, dict) else entry
            data_path = (
                entry.get("data_path", DEFAULT_DATA_PATHS.get(name))
                if isinstance(entry, dict)
                else DEFAULT_DATA_PATHS.get(name)
            )
            if not data_path:
                raise ValueError(
                    f"No data_path for dataset {name!r} and no default known. "
                    f"Known defaults: {list(DEFAULT_DATA_PATHS)}"
                )
            resolved.append({**entry, "name": name, "data_path": data_path})
        return resolved

    name = cfg.get("dataset", "casino")
    data_path = cfg.get(
        "csv_path", cfg.get("data_path", DEFAULT_DATA_PATHS.get(name, ""))
    )
    return [{"name": name, "data_path": data_path}]


def run_evaluation(cfg: dict) -> dict:
    """Run the full evaluation loop across datasets and matchups.

    Args:
        cfg: Evaluation config dict (from YAML).

    Returns:
        Dict mapping run_key to matchup summary dicts.
    """
    economic_regulation = cfg.get("economic_regulation", True)
    if type(economic_regulation) is not bool:
        raise ValueError("economic_regulation must be a YAML boolean (true/false)")
    dataset_specs = _resolve_datasets(cfg)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        Path(cfg.get("storage_dir", cfg.get("output_dir", "logs/negotiate")))
        / f"run_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    num_episodes = cfg.get("num_episodes", 200)
    max_turns = cfg.get("max_turns", 18)
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.9)
    seed = cfg.get("seed", 42)
    max_concurrent = cfg.get("max_concurrent", 1)

    default_base_url = cfg.get("base_url", OPENAI_BASE_URL)
    default_api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")

    matchups = cfg.get("matchups", [])
    all_results: dict[str, list[EpisodeResult]] = {}
    all_summaries: dict[str, dict] = {}

    for ds_spec in dataset_specs:
        dataset = ds_spec["name"]
        data_path = ds_spec["data_path"]
        env = get_env(dataset)
        scenarios = env.load_scenarios(data_path)
        logger.info(f"Dataset: {dataset} — {len(scenarios)} scenarios from {data_path}")

        if ds_spec.get("manifest_path"):
            manifest = load_manifest(ds_spec["manifest_path"], dataset, env, scenarios)
        else:
            manifest = build_manifest(
                dataset,
                env,
                scenarios,
                seed=seed,
                num_episodes=ds_spec.get("num_episodes", num_episodes),
                repeats=ds_spec.get("repeats", cfg.get("repeats", 1)),
                persona_weights=cfg.get("persona_weights"),
            )
        comparison = None
        if ds_spec.get("comparison_manifest_path"):
            comparison = load_manifest(
                ds_spec["comparison_manifest_path"], dataset, env, scenarios
            )
            validate_subset(manifest, comparison)
        schedule = manifest["episodes"]
        manifest_sha = digest(manifest)
        save_dir = output_dir / dataset
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if comparison is not None:
            (save_dir / "comparison_manifest.json").write_text(
                json.dumps(comparison, indent=2) + "\n"
            )
        if (
            manifest["selected_unique_scenarios"]
            < manifest["requested_unique_scenarios"]
        ):
            logger.warning(
                f"{dataset}: capped selection at {manifest['selected_unique_scenarios']} "
                "unique scenarios; use repeats for explicit repeated trials"
            )

        for matchup in matchups:
            learner_cfg = matchup["learner"]
            opponent_cfg = matchup["opponent"]
            matchup_name = f"{learner_cfg['label']}_vs_{opponent_cfg['label']}"
            run_key = f"{dataset}/{matchup_name}"
            logger.info(f"\n{'#' * 60}\n{run_key}\n{'#' * 60}")

            learner_gen = make_generator(
                learner_cfg, default_base_url, default_api_key_env,
                default_tinker_project_id=cfg.get("tinker_project_id"),
            )
            opponent_gen = make_generator(
                opponent_cfg, default_base_url, default_api_key_env,
                default_tinker_project_id=cfg.get("tinker_project_id"),
            )

            overall = AggregateMetrics()
            per_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
            episodes: list[EpisodeResult] = []
            failed: list[dict] = []

            t0 = time.time()
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"{dataset} | {matchup_name}", total=len(schedule)
                )

                def _run_one(
                    entry: dict,
                    *,
                    scenarios=scenarios,
                    env=env,
                    learner_gen=learner_gen,
                    opponent_gen=opponent_gen,
                    learner_label=learner_cfg["label"],
                    opponent_label=opponent_cfg["label"],
                ) -> EpisodeResult:
                    sc = deepcopy(scenarios[entry["scenario_index"]])
                    return run_episode(
                        env=env,
                        learner_gen=learner_gen,
                        opponent_gen=opponent_gen,
                        scenario=sc,
                        agent_ids=sc["agent_ids"],
                        persona=entry["persona"],
                        episode_id=entry["episode_id"],
                        learner_label=learner_label,
                        opponent_label=opponent_label,
                        max_turns=max_turns,
                        temperature=temperature,
                        top_p=top_p,
                        learner_agent_id=entry["learner_agent_id"],
                        learner_goes_first=entry["learner_goes_first"],
                        scenario_id=entry["scenario_id"],
                        repeat=entry["repeat"],
                        economic_regulation=economic_regulation,
                    )

                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    futures = {
                        executor.submit(_run_one, entry): entry for entry in schedule
                    }
                    for future in as_completed(futures):
                        entry = futures[future]
                        idx = entry["episode_id"]
                        sc = scenarios[entry["scenario_index"]]
                        try:
                            ep = future.result()
                        except Exception as e:  # noqa: BLE001 - persist each failed episode for retry
                            logger.warning(f"Episode {idx} failed: {e}")
                            failed.append(
                                {
                                    **entry,
                                    "scenario": serialize_scenario(dataset, sc),
                                    "economic_regulation": economic_regulation,
                                    "error": str(e),
                                }
                            )
                            progress.advance(task)
                            continue
                        compute_episode_metrics(ep, env, sc)
                        episodes.append(ep)
                        overall.add(ep)
                        per_persona[ep.persona].add(ep)
                        progress.advance(task)

            episodes.sort(key=lambda ep: ep.episode_id)
            failed.sort(key=lambda entry: entry["episode_id"])
            elapsed = time.time() - t0
            logger.info(f"Finished {run_key} in {elapsed:.1f}s")
            if failed:
                logger.warning(f"{len(failed)} episodes failed for {run_key}")

            per_persona_summaries = {
                p: m.summary() for p, m in sorted(per_persona.items())
            }
            matchup_summary = {
                "matchup": matchup_name,
                "dataset": dataset,
                "learner": learner_cfg["label"],
                "opponent": opponent_cfg["label"],
                "manifest_sha256": manifest_sha,
                "protocol_version": PROTOCOL_VERSION,
                "economic_regulation": economic_regulation,
                "comparison_group": digest(
                    {
                        "opponent": opponent_cfg,
                        "protocol_version": PROTOCOL_VERSION,
                        "economic_regulation": economic_regulation,
                        "max_turns": max_turns,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                ),
                "evaluation_settings": {
                    "max_turns": max_turns,
                    "temperature": temperature,
                    "top_p": top_p,
                    "economic_regulation": economic_regulation,
                },
                "scheduled_episodes": len(schedule),
                "failed_episodes": len(failed),
                "overall": overall.summary(),
                "per_persona": per_persona_summaries,
                "elapsed_seconds": round(elapsed, 1),
            }
            all_summaries[run_key] = matchup_summary
            all_results[run_key] = episodes

            save_dir = output_dir / dataset
            _save_matchup(
                save_dir, matchup_name, matchup_summary, episodes, dataset, failed
            )
            print_matchup_report(matchup_summary)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    _save_comparison_csv(output_dir / "comparison.csv", all_summaries)
    print_comparison_table(all_summaries)
    refresh_shared_run(output_dir)
    logger.info(f"Results saved to {output_dir}")
    return all_summaries


def _serialize_episodes(episodes: list[EpisodeResult]) -> list[dict]:
    """Convert EpisodeResult objects to JSON-serializable dicts."""
    return [
        {
            "episode_id": ep.episode_id,
            "scenario_id": ep.scenario_id,
            "repeat": ep.repeat,
            "economic_regulation": ep.economic_regulation,
            "regulation_interventions": ep.regulation_interventions,
            "protocol_version": ep.protocol_version,
            "format_failures": ep.format_failures,
            "repeated_offer_events": ep.repeated_offer_events,
            "persona": ep.persona,
            "outcome": ep.outcome,
            "learner_points": ep.learner_points,
            "opponent_points": ep.opponent_points,
            "num_turns": ep.num_turns,
            "who_terminated": ep.who_terminated,
            "learner_agent_id": ep.learner_agent_id,
            "opponent_agent_id": ep.opponent_agent_id,
            "learner_goes_first": ep.learner_goes_first,
            "learner_format_ok": ep.learner_format_ok,
            "learner_total_turns": ep.learner_total_turns,
            "opponent_format_ok": ep.opponent_format_ok,
            "opponent_total_turns": ep.opponent_total_turns,
            "bargained_ratio": ep.bargained_ratio,
            "first_bid_ratio": ep.first_bid_ratio,
            "pareto_efficient": ep.pareto_efficient,
            "joint_surplus_norm": ep.joint_surplus_norm,
            "learner_messages": ep.learner_messages,
            "opponent_messages": ep.opponent_messages,
        }
        for ep in episodes
    ]


def _save_matchup(
    output_dir: Path,
    matchup_name: str,
    matchup_summary: dict,
    episodes: list[EpisodeResult],
    dataset: str = "",
    failed: list[dict] | None = None,
) -> None:
    """Save matchup results (metrics JSON, episodes JSON, HTML report).

    Args:
        output_dir: Parent directory for this dataset's results.
        matchup_name: Name of the matchup (used as subdirectory).
        matchup_summary: Summary dict to write as metrics.json.
        episodes: List of EpisodeResult objects.
        dataset: Dataset name for HTML title.
        failed: List of failed episode dicts to save for retry.
    """
    matchup_dir = output_dir / matchup_name
    matchup_dir.mkdir(parents=True, exist_ok=True)

    with open(matchup_dir / "metrics.json", "w") as f:
        json.dump(matchup_summary, f, indent=2)

    serializable_episodes = _serialize_episodes(episodes)
    with open(matchup_dir / "episodes.json", "w") as f:
        json.dump(serializable_episodes, f, indent=2)

    if failed:
        with open(matchup_dir / "failed.json", "w") as f:
            json.dump(failed, f, indent=2, default=str)
        logger.warning(f"{len(failed)} episodes saved to {matchup_dir / 'failed.json'}")

    from .render import render_episodes_html

    html_content = render_episodes_html(
        serializable_episodes, title=matchup_name, dataset=dataset
    )
    with open(matchup_dir / "episodes.html", "w") as f:
        f.write(html_content)


def score_negotiate_logs(log_dir: str) -> dict:
    """Re-score saved negotiation episodes from a log directory.

    Saves summary.json and comparison.csv alongside the logs, then prints
    to terminal.

    Args:
        log_dir: Path to directory containing matchup subdirs with episodes.json.

    Returns:
        Dict mapping matchup names to summary dicts.
    """
    root = Path(log_dir)
    if not root.is_dir():
        logger.error(f"Directory not found: {root}")
        return {}

    episode_files = sorted(root.rglob("episodes.json"))
    if not episode_files:
        logger.warning(f"No episodes.json files found under {root}")
        return {}

    all_summaries: dict[str, dict] = {}

    for ep_path in episode_files:
        matchup_name = ep_path.parent.name
        dataset_name = ep_path.parent.parent.name
        with open(ep_path) as f:
            raw_episodes = json.load(f)

        dataset = None
        saved_metadata = {}
        metrics_path = ep_path.parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                saved_metadata = json.load(f)
                dataset = saved_metadata.get("dataset")
        if not dataset:
            dataset = dataset_name

        overall = AggregateMetrics()
        per_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)

        for raw in raw_episodes:
            ep = episode_from_dict(
                raw,
                learner=saved_metadata.get("learner", ""),
                opponent=saved_metadata.get("opponent", ""),
            )
            overall.add(ep)
            per_persona[ep.persona].add(ep)

        per_persona_summaries = {p: m.summary() for p, m in sorted(per_persona.items())}
        matchup_summary = {
            **saved_metadata,
            "matchup": matchup_name,
            "dataset": dataset,
            "overall": overall.summary(),
            "per_persona": per_persona_summaries,
        }
        run_key = f"{dataset}/{matchup_name}"
        all_summaries[run_key] = matchup_summary
        print_matchup_report(matchup_summary)

    print_comparison_table(all_summaries)

    # Save outputs
    with open(root / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    _save_comparison_csv(root / "comparison.csv", all_summaries)
    refresh_shared_run(root)
    logger.info(f"Saved summary.json and comparison.csv to {root}")

    return all_summaries


def _save_comparison_csv(path: Path, all_summaries: dict) -> None:
    """Write the comparison table as a CSV file."""
    import csv

    fields = [
        "matchup",
        "dataset",
        "economic_regulation",
        "deal_rate",
        "bargained_ratio",
        "learner_pts",
        "opponent_pts",
        "pareto_rate",
        "joint_surplus",
        "first_bid_ratio",
        "turns",
        "format_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, s in all_summaries.items():
            o = s.get("overall", {})
            writer.writerow(
                {
                    "matchup": name,
                    "dataset": s.get("dataset", ""),
                    "economic_regulation": s.get("economic_regulation"),
                    "deal_rate": o.get("deal_rate"),
                    "bargained_ratio": o.get("avg_bargained_ratio"),
                    "learner_pts": o.get("avg_learner_points"),
                    "opponent_pts": o.get("avg_opponent_points"),
                    "pareto_rate": o.get("pareto_rate"),
                    "joint_surplus": o.get("avg_joint_surplus_norm"),
                    "first_bid_ratio": o.get("avg_first_bid_ratio"),
                    "turns": o.get("avg_turns_all"),
                    "format_rate": o.get("learner_format_rate"),
                }
            )


def load_eval_config(config_path: str) -> dict:
    """Load an evaluation config from YAML or TOML."""
    path = Path(config_path)
    if path.suffix.lower() == ".toml":
        return tomllib.loads(path.read_text())
    return yaml.safe_load(path.read_text())


def retry_failed(log_dir: str, config_path: str = "configs/eval.yaml") -> None:
    """Retry failed episodes in a log directory and merge results back.

    Finds all failed.json files under log_dir, re-runs the failed episodes using
    model configs from the eval config (matched by learner/opponent labels from
    sibling metrics.json), and merges successful results into episodes.json.

    Args:
        log_dir: Path to the run directory containing matchup subdirs.
        config_path: Path to the eval YAML or TOML config (for model/matchup settings).
    """
    root = Path(log_dir)
    if not root.is_dir():
        logger.error(f"Directory not found: {root}")
        return

    cfg = load_eval_config(config_path)

    max_turns = cfg.get("max_turns", 18)
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.9)
    max_concurrent = cfg.get("max_concurrent", 1)
    default_base_url = cfg.get("base_url", OPENAI_BASE_URL)
    default_api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")

    matchup_cfgs = {
        f"{m['learner']['label']}_vs_{m['opponent']['label']}": m
        for m in cfg.get("matchups", [])
    }

    failed_files = sorted(root.rglob("failed.json"))
    if not failed_files:
        logger.info(f"No failed.json files found under {root}")
        return

    for failed_path in failed_files:
        matchup_dir = failed_path.parent
        matchup_name = matchup_dir.name

        metrics_path = matchup_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"No metrics.json in {matchup_dir}, skipping")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        if metrics.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                "Cannot retry evaluations from a different negotiation protocol. "
                "Start a fresh run; saved-result scoring remains available."
            )
        settings = metrics.get("evaluation_settings", {})
        original_regulation = metrics.get("economic_regulation", True)
        if cfg.get("economic_regulation", True) != original_regulation:
            logger.warning(
                "Retry preserves the original economic regulation setting; use a fresh run to change regimes"
            )
        if type(original_regulation) is not bool:
            raise ValueError("Invalid saved economic regulation setting")
        dataset = metrics.get("dataset", "")
        learner_label = metrics.get("learner", "")
        opponent_label = metrics.get("opponent", "")
        lookup_key = f"{learner_label}_vs_{opponent_label}"

        if lookup_key not in matchup_cfgs:
            logger.warning(
                f"No matchup config for {lookup_key} in {config_path}, skipping"
            )
            continue

        matchup = matchup_cfgs[lookup_key]
        learner_gen = make_generator(
            matchup["learner"], default_base_url, default_api_key_env,
            default_tinker_project_id=cfg.get("tinker_project_id"),
        )
        opponent_gen = make_generator(
            matchup["opponent"], default_base_url, default_api_key_env,
            default_tinker_project_id=cfg.get("tinker_project_id"),
        )

        env = get_env(dataset)

        with open(failed_path) as f:
            failed_entries = json.load(f)

        if any(
            "learner_goes_first" not in e or "learner_agent_id" not in e
            for e in failed_entries
        ):
            logger.warning(
                "Legacy failures lack role/first-mover metadata; retries cannot guarantee pairing"
            )
        logger.info(f"Retrying {len(failed_entries)} failed episodes in {matchup_dir}")

        succeeded: list[EpisodeResult] = []
        still_failed: list[dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"retry {matchup_name}", total=len(failed_entries))

            def _run_one(
                entry: dict,
                *,
                dataset=dataset,
                env=env,
                learner_gen=learner_gen,
                opponent_gen=opponent_gen,
                learner_label=learner_label,
                opponent_label=opponent_label,
                settings=settings,
                original_regulation=original_regulation,
            ) -> EpisodeResult:
                if (
                    entry.get("economic_regulation", original_regulation)
                    != original_regulation
                ):
                    raise ValueError(
                        "Failure metadata has inconsistent regulation setting"
                    )
                sc = deserialize_scenario(dataset, entry["scenario"])
                return run_episode(
                    env=env,
                    learner_gen=learner_gen,
                    opponent_gen=opponent_gen,
                    scenario=sc,
                    agent_ids=sc["agent_ids"],
                    persona=entry["persona"],
                    episode_id=entry["episode_id"],
                    learner_label=learner_label,
                    opponent_label=opponent_label,
                    max_turns=settings.get("max_turns", max_turns),
                    temperature=settings.get("temperature", temperature),
                    top_p=settings.get("top_p", top_p),
                    learner_agent_id=entry.get("learner_agent_id"),
                    learner_goes_first=entry.get("learner_goes_first"),
                    scenario_id=entry.get("scenario_id", ""),
                    repeat=entry.get("repeat", 0),
                    economic_regulation=original_regulation,
                )

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {
                    executor.submit(_run_one, entry): entry for entry in failed_entries
                }
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        ep = future.result()
                    except Exception as e:  # noqa: BLE001 - persist each failed episode for retry
                        logger.warning(
                            f"Episode {entry['episode_id']} failed again: {e}"
                        )
                        entry["error"] = str(e)
                        still_failed.append(entry)
                        progress.advance(task)
                        continue
                    compute_episode_metrics(
                        ep, env, deserialize_scenario(dataset, entry["scenario"])
                    )
                    succeeded.append(ep)
                    progress.advance(task)

        if not succeeded:
            logger.warning(f"All retries failed for {matchup_dir}")
            if still_failed:
                with open(failed_path, "w") as f:
                    json.dump(still_failed, f, indent=2)
            continue

        # Merge succeeded episodes into existing episodes.json
        episodes_path = matchup_dir / "episodes.json"
        existing: list[dict] = []
        if episodes_path.exists():
            with open(episodes_path) as f:
                existing = json.load(f)

        existing.extend(_serialize_episodes(succeeded))
        existing.sort(key=lambda ep: ep["episode_id"])
        with open(episodes_path, "w") as f:
            json.dump(existing, f, indent=2)

        # Update or remove failed.json
        if still_failed:
            with open(failed_path, "w") as f:
                json.dump(still_failed, f, indent=2)
            logger.info(
                f"{len(succeeded)} retried OK, {len(still_failed)} still failed"
            )
        else:
            failed_path.unlink()
            logger.info(f"All {len(succeeded)} retries succeeded, removed failed.json")

        overall = AggregateMetrics()
        by_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
        for raw in existing:
            fields = {
                k: v for k, v in raw.items() if k in EpisodeResult.__dataclass_fields__
            }
            fields.setdefault("learner_label", learner_label)
            fields.setdefault("opponent_label", opponent_label)
            episode = EpisodeResult(**fields)
            overall.add(episode)
            by_persona[episode.persona].add(episode)
        metrics.update(
            overall=overall.summary(),
            failed_episodes=len(still_failed),
            per_persona={p: a.summary() for p, a in sorted(by_persona.items())},
        )
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        summary_path = root / "summary.json"
        if summary_path.exists():
            summaries = json.loads(summary_path.read_text())
            summaries[f"{dataset}/{matchup_name}"] = metrics
            summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
            _save_comparison_csv(root / "comparison.csv", summaries)

        # Regenerate HTML
        from .render import render_episodes_html

        html_content = render_episodes_html(
            existing, title=matchup_name, dataset=dataset
        )
        with open(matchup_dir / "episodes.html", "w") as f:
            f.write(html_content)

    refresh_shared_run(root)


def main(
    config_path: str = "configs/eval.yaml",
    evaluate_only: str | None = None,
    retry: str | None = None,
):
    """CLI entry point for negotiation evaluation.

    Args:
        config_path: Path to YAML or TOML evaluation config.
        evaluate_only: If set, re-score existing logs in this directory instead.
        retry: If set, retry failed episodes in this log directory.
    """
    if retry:
        retry_failed(retry, config_path)
        return
    if evaluate_only:
        score_negotiate_logs(evaluate_only)
        return
    cfg = load_eval_config(config_path)
    run_evaluation(cfg)
