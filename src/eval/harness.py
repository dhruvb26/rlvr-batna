from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .display import print_comparison_table, print_matchup_report
from .generator import OPENAI_BASE_URL, make_generator
from .runner import (
    AggregateMetrics,
    EpisodeResult,
    compute_episode_metrics,
    run_episode,
)

DEFAULT_DATA_PATHS: dict[str, str] = {
    "casino": "data/casino/ca.test.csv",
    "dnd": "data/dnd/dnd.test.csv",
    "amazon": "data/ahp",
    "craigslist": "data/craigslist/test.json",
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
            resolved.append({"name": name, "data_path": data_path})
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
    dataset_specs = _resolve_datasets(cfg)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(cfg.get("storage_dir", cfg.get("output_dir", "logs/negotiate")))
        / f"run_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = cfg.get("num_episodes", 200)
    max_turns = cfg.get("max_turns", 18)
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.9)
    seed = cfg.get("seed", 42)
    max_concurrent = cfg.get("max_concurrent", 1)

    pw = cfg.get("persona_weights") or {}
    persona_names = list(pw.keys()) if pw else ["none"]
    persona_weights = list(pw.values()) if pw else [1.0]

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

        random.seed(seed)

        for matchup in matchups:
            learner_cfg = matchup["learner"]
            opponent_cfg = matchup["opponent"]
            matchup_name = f"{learner_cfg['label']}_vs_{opponent_cfg['label']}"
            run_key = f"{dataset}/{matchup_name}"
            logger.info(f"\n{'#' * 60}\n{run_key}\n{'#' * 60}")

            learner_gen = make_generator(
                learner_cfg, default_base_url, default_api_key_env
            )
            opponent_gen = make_generator(
                opponent_cfg, default_base_url, default_api_key_env
            )

            sampled_scenarios = random.choices(scenarios, k=num_episodes)
            sampled_personas = random.choices(
                persona_names, weights=persona_weights, k=num_episodes
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
                    f"{dataset} | {matchup_name}", total=num_episodes
                )

                def _run_one(idx: int, sc: dict, pers: str) -> EpisodeResult:
                    return run_episode(
                        env=env,
                        learner_gen=learner_gen,
                        opponent_gen=opponent_gen,
                        scenario=sc,
                        agent_ids=sc["agent_ids"],
                        persona=pers,
                        episode_id=idx,
                        learner_label=learner_cfg["label"],
                        opponent_label=opponent_cfg["label"],
                        max_turns=max_turns,
                        temperature=temperature,
                        top_p=top_p,
                    )

                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    futures = {
                        executor.submit(_run_one, i, sc, pers): i
                        for i, (sc, pers) in enumerate(
                            zip(sampled_scenarios, sampled_personas)
                        )
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            ep = future.result()
                        except Exception as e:
                            logger.warning(f"Episode {idx} failed: {e}")
                            failed.append(
                                {
                                    "episode_id": idx,
                                    "scenario": sampled_scenarios[idx],
                                    "persona": sampled_personas[idx],
                                    "error": str(e),
                                }
                            )
                            progress.advance(task)
                            continue
                        compute_episode_metrics(ep, env, sampled_scenarios[idx])
                        episodes.append(ep)
                        overall.add(ep)
                        per_persona[ep.persona].add(ep)
                        progress.advance(task)

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
    logger.info(f"Results saved to {output_dir}")
    return all_summaries


def _serialize_episodes(episodes: list[EpisodeResult]) -> list[dict]:
    """Convert EpisodeResult objects to JSON-serializable dicts."""
    return [
        {
            "episode_id": ep.episode_id,
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
        metrics_path = ep_path.parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                dataset = json.load(f).get("dataset")
        if not dataset:
            dataset = dataset_name

        overall = AggregateMetrics()
        per_persona: dict[str, AggregateMetrics] = defaultdict(AggregateMetrics)

        for raw in raw_episodes:
            ep = EpisodeResult(
                episode_id=raw["episode_id"],
                learner_label=raw.get(
                    "learner_label",
                    matchup_name.split("_vs_")[0] if "_vs_" in matchup_name else "",
                ),
                opponent_label=raw.get(
                    "opponent_label",
                    matchup_name.split("_vs_")[-1] if "_vs_" in matchup_name else "",
                ),
                persona=raw["persona"],
                outcome=raw["outcome"],
                learner_points=raw.get("learner_points"),
                opponent_points=raw.get("opponent_points"),
                num_turns=raw["num_turns"],
                learner_messages=raw.get("learner_messages", []),
                opponent_messages=raw.get("opponent_messages", []),
                learner_agent_id=raw.get("learner_agent_id", ""),
                opponent_agent_id=raw.get("opponent_agent_id", ""),
                who_terminated=raw.get("who_terminated", ""),
                learner_goes_first=raw.get("learner_goes_first", True),
                learner_total_turns=raw.get("learner_total_turns", 0),
                learner_format_ok=raw.get("learner_format_ok", 0),
                opponent_total_turns=raw.get("opponent_total_turns", 0),
                opponent_format_ok=raw.get("opponent_format_ok", 0),
                bargained_ratio=raw.get("bargained_ratio"),
                first_bid_ratio=raw.get("first_bid_ratio"),
                pareto_efficient=raw.get("pareto_efficient"),
                joint_surplus_norm=raw.get("joint_surplus_norm"),
            )
            overall.add(ep)
            per_persona[ep.persona].add(ep)

        per_persona_summaries = {p: m.summary() for p, m in sorted(per_persona.items())}
        matchup_summary = {
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
    logger.info(f"Saved summary.json and comparison.csv to {root}")

    return all_summaries


def _save_comparison_csv(path: Path, all_summaries: dict) -> None:
    """Write the comparison table as a CSV file."""
    import csv

    fields = [
        "matchup",
        "dataset",
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


def retry_failed(log_dir: str, config_path: str = "configs/eval.yaml") -> None:
    """Retry failed episodes in a log directory and merge results back.

    Finds all failed.json files under log_dir, re-runs the failed episodes using
    model configs from the eval YAML (matched by learner/opponent labels from
    sibling metrics.json), and merges successful results into episodes.json.

    Args:
        log_dir: Path to the run directory containing matchup subdirs.
        config_path: Path to the eval YAML config (for model/matchup settings).
    """
    root = Path(log_dir)
    if not root.is_dir():
        logger.error(f"Directory not found: {root}")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

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
            matchup["learner"], default_base_url, default_api_key_env
        )
        opponent_gen = make_generator(
            matchup["opponent"], default_base_url, default_api_key_env
        )

        env = get_env(dataset)

        with open(failed_path) as f:
            failed_entries = json.load(f)

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

            def _run_one(entry: dict) -> EpisodeResult:
                sc = entry["scenario"]
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
                )

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {
                    executor.submit(_run_one, entry): entry for entry in failed_entries
                }
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        ep = future.result()
                    except Exception as e:
                        logger.warning(
                            f"Episode {entry['episode_id']} failed again: {e}"
                        )
                        entry["error"] = str(e)
                        still_failed.append(entry)
                        progress.advance(task)
                        continue
                    compute_episode_metrics(ep, env, entry["scenario"])
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

        # Regenerate HTML
        from .render import render_episodes_html

        html_content = render_episodes_html(
            existing, title=matchup_name, dataset=dataset
        )
        with open(matchup_dir / "episodes.html", "w") as f:
            f.write(html_content)


def main(
    config_path: str = "configs/eval.yaml",
    evaluate_only: str | None = None,
    retry: str | None = None,
):
    """CLI entry point for negotiation evaluation.

    Args:
        config_path: Path to YAML evaluation config.
        evaluate_only: If set, re-score existing logs in this directory instead.
        retry: If set, retry failed episodes in this log directory.
    """
    if retry:
        retry_failed(retry, config_path)
        return
    if evaluate_only:
        score_negotiate_logs(evaluate_only)
        return
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_evaluation(cfg)
