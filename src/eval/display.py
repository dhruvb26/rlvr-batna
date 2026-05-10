from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _fmt(val, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a numeric value for display, returning em-dash for None."""
    if val is None:
        return "\u2014"
    if "%" in fmt:
        return f"{val:{fmt}}"
    return f"{val:{fmt}}{suffix}"


def print_matchup_report(summary: dict) -> None:
    """Print a rich panel summarizing a single matchup evaluation.

    Args:
        summary: Dict with 'overall' metrics and optional 'per_persona' breakdown.
    """
    o = summary.get("overall", {})

    lines = Text()
    lines.append(f"Episodes:          {o.get('total_episodes', 0)}\n")
    lines.append(
        f"Deal rate:         {_fmt(o.get('deal_rate'), '.1%')}  ({o.get('deal_count', 0)})\n"
    )
    lines.append(
        f"Walk-away rate:    {_fmt(o.get('walk_away_rate'), '.1%')}  ({o.get('walk_away_count', 0)})\n"
    )
    lines.append(
        f"Reject-loop rate:  {_fmt(o.get('reject_loop_rate'), '.1%')}  ({o.get('reject_loop_count', 0)})\n"
    )
    lines.append(
        f"Max-turns rate:    {_fmt(o.get('max_turns_rate'), '.1%')}  ({o.get('max_turns_count', 0)})\n"
    )
    lines.append("\n")
    lines.append(f"Bargained ratio:   {_fmt(o.get('avg_bargained_ratio'), '.3f')}\n")
    lines.append(
        f"Avg learner pts:   {_fmt(o.get('avg_learner_points'))}  (std {_fmt(o.get('std_learner_points'))})\n"
    )
    lines.append(f"Avg opponent pts:  {_fmt(o.get('avg_opponent_points'))}\n")
    lines.append(f"Avg joint score:   {_fmt(o.get('avg_joint_score'))}\n")
    lines.append(f"Avg score ratio:   {_fmt(o.get('avg_score_ratio'), '.3f')}\n")
    if o.get("avg_joint_surplus_norm") is not None:
        lines.append(
            f"Joint surplus:     {_fmt(o.get('avg_joint_surplus_norm'), '.3f')}\n"
        )
    if o.get("pareto_rate") is not None:
        lines.append(f"Pareto rate:       {_fmt(o.get('pareto_rate'), '.1%')}\n")
    if o.get("avg_first_bid_ratio") is not None:
        lines.append(
            f"First bid ratio:   {_fmt(o.get('avg_first_bid_ratio'), '.3f')}\n"
        )
    lines.append(f"Avg turns (deal):  {_fmt(o.get('avg_turns_to_deal'))}\n")
    lines.append(f"Avg turns (all):   {_fmt(o.get('avg_turns_all'))}\n")
    lines.append(f"Points/turn:       {_fmt(o.get('points_per_turn'), '.3f')}\n")
    lines.append("\n")
    lines.append(
        f"Learner format:    {_fmt(o.get('learner_format_rate'), '.1%')}  ({o.get('learner_total_turns', 0)} turns)\n"
    )
    lines.append(
        f"Opponent format:   {_fmt(o.get('opponent_format_rate'), '.1%')}  ({o.get('opponent_total_turns', 0)} turns)\n"
    )

    title = summary.get("matchup", "")
    if summary.get("dataset"):
        title = f"{title}  [{summary['dataset']}]"
    panel = Panel(lines, title=title, border_style="blue", padding=(1, 2))
    console.print(panel)

    if summary.get("per_persona"):
        table = Table(title="Per-Persona Breakdown", show_lines=False)
        table.add_column("Persona", style="cyan")
        table.add_column("Deal%", justify="right")
        table.add_column("Learner Pts", justify="right")
        table.add_column("Opp Pts", justify="right")
        table.add_column("Ratio", justify="right")
        table.add_column("Joint", justify="right")
        table.add_column("Turns", justify="right")
        table.add_column("Format%", justify="right")
        table.add_column("N", justify="right")

        for persona, pm in summary["per_persona"].items():
            table.add_row(
                persona,
                _fmt(pm.get("deal_rate"), ".0%"),
                _fmt(pm.get("avg_learner_points")),
                _fmt(pm.get("avg_opponent_points")),
                _fmt(pm.get("avg_score_ratio"), ".3f"),
                _fmt(pm.get("avg_joint_score"), ".1f"),
                _fmt(pm.get("avg_turns_to_deal")),
                _fmt(pm.get("learner_format_rate"), ".0%"),
                str(pm.get("total_episodes", 0)),
            )
        console.print(table)
    console.print()


def print_comparison_table(all_summaries: dict) -> None:
    """Print a side-by-side comparison table across multiple matchups.

    Args:
        all_summaries: Dict mapping run_key to matchup summary dicts.
    """
    if len(all_summaries) < 2:
        return

    table = Table(title="A/B Comparison", show_lines=True, border_style="green")
    table.add_column("Matchup", style="bold")
    table.add_column("Dataset", style="dim")
    table.add_column("Deal%", justify="right")
    table.add_column("Bargain", justify="right")
    table.add_column("Learner", justify="right")
    table.add_column("Opp", justify="right")
    table.add_column("Pareto", justify="right")
    table.add_column("Joint S.", justify="right")
    table.add_column("1st Bid", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Format%", justify="right")

    for name, s in all_summaries.items():
        o = s.get("overall", {})
        table.add_row(
            name,
            s.get("dataset") or "\u2014",
            _fmt(o.get("deal_rate"), ".1%"),
            _fmt(o.get("avg_bargained_ratio"), ".3f"),
            _fmt(o.get("avg_learner_points")),
            _fmt(o.get("avg_opponent_points")),
            _fmt(o.get("pareto_rate"), ".1%"),
            _fmt(o.get("avg_joint_surplus_norm"), ".3f"),
            _fmt(o.get("avg_first_bid_ratio"), ".3f"),
            _fmt(o.get("avg_turns_to_deal")),
            _fmt(o.get("learner_format_rate"), ".1%"),
        )

    console.print(table)
    console.print()
