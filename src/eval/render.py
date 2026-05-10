from __future__ import annotations

import html
import json
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _extract_reasoning(content: str) -> tuple[str | None, str]:
    """Split content into reasoning block and remaining text.

    Args:
        content: Raw model output potentially containing <think>...</think>.

    Returns:
        Tuple of (reasoning_text or None, remaining_content).
    """
    m = _THINK_RE.search(content)
    if m:
        reasoning = m.group(1).strip()
        rest = content[: m.start()] + content[m.end() :]
        return (reasoning or None), rest.strip()
    return None, content


def _extract_section(text: str, section: str) -> str | None:
    pat = re.compile(
        rf"(?:^|\n)\s*{section}\s*:\s*(.*?)(?=\n\s*(?:Thought|Talk|Action)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else None


def _parse_message(content: str) -> dict:
    """Parse a model output into reasoning/Thought/Talk/Action parts.

    Args:
        content: Raw assistant message content.

    Returns:
        Dict with 'reasoning', 'thought', 'talk', 'action', 'malformed' keys.
    """
    reasoning, rest = _extract_reasoning(content)
    thought = _extract_section(rest, "Thought")
    talk = _extract_section(rest, "Talk")
    action = _extract_section(rest, "Action")
    malformed = thought is None or talk is None or action is None
    return {
        "reasoning": reasoning,
        "thought": thought,
        "talk": talk,
        "action": action,
        "malformed": malformed,
    }


def _esc(text: str | None) -> str:
    if text is None:
        return ""
    return html.escape(text)


def _render_sidebar_item(ep: dict, idx: int) -> str:
    """Render a single sidebar list item for an episode."""
    outcome = ep.get("outcome", "unknown")
    outcome_cls = {
        "deal": "tag-deal",
        "walk_away": "tag-walkaway",
        "reject_loop": "tag-reject",
        "max_turns": "tag-maxturns",
    }.get(outcome, "tag-unknown")

    active = " active" if idx == 0 else ""
    return (
        f'<button class="sb-item{active}" data-idx="{idx}">'
        f'<span class="sb-id">#{ep["episode_id"]}</span>'
        f'<span class="{outcome_cls}">{outcome}</span>'
        f"</button>"
    )


def _render_episode(ep: dict, idx: int) -> str:
    """Render episode content as an <article> for the main viewer area."""
    outcome = ep.get("outcome", "unknown")
    outcome_cls = {
        "deal": "tag-deal",
        "walk_away": "tag-walkaway",
        "reject_loop": "tag-reject",
        "max_turns": "tag-maxturns",
    }.get(outcome, "tag-unknown")

    learner_ok = ep.get("learner_format_ok", 0)
    learner_total = ep.get("learner_total_turns", 0)
    opponent_ok = ep.get("opponent_format_ok", 0)
    opponent_total = ep.get("opponent_total_turns", 0)

    hidden = "" if idx == 0 else ' style="display:none"'
    parts = [f'<article class="episode" data-idx="{idx}"{hidden}>']

    # Header with metadata
    parts.append('<div class="ep-header">')
    parts.append(
        f'<div class="ep-title">'
        f'<span class="ep-num">#{ep["episode_id"]}</span>'
        f'<span class="{outcome_cls}">{outcome}</span>'
        f'<span class="ep-turns">{ep.get("num_turns", 0)} turns</span>'
        f"</div>"
    )
    parts.append('<div class="ep-meta">')
    meta_items = [
        f'<span class="meta-label">Persona</span> {_esc(ep.get("persona", "none"))}',
        f'<span class="meta-label">Terminated by</span> {_esc(ep.get("who_terminated", ""))}',
        f'<span class="meta-label">Format</span> L {learner_ok}/{learner_total} · O {opponent_ok}/{opponent_total}',
    ]
    if outcome == "deal":
        meta_items.append(
            f'<span class="meta-label">Points</span> L {ep.get("learner_points")} · O {ep.get("opponent_points")}'
        )
    if ep.get("bargained_ratio") is not None:
        meta_items.append(
            f'<span class="meta-label">Bargained</span> {ep["bargained_ratio"]:.3f}'
        )
    parts.append('<span class="meta-sep"> / </span>'.join(meta_items))
    parts.append("</div>")
    parts.append("</div>")

    # System prompts (collapsible env info)
    learner_msgs = ep.get("learner_messages", [])
    opponent_msgs = ep.get("opponent_messages", [])

    l_sys = [m for m in learner_msgs if m["role"] == "system"]
    o_sys = [m for m in opponent_msgs if m["role"] == "system"]

    if l_sys or o_sys:
        parts.append('<details class="env-info">')
        parts.append('<summary class="env-info-summary">Scenario</summary>')
        parts.append('<div class="env-info-body">')
        if l_sys:
            parts.append(
                f'<div class="sys-block"><div class="sys-label">Learner</div><pre>{_esc(l_sys[0]["content"])}</pre></div>'
            )
        if o_sys:
            parts.append(
                f'<div class="sys-block"><div class="sys-label">Opponent</div><pre>{_esc(o_sys[0]["content"])}</pre></div>'
            )
        parts.append("</div></details>")

    # Chat thread
    parts.append('<div class="chat">')
    dialogue = _interleave_messages(
        learner_msgs, opponent_msgs, ep.get("learner_goes_first", True)
    )

    for speaker, role, msg in dialogue:
        if role == "system":
            continue

        if role == "assistant":
            content = msg["content"]
            parsed = _parse_message(content)
            format_ok = msg.get("format_ok", True)
            css = "msg-learner" if speaker == "Learner" else "msg-opponent"
            parts.append(f'<div class="msg {css}">')
            parts.append(f'<div class="msg-header">{speaker}')
            if not format_ok:
                parts.append(' <span class="malformed-tag">!</span>')
            parts.append("</div>")

            if parsed["reasoning"]:
                parts.append(
                    '<details class="reasoning-details">'
                    '<summary class="reasoning-summary">Reasoning</summary>'
                    f'<pre class="reasoning-output">{_esc(parsed["reasoning"])}</pre>'
                    "</details>"
                )

            for key, label in [("thought", "Thought"), ("talk", "Talk")]:
                if parsed[key]:
                    parts.append(
                        f'<div class="field {key}"><span class="field-label">{label}</span> {_esc(parsed[key])}</div>'
                    )
            if parsed["action"]:
                parts.append(
                    f'<div class="field action"><span class="field-label">Action</span> <code>{_esc(parsed["action"])}</code></div>'
                )

            if not format_ok:
                parts.append(
                    '<details class="raw-details"><summary class="raw-summary">Raw output</summary>'
                    f'<pre class="raw-output">{_esc(content)}</pre>'
                    "</details>"
                )

            parts.append("</div>")

    parts.append("</div>")  # .chat
    parts.append("</article>")
    return "\n".join(parts)


def _interleave_messages(
    learner_msgs: list[dict],
    opponent_msgs: list[dict],
    learner_goes_first: bool = True,
) -> list[tuple[str, str, dict]]:
    """Interleave learner and opponent messages into dialogue order.

    Args:
        learner_msgs: Learner's message history (system + user/assistant pairs).
        opponent_msgs: Opponent's message history.
        learner_goes_first: Whether the learner spoke first.

    Returns:
        List of (speaker_label, role, msg_dict) tuples in dialogue order.
    """
    dialogue: list[tuple[str, str, dict]] = []

    l_sys = [m for m in learner_msgs if m["role"] == "system"]
    o_sys = [m for m in opponent_msgs if m["role"] == "system"]
    if l_sys:
        dialogue.append(("Learner", "system", l_sys[0]))
    if o_sys:
        dialogue.append(("Opponent", "system", o_sys[0]))

    l_turns = [m for m in learner_msgs if m["role"] == "assistant"]
    o_turns = [m for m in opponent_msgs if m["role"] == "assistant"]

    if learner_goes_first:
        first_turns, second_turns = l_turns, o_turns
        first_label, second_label = "Learner", "Opponent"
    else:
        first_turns, second_turns = o_turns, l_turns
        first_label, second_label = "Opponent", "Learner"

    fi, si = 0, 0
    while fi < len(first_turns) or si < len(second_turns):
        if fi < len(first_turns):
            dialogue.append((first_label, "assistant", first_turns[fi]))
            fi += 1
        if si < len(second_turns):
            dialogue.append((second_label, "assistant", second_turns[si]))
            si += 1

    return dialogue


_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #09090b;
    background: #fafafa;
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* Sidebar */
.sidebar {
    width: 260px;
    flex-shrink: 0;
    border-right: 1px solid #e4e4e7;
    background: #fff;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sb-header {
    padding: 16px 16px 12px;
    border-bottom: 1px solid #e4e4e7;
    flex-shrink: 0;
}
.sb-header h1 {
    font-size: 14px;
    font-weight: 600;
    color: #09090b;
    letter-spacing: -0.01em;
}
.sb-header p {
    font-size: 12px;
    color: #71717a;
    margin-top: 2px;
}

.sb-list {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
}

.sb-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 16px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 12px;
    color: #3f3f46;
    text-align: left;
    border-left: 3px solid transparent;
    transition: background 0.1s;
}
.sb-item:hover { background: #f4f4f5; }
.sb-item.active {
    background: #f4f4f5;
}

.sb-id {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: #09090b;
    min-width: 28px;
}
.sb-turns {
    margin-left: auto;
    color: #a1a1aa;
    font-variant-numeric: tabular-nums;
}
.sb-pts {
    font-size: 11px;
    color: #71717a;
    margin-left: 4px;
}

/* Main viewer */
.viewer {
    flex: 1;
    overflow-y: auto;
    padding: 0;
}

.episode {
    max-width: 720px;
    margin: 0 auto;
    padding: 24px 24px 80px;
}

.ep-header {
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e4e4e7;
}
.ep-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.ep-num {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: #09090b;
}
.ep-turns {
    margin-left: auto;
    font-size: 12px;
    color: #a1a1aa;
    font-variant-numeric: tabular-nums;
}
.ep-meta {
    font-size: 12px;
    color: #71717a;
    line-height: 1.8;
}
.meta-label {
    font-weight: 500;
    color: #52525b;
}
.meta-sep {
    color: #d4d4d8;
}

/* Outcome tags */
.tag-deal, .tag-walkaway, .tag-reject, .tag-maxturns, .tag-unknown {
    font-size: 11px;
    font-weight: 500;
    padding: 1px 8px;
    border-radius: 9999px;
}
.tag-deal       { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.tag-walkaway   { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.tag-reject     { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.tag-maxturns   { background: #f4f4f5; color: #52525b; border: 1px solid #e4e4e7; }
.tag-unknown    { background: #f4f4f5; color: #71717a; border: 1px solid #e4e4e7; }

/* Env info (scenario) */
.env-info {
    margin-bottom: 16px;
    border: 1px solid #e4e4e7;
    border-radius: 6px;
    background: #fff;
    overflow: hidden;
}
.env-info-summary {
    display: block;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #52525b;
    cursor: pointer;
    user-select: none;
    list-style: none;
}
.env-info-summary::-webkit-details-marker { display: none; }
.env-info-summary:hover { background: #f4f4f5; }
.env-info-body {
    padding: 0 12px 12px;
}
.sys-block {
    margin-bottom: 8px;
}
.sys-block:last-child { margin-bottom: 0; }
.sys-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #a1a1aa;
    margin-bottom: 4px;
}
.sys-block pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.5;
    color: #52525b;
    background: #fafafa;
    border: 1px solid #f4f4f5;
    border-radius: 4px;
    padding: 8px 10px;
    max-height: 200px;
    overflow: auto;
}

/* Chat messages */
.chat {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.msg {
    padding: 10px 14px;
    border-radius: 6px;
    border: 1px solid #e4e4e7;
}
.msg-header {
    font-size: 12px;
    font-weight: 600;
    color: #09090b;
    margin-bottom: 6px;
}

.msg-learner { background: #fafbff; border-color: #e0e7ff; }
.msg-opponent { background: #fffbfa; border-color: #fde8e4; }

.field {
    margin-bottom: 4px;
    font-size: 13px;
    line-height: 1.55;
}
.field-label {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #a1a1aa;
    width: 60px;
    vertical-align: top;
}

.thought { color: #71717a; }
.talk { color: #27272a; }
.action code {
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    color: #3f3f46;
    background: #f4f4f5;
    padding: 1px 5px;
    border-radius: 3px;
}

.malformed-tag {
    font-size: 10px;
    font-weight: 700;
    color: #b91c1c;
    background: #fef2f2;
    border: 1px solid #fecaca;
    padding: 0 5px;
    border-radius: 9999px;
    margin-left: 4px;
}

.reasoning-details {
    margin: 4px 0 6px;
    border: 1px solid #e0e7ff;
    border-radius: 4px;
    background: #f5f3ff;
    overflow: hidden;
}
.reasoning-details[open] { margin-bottom: 8px; }
.reasoning-summary {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #6d28d9;
    padding: 4px 8px;
    cursor: pointer;
    user-select: none;
    list-style: none;
}
.reasoning-summary::-webkit-details-marker { display: none; }
.reasoning-summary:hover { background: #ede9fe; }
.reasoning-output {
    white-space: pre-wrap;
    word-break: break-word;
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #4c1d95;
    background: #ede9fe;
    padding: 8px 10px;
    max-height: 300px;
    overflow: auto;
}

.raw-details { margin-top: 6px; }
.raw-summary {
    font-size: 11px;
    color: #a1a1aa;
    cursor: pointer;
    user-select: none;
    list-style: none;
}
.raw-summary::-webkit-details-marker { display: none; }
.raw-summary:hover { color: #71717a; }
.raw-output {
    white-space: pre-wrap;
    word-break: break-word;
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #3f3f46;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 4px;
    padding: 8px 10px;
    max-height: 300px;
    overflow: auto;
}
"""

_JS = """\
document.addEventListener('DOMContentLoaded', function() {
    var items = document.querySelectorAll('.sb-item');
    var articles = document.querySelectorAll('.episode');
    items.forEach(function(btn) {
        btn.addEventListener('click', function() {
            var idx = btn.dataset.idx;
            items.forEach(function(b) { b.classList.remove('active'); });
            btn.classList.add('active');
            articles.forEach(function(a) {
                a.style.display = a.dataset.idx === idx ? '' : 'none';
            });
        });
    });
});
"""


def render_episodes_html(
    episodes: list[dict],
    title: str = "Negotiation Episodes",
    dataset: str = "",
) -> str:
    """Render a list of episode dicts to a self-contained HTML string.

    Args:
        episodes: List of serialized episode dicts.
        title: Page title.
        dataset: Dataset name for subtitle.

    Returns:
        Complete HTML document as a string.
    """
    subtitle = f"{len(episodes)} episodes"
    if dataset:
        subtitle = f"{dataset} / {subtitle}"

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_esc(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        # Sidebar
        '<nav class="sidebar">',
        '<div class="sb-header">',
        f"<h1>{_esc(title)}</h1>",
        f"<p>{_esc(subtitle)}</p>",
        "</div>",
        '<div class="sb-list">',
    ]
    for i, ep in enumerate(episodes):
        parts.append(_render_sidebar_item(ep, i))
    parts.append("</div>")  # .sb-list
    parts.append("</nav>")

    # Main viewer
    parts.append('<main class="viewer">')
    for i, ep in enumerate(episodes):
        parts.append(_render_episode(ep, i))
    parts.append("</main>")

    parts.append(f"<script>{_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_from_json(
    json_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Read episodes.json and write an HTML file next to it.

    Args:
        json_path: Path to the episodes JSON file.
        output_path: Optional explicit output path. Defaults to sibling episodes.html.

    Returns:
        Path to the written HTML file.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        episodes = json.load(f)

    matchup_dir = json_path.parent
    title = matchup_dir.name
    dataset = matchup_dir.parent.name if matchup_dir.parent.name != "negotiate" else ""
    html_content = render_episodes_html(episodes, title=title, dataset=dataset)

    if output_path is None:
        output_path = json_path.parent / "episodes.html"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path
