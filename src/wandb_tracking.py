"""Keep one W&B identity across checkpointed CLI training launches.

This logging-only integration lives outside the manifest-pinned training code,
so adopting it does not invalidate existing weights/optimizer checkpoints.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log


def resolve_wandb_identity(log_path, project):
    """Read a saved identity, or recover one unambiguous legacy session link."""
    directory = Path(log_path)
    identities = set()
    marker = directory / "wandb_run.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        identities.add(tuple(record[key] for key in ("entity", "project", "id")))
    sessions = directory / "tinker_sessions.jsonl"
    if sessions.exists():
        for line in sessions.read_text().splitlines():
            link = json.loads(line).get("user_metadata", {}).get("wandb_link")
            if not link:
                continue
            parts = [unquote(p) for p in urlparse(link).path.strip("/").split("/")]
            if len(parts) != 4 or parts[2] != "runs":
                raise ValueError(f"Cannot recover W&B identity from session link: {link}")
            identities.add((parts[0], parts[1], parts[3]))
    if len(identities) > 1:
        raise ValueError("Multiple W&B runs found for this experiment; cannot choose one automatically")
    if not identities:
        return None
    entity, saved_project, run_id = identities.pop()
    if not all(isinstance(value, str) and value for value in (entity, saved_project, run_id)):
        raise ValueError("Incomplete saved W&B run identity")
    if saved_project != project:
        raise ValueError("Saved W&B project does not match the training configuration")
    return {"entity": entity, "project": saved_project, "id": run_id}


class _ConfigAdapter:
    def __init__(self, original, total_steps):
        self.original, self.total_steps = original, total_steps

    def __getattr__(self, name):
        return getattr(self.original, name)

    def update(self, values, **kwargs):
        # Cookbook logs runtime hparams again after init. Keep its launch
        # endpoint from overwriting the immutable experiment budget.
        if "max_steps" in values:
            values = {**values, "max_steps": self.total_steps}
        return self.original.update(values, **kwargs)


class _WandbAdapter:
    def __init__(self, original, fork_checkpoint=None):
        self.original = original
        self.fork_checkpoint = fork_checkpoint
        self.total_steps = None

    def __getattr__(self, name):
        if name == "config" and self.total_steps is not None:
            return _ConfigAdapter(self.original.config, self.total_steps)
        return getattr(self.original, name)

    def init(self, **kwargs):
        directory = Path(kwargs["dir"])
        identity = resolve_wandb_identity(directory, kwargs["project"])
        checkpoint = checkpoint_utils.get_last_checkpoint(str(directory))
        new_fork = (identity is None and checkpoint is not None
                    and self.fork_checkpoint is not None
                    and checkpoint.to_dict() == self.fork_checkpoint)
        if checkpoint is not None and identity is None and not new_fork:
            raise ValueError("Cannot resume W&B: checkpoint exists but no saved W&B identity was found")
        if self.original.run is not None:
            raise ValueError("An active W&B run already exists; launch training in a separate process")
        if identity:
            kwargs.update(identity, resume="must")
            # Preserve the original run's configuration; max_steps in Cookbook's
            # config is a per-launch endpoint, recorded in executions.jsonl.
            kwargs.pop("config", None)
        else:
            kwargs.update(id=uuid4().hex[:8], resume="never")
        run = self.original.init(**kwargs)
        try:
            if identity and (not run.resumed or run.id != identity["id"]):
                raise RuntimeError("W&B did not resume the requested run; training has not started")
            start = checkpoint.batch if checkpoint is not None else 0
            if run.step > start:
                raise ValueError(
                    f"W&B history is ahead of the checkpoint (next step {run.step}, checkpoint {start}); "
                    "resolve the history/checkpoint mismatch before resuming"
                )
            record = {"entity": run.entity, "project": run.project, "id": run.id, "url": run.url}
            temporary = directory / "wandb_run.json.tmp"
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(directory / "wandb_run.json")
            # The declared full budget is stable even when the launch stops early.
            manifest = directory / "training_manifest.json"
            if manifest.exists():
                total = json.loads(manifest.read_text())["config"]["max_steps"]
                self.total_steps = total
                run.config.update({"max_steps": total}, allow_val_change=True)
            fork = directory / "checkpoint_fork.json"
            if fork.exists():
                run.config.update({"checkpoint_fork": json.loads(fork.read_text())}, allow_val_change=True)
            print(f"W&B {'resumed' if identity else 'run'}: {run.url}", flush=True)
            return run
        except BaseException:
            run.finish(exit_code=1)
            raise


@contextmanager
def reuse_wandb_run(*, fork_checkpoint=None):
    """Adapt only Cookbook logging; no training state or SDK globals are changed."""
    original = ml_log.wandb
    if original is None:
        yield
        return
    ml_log.wandb = _WandbAdapter(original, fork_checkpoint=fork_checkpoint)
    try:
        yield
    finally:
        ml_log.wandb = original
