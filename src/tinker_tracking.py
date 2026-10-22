"""Attach experiment identity to Tinker sessions and retain their console links."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import logging
from pathlib import Path
import json
import sys
from urllib.parse import unquote, urlparse

import tinker

logger = logging.getLogger(__name__)
_active_tracker: ContextVar[SessionTracker | None] = ContextVar("tinker_tracker", default=None)


class SessionTracker:
    def __init__(self, log_path, project_id, metadata, *, wandb_enabled):
        self.path = Path(log_path) / "tinker_sessions.jsonl"
        self.project_id = project_id
        self.metadata = metadata
        self.wandb_enabled = wandb_enabled
        self.clients = []
        self.recorded_urls = set()

    def create_client(self, role, *, user_metadata=None, **kwargs):
        metadata = {**(user_metadata or {}), **self.metadata, "role": role}
        # Cookbook initializes W&B before creating its learner ServiceClient.
        wandb_run = getattr(sys.modules.get("wandb"), "run", None)
        if self.wandb_enabled and wandb_run is not None:
            metadata["wandb_link"] = wandb_run.url
        kwargs.setdefault("project_id", self.project_id)
        client = tinker.ServiceClient(user_metadata=metadata, **kwargs)
        self.clients.append((role, client, metadata))
        self.record_sessions()
        return client

    def record_sessions(self):
        for role, client, metadata in self.clients:
            url = client.get_console_url()
            # SDK 0.30.4 creates sessions lazily. Before the first training or
            # sampling client, this returns the generic /sessions page.
            path = urlparse(url).path
            if "/sessions/" not in path or url in self.recorded_urls:
                continue
            session_id = unquote(path.rsplit("/", 1)[-1])
            if not session_id:
                continue
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "run_name": self.metadata["run_name"],
                "role": role,
                "project_id": self.project_id,
                "session_id": session_id,
                "session_url": url,
                "user_metadata": metadata,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append so resumed launches retain links to their earlier sessions.
            with self.path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            self.recorded_urls.add(url)
            logger.info("Tinker %s session: %s", role, url)


class _CookbookTinkerAdapter:
    """Override only Cookbook's client factory; leave the SDK module untouched."""

    def __init__(self, original, tracker):
        self.original = original
        self.tracker = tracker

    def ServiceClient(self, **kwargs):
        return self.tracker.create_client("learner", **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def track_tinker_sessions(cookbook, *, log_path, project_id, metadata, wandb_enabled):
    """Scope the adapter to one CLI training invocation, including failures.

    Cookbook 0.4.3 has no ServiceClient injection setting. This narrow adapter
    covers its learner factory while our dataset builder uses the helper below.
    Like the CLI's environment settings, it assumes one training job per process.
    """
    tracker = SessionTracker(log_path, project_id, metadata, wandb_enabled=wandb_enabled)
    original = cookbook.tinker
    token = _active_tracker.set(tracker)
    cookbook.tinker = _CookbookTinkerAdapter(original, tracker)
    try:
        yield tracker
    finally:
        cookbook.tinker = original
        _active_tracker.reset(token)
        # Also capture a session if model initialization failed before the
        # dataset builder had a chance to record it. Preserve the original error.
        failing = sys.exc_info()[0] is not None
        try:
            tracker.record_sessions()
        except Exception:
            if not failing:
                raise
            logger.exception("Could not save Tinker session links during cleanup")


def create_service_client(role, **kwargs):
    tracker = _active_tracker.get()
    if tracker is None:
        return tinker.ServiceClient(**kwargs)
    return tracker.create_client(role, **kwargs)


def record_tinker_sessions():
    tracker = _active_tracker.get()
    if tracker is not None:
        tracker.record_sessions()
