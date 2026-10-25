"""In-memory session store. A project is opened once and persists across turns
(SPEC §3), so the agent mutates server-held state instead of resending the graph.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

from .model import Project

_projects: dict[str, Project] = {}
_last_render: dict[str, dict[str, Any]] = {}  # key "project/scene" -> render report
_last_probe: dict[str, dict[str, Any]] = {}   # key "project/scene" -> geometry probe report
_current: dict[str, str | None] = {"id": None}
_counter = itertools.count(1)


def _base_dir() -> Path:
    return Path(os.environ.get("CHALKCAST_HOME", Path.cwd() / ".chalkcast"))


def workdir(project_id: str, scene_id: str) -> Path:
    safe_scene = scene_id.replace(":", "_")
    return _base_dir() / project_id / safe_scene


def create_project(**kw: Any) -> Project:
    # skip ids whose directory exists: another (earlier) server process used them,
    # and reusing one would overwrite artifacts it already handed out
    pid = f"proj-{next(_counter):04d}"
    while pid in _projects or (_base_dir() / pid).exists():
        pid = f"proj-{next(_counter):04d}"
    project = Project(id=pid, **kw)
    _projects[pid] = project
    _current["id"] = pid
    return project


def open_project_ids() -> list[str]:
    return list(_projects)


def get_project(project_id: str | None = None) -> Project | None:
    pid = project_id or _current["id"]
    return _projects.get(pid) if pid else None


def close_project(project_id: str | None = None) -> bool:
    pid = project_id or _current["id"]
    if pid and pid in _projects:
        del _projects[pid]
        if _current["id"] == pid:
            # fall back to the most recently opened project still open, if any
            _current["id"] = next(reversed(_projects), None)
        return True
    return False


def set_last_render(project_id: str, scene_id: str, report: dict[str, Any]) -> None:
    _last_render[f"{project_id}/{scene_id}"] = report


def get_last_render(project_id: str, scene_id: str) -> dict[str, Any] | None:
    return _last_render.get(f"{project_id}/{scene_id}")


def set_last_probe(project_id: str, scene_id: str, report: dict[str, Any]) -> None:
    _last_probe[f"{project_id}/{scene_id}"] = report


def get_last_probe(project_id: str, scene_id: str) -> dict[str, Any] | None:
    return _last_probe.get(f"{project_id}/{scene_id}")
