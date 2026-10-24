"""The server imports and its envelope + session flow behave."""

import pytest

from chalkcast import session
from chalkcast.envelope import err, ok


def test_server_imports_and_registers_mcp():
    pytest.importorskip("mcp", reason="mcp extra not installed (needs Python >=3.10)")
    import chalkcast.server as srv
    assert srv.mcp is not None
    assert srv.mcp.name == "chalkcast"


def test_envelope_shape():
    e = ok({"x": 1}, artifacts=["a.mp4"], verdict={"status": "pass"})
    assert e["ok"] is True
    assert e["result"] == {"x": 1}
    assert e["evidence"]["artifacts"] == ["a.mp4"]
    assert e["verdict"]["status"] == "pass"
    bad = err("nope")
    assert bad["ok"] is False and bad["error"] == "nope"
    # regression: err must accept a payload (render_scene's failure path relies on it)
    bad2 = err("render failed", payload={"scene_id": "scene:0001"},
               verdict={"status": "needs_fix"}, obligations=[])
    assert bad2["ok"] is False
    assert bad2["result"] == {"scene_id": "scene:0001"}
    assert bad2["verdict"]["status"] == "needs_fix"


def test_session_lifecycle():
    p = session.create_project(quality="low", theme="chalkboard")
    assert session.get_project() is p
    assert session.get_project(p.id) is p
    assert session.close_project(p.id) is True
    assert session.get_project(p.id) is None
