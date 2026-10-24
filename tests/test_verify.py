"""The geometric verification lanes catch the problems they claim to."""

from chalkcast.demo import build_demo_project
from chalkcast.model import Obj, Project
from chalkcast.verify import inspect_scene


def _scene_with(*objs, steps=()):
    project = Project(id="p")
    scene = project.new_scene()
    scene.objects.extend(objs)
    for kind, kw in steps:
        scene.add_step(kind, **kw)
    return scene


def test_demo_scene_passes():
    _, scene = build_demo_project()
    insp = inspect_scene(scene)
    errors = [o for o in insp["obligations"] if o["severity"] == "error"]
    assert errors == [], f"demo should be clean, got: {errors}"
    assert insp["verdict"]["status"] in ("pass", "pass_with_warnings")


def test_framing_lane_flags_offscreen_object():
    scene = _scene_with(Obj(id="text_1", kind="text", position=(20, 0),
                            payload={"text": "way off", "font_size": 36}))
    insp = inspect_scene(scene)
    lanes = {o["lane"] for o in insp["obligations"]}
    assert "framing" in lanes
    assert insp["verdict"]["status"] == "needs_fix"


def test_timing_lane_flags_highlight_before_write():
    scene = _scene_with(
        Obj(id="text_1", kind="text", position=(0, 0), payload={"text": "hi", "font_size": 36}),
        steps=[("highlight", {"targets": ["text_1"]})],  # highlight with no prior write
    )
    insp = inspect_scene(scene)
    timing = [o for o in insp["obligations"] if o["lane"] == "timing"]
    assert timing and timing[0]["severity"] == "error"


def test_timing_lane_flags_unknown_target():
    # regression: a step targeting an id with no matching object would make
    # codegen emit an undefined variable and crash the render.
    scene = _scene_with(
        Obj(id="text_1", kind="text", position=(0, 0), payload={"text": "hi", "font_size": 36}),
        steps=[("write", {"targets": ["plot_9"]})],  # no such object
    )
    insp = inspect_scene(scene)
    unknown = [o for o in insp["obligations"]
               if o["lane"] == "timing" and "unknown object" in o["summary"]]
    assert unknown and unknown[0]["severity"] == "error"


def test_collision_lane_flags_overlap():
    scene = _scene_with(
        Obj(id="text_1", kind="text", position=(0, 0), payload={"text": "AAAA", "font_size": 48}),
        Obj(id="text_2", kind="text", position=(0, 0), payload={"text": "BBBB", "font_size": 48}),
    )
    insp = inspect_scene(scene)
    assert any(o["lane"] == "collision" for o in insp["obligations"])


def test_compile_lane_runs_only_with_render_report():
    _, scene = build_demo_project()
    without = inspect_scene(scene)
    assert "compile" not in without["lanes_run"]
    withr = inspect_scene(scene, render_report={"ok": False, "error": "boom"})
    assert "compile" in withr["lanes_run"]
    assert any(o["lane"] == "compile" for o in withr["obligations"])
