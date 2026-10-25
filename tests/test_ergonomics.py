"""No surprising implicit behaviour: one test per finding of an independent audit.

An auditor who had not written the server drove every tool looking for calls that
silently did something other than what their arguments said. Each finding is pinned
here: the call now either does exactly what it says, or fails with an explicit
error, or reports the side effect in its result.
"""

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

import chalkcast.server as srv  # noqa: E402
from chalkcast import session  # noqa: E402
from chalkcast.codegen import generate_module  # noqa: E402
from chalkcast.verify import inspect_scene  # noqa: E402


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CHALKCAST_HOME", str(tmp_path))


@pytest.fixture()
def scene():
    srv.open_project()
    srv.add_scene("t")
    return session.get_project().get_scene(None)


def _id(r):
    assert r["ok"], r.get("error")
    return r["result"]["object_id"]


def _lanes(sc, severity="error"):
    return [o for o in inspect_scene(sc)["obligations"] if o["severity"] == severity]


# 1. a scene with objects but no steps used to render blank and still pass
def test_no_steps_is_an_error(scene):
    _id(srv.add_text("NO STEPS"))
    assert any("no steps" in o["summary"] for o in _lanes(scene))


# 2. cascade removal used to delete a shared step, silently hiding other objects
def test_cascade_keeps_other_targets_of_a_shared_step(scene):
    a, b = _id(srv.add_text("A", y=1)), _id(srv.add_text("B", y=-1))
    srv.add_step("write", targets=[a, b])
    r = srv.remove_object(a, cascade=True)["result"]
    assert r["removed_steps"] == [] and r["edited_steps"] == [1]
    assert scene.steps[0].targets == [b] and r["objects_without_step"] == []


# 3. camera zoom used to be cumulative and a zoom-only move re-centred on the origin
def test_camera_zoom_is_absolute_and_focus_optional(scene):
    t = _id(srv.add_text("A"))
    srv.add_step("write", targets=[t])
    srv.add_step("camera_move", zoom=2, focus_x=1, focus_y=1)
    srv.add_step("camera_move", zoom=2)
    srv.add_step("camera_move", zoom=1)
    code = generate_module(scene, "chalkboard")
    assert code.count("set_width(config.frame_width / 2.0).move_to([1.0, 1.0, 0])") == 1
    assert "set_width(config.frame_width / 2.0))" in code      # keeps the centre
    assert "set_width(config.frame_width / 1.0))" in code      # back to the full frame
    assert ".scale(" not in code.split("def construct")[1]
    assert srv.add_step("camera_move", zoom=2, focus_x=1)["ok"] is False  # half a focus


# 4. with_previous after a wait was dropped; multi-target writes were sequential
def test_with_previous_and_multi_target_semantics(scene):
    a, b, c = (_id(srv.add_text(s, y=y)) for s, y in (("A", 2), ("B", 0), ("C", -2)))
    srv.add_step("write", targets=[a, b], duration=2)
    srv.add_step("wait", duration=1)
    assert srv.add_step("highlight", targets=[a], with_previous=True)["ok"] is False
    assert srv.add_step("wait", with_previous=True)["ok"] is False
    srv.add_step("write", targets=[c])
    srv.add_step("highlight", targets=[a], with_previous=True)
    code = generate_module(scene, "chalkboard")
    assert "self.play(Write(text_1, run_time=2), Write(text_2, run_time=2))" in code
    assert "self.play(Write(text_3, run_time=1.0), Indicate(text_1, color='#f6c76b', run_time=1.0))" in code


def test_remove_step_reports_regrouping(scene):
    a, b = _id(srv.add_text("A", y=1)), _id(srv.add_text("B", y=-1))
    srv.add_step("write", targets=[a])
    srv.add_step("write", targets=[b])
    srv.add_step("highlight", targets=[a], with_previous=True)
    r = srv.remove_step(2)["result"]
    assert r["regrouped_steps"] == [2] and scene.steps[1].with_previous is False


# 5. the reported duration ignored parallel steps
def test_duration_matches_the_timeline(scene):
    a, b = _id(srv.add_text("A", y=1)), _id(srv.add_text("B", y=-1))
    srv.add_step("write", targets=[a, b], duration=2)
    srv.add_step("highlight", targets=[a], duration=1, with_previous=True)
    srv.add_step("wait", duration=0.5)
    assert scene.duration() == 2.5


# 6. export_gif used to export a stale render after edits
def test_export_refuses_a_stale_render(scene, tmp_path):
    t = _id(srv.add_text("A"))
    srv.add_step("write", targets=[t])
    fake = tmp_path / "v.mp4"
    fake.write_bytes(b"")
    project = session.get_project()
    session.set_last_render(project.id, scene.id, {"ok": True, "artifact": str(fake),
                                                   "fingerprint": srv._fingerprint(scene)})
    srv.update_object(t, text="EDITED AFTER RENDER")
    r = srv.export_gif()
    assert r["ok"] is False and "changed since its last render" in r["error"]


# 7. update_object used to commit some fields and then fail on a later one
def test_update_is_all_or_nothing(scene):
    t = _id(srv.add_text("orig"))
    assert srv.update_object(t, x=5, text="CHANGED", scale=-1)["ok"] is False
    obj = scene.get_obj(t)
    assert obj.payload["text"] == "orig" and obj.position == (0, 0) and obj.scale == 1.0


# 8. update_object skipped the add-time validation
@pytest.mark.parametrize("field,value", [("font_size", -10), ("width", -3), ("opacity", 4),
                                         ("dx", -1), ("size", -2)])
def test_update_validates_like_add(scene, field, value):
    t = _id(srv.add_text("a"))
    p = _id(srv.add_plot(functions=[{"expr": "x"}], x_range=[0, 2], y_range=[0, 2]))
    area = _id(srv.add_area(p, 0, 1))
    rie = _id(srv.add_riemann(p, 0, 1, dx=0.25))
    dot = _id(srv.add_shape("dot"))
    target = {"font_size": t, "width": p, "opacity": area, "dx": rie, "size": dot}[field]
    assert srv.update_object(target, **{field: value})["ok"] is False


def test_add_validates_sizes_and_opacity(scene):
    p = _id(srv.add_plot(functions=[{"expr": "x"}], x_range=[0, 2], y_range=[0, 2]))
    assert srv.add_area(p, 0, 1, opacity=7)["ok"] is False
    assert srv.add_shape("square", size=-2)["ok"] is False


# 9. replacing a plot's functions left a dependent area pointing at a missing one
def test_update_functions_checks_dependents(scene):
    p = _id(srv.add_plot(functions=["x", "x**2"], x_range=[0, 2], y_range=[0, 4]))
    _id(srv.add_area(p, 0, 1, function_index=2))
    r = srv.update_object(p, functions=[{"expr": "x"}])
    assert r["ok"] is False and "area_2" in r["error"]


# 10. moving/scaling an area was accepted and had no effect
def test_overlays_cannot_be_moved_directly(scene):
    p = _id(srv.add_plot(functions=["x"], x_range=[0, 2], y_range=[0, 2]))
    area = _id(srv.add_area(p, 0, 1))
    assert "move the plot" in srv.update_object(area, x=3)["error"]
    assert srv.update_object(area, scale=0.5)["ok"] is False


# 11. shape arguments that do not apply were silently ignored
@pytest.mark.parametrize("kwargs", [
    {"shape": "dot", "at_y": "1"},                                   # missing at_x
    {"shape": "dot", "at_x": "1", "at_y": "1", "x": 4},              # x ignored by anchor
    {"shape": "line", "size": 5, "x1": 0, "y1": 0, "x2": 1, "y2": 1},  # size ignored
    {"shape": "line", "x": 3, "x1": 0, "y1": 0, "x2": 1, "y2": 1},     # x ignored
    {"shape": "circle", "width": 5},                                 # width ignored
    {"shape": "rectangle", "size": 2},                               # size ignored
])
def test_inapplicable_shape_arguments_rejected(scene, kwargs):
    p = _id(srv.add_plot(functions=["x"], x_range=[0, 2], y_range=[0, 2]))
    if "at_x" in kwargs or "at_y" in kwargs:
        kwargs = {**kwargs, "plot_id": p}
    assert srv.add_shape(**kwargs)["ok"] is False


# 13. highlighting a faded object silently brought it back
def test_highlight_after_fade_is_an_error(scene):
    t = _id(srv.add_text("GHOST"))
    srv.add_step("write", targets=[t])
    srv.add_step("fade", targets=[t])
    srv.add_step("highlight", targets=[t])
    assert any("faded out" in o["summary"] for o in _lanes(scene))


# 14. fps was ignored and 'draft' was the same as 'low'
def test_fps_and_quality_are_explicit():
    assert srv.open_project(quality="draft")["ok"] is False
    assert srv.open_project(fps=0)["ok"] is False
    assert srv.open_project(quality="medium")["result"]["fps"] == 30
    assert session.get_project().fps == 30


# 15. areas outside the plot's x_range were accepted
def test_area_interval_must_lie_in_the_plot(scene):
    p = _id(srv.add_plot(functions=["x"], x_range=[-5, 5], y_range=[-5, 5]))
    assert "outside the x_range" in srv.add_area(p, -20, 20)["error"]
    assert srv.add_riemann(p, 0, 9, dx=1)["ok"] is False


# 16. two curves without a color were identical; default ranges were invisible
def test_curves_get_distinct_default_colors_and_ranges_are_echoed(scene):
    r = srv.add_plot(functions=["sin(x)", "cos(x)"])
    assert r["result"]["x_range"] == [-5, 5, 1] and r["result"]["y_range"] == [-3, 3, 1]
    code = generate_module(scene, "chalkboard")
    assert "#f6c76b" in code and "#6bc5f6" in code


# 18. step arguments that do not apply were stored silently
@pytest.mark.parametrize("kwargs", [
    {"kind": "wait", "targets": ["text_1"]},
    {"kind": "wait", "zoom": 3},
    {"kind": "highlight", "targets": ["text_1"], "to": "text_1"},
    {"kind": "write", "targets": ["text_1"], "param": "t", "value": 1},
])
def test_inapplicable_step_arguments_rejected(scene, kwargs):
    _id(srv.add_text("a"))
    assert srv.add_step(**kwargs)["ok"] is False


# 19. results now name the scene they acted on
def test_results_name_their_scene(scene):
    r = srv.add_text("a")
    assert r["result"]["scene_id"] == scene.id


# 20. redeclaring a parameter is reported
def test_parameter_redeclaration_is_reported(scene):
    srv.add_parameter("t", 1.0)
    assert srv.add_parameter("t", 5.0)["result"]["redeclared_from"] == 1.0


# 21. transforming into an object already on screen leaves two copies
def test_transform_into_visible_object_warns(scene):
    a, b = _id(srv.add_text("A", y=1)), _id(srv.add_text("B", y=-1))
    srv.add_step("write", targets=[a, b])
    srv.add_step("transform", targets=[a], to=b)
    assert any("already on screen" in o["summary"] for o in _lanes(scene, "warning"))


# 22. small cases
def test_small_explicit_failures(scene):
    assert srv.fix_scene(max_rounds=0)["ok"] is False
    assert srv.close_project("proj-9999")["ok"] is False
    p = _id(srv.add_plot(functions=["x"], x_range=[0, 2], y_range=[0, 2]))
    area = _id(srv.add_area(p, 0, 1, color=None))
    assert scene.get_obj(area).color == "BLUE"


# ---- second audit ---------------------------------------------------------------

# A1. transforming a parameter-driven object snapped back afterwards
def test_transform_of_dynamic_object_rejected(scene):
    srv.add_parameter("t", 0)
    p1 = _id(srv.add_plot(functions=["sin(x - t)"]))
    p2 = _id(srv.add_plot(functions=["2*cos(3*x)"]))
    srv.add_step("write", targets=[p1])
    assert "snap back" in srv.add_step("transform", targets=[p1], to=p2)["error"]


# A2. a [min, max] range meant one tick per unit (200 ticks on [0, 200])
def test_two_element_range_gets_readable_ticks(scene):
    r = srv.add_plot(functions=["0.01*x**2"], x_range=[0, 200], y_range=[0, 400])["result"]
    assert r["x_range"] == [0, 200, 25] and r["y_range"] == [0, 400, 50]


# A3/A6/A17. areas and Riemann sums that could not show the integral
def test_overlays_must_show_their_integral(scene):
    p = _id(srv.add_plot(functions=["sqrt(x)"], x_range=[-2, 4], y_range=[-1, 3]))
    assert "undefined on part" in srv.add_area(p, -1, 2)["error"]
    q = _id(srv.add_plot(functions=["x**2"], x_range=[0, 3], y_range=[0, 4]))
    assert "leaves y_range" in srv.add_riemann(q, 0, 3, dx=0.5)["error"]
    r = _id(srv.add_plot(functions=["x+3"], x_range=[0, 5], y_range=[2, 9]))
    assert "y = 0" in srv.add_area(r, 0, 5)["error"]


# A4. a last Riemann rectangle hung past x_max
def test_riemann_needs_whole_number_of_rectangles(scene):
    p = _id(srv.add_plot(functions=["sin(x)"], x_range=[0, 6.3], y_range=[-1.5, 1.5]))
    r = srv.add_riemann(p, 0, 6.3, dx=0.5)
    assert r["ok"] is False and "whole number" in r["error"]
    assert srv.add_riemann(p, 0, 6.0, dx=0.5)["ok"]


# A5. rectangles below the axis changed colour
def test_riemann_is_one_color(scene):
    p = _id(srv.add_plot(functions=["sin(x)"], x_range=[0, 6.3], y_range=[-1.5, 1.5]))
    _id(srv.add_riemann(p, 0, 6.0, dx=0.5))
    assert "show_signed_area=False" in generate_module(scene, "chalkboard")


# A7. a size-2 arrow was 1.5 long
def test_arrow_size_is_its_length(scene):
    _id(srv.add_shape("arrow", size=2))
    assert "Arrow([-1.0, 0, 0], [1.0, 0, 0], buff=0" in generate_module(scene, "chalkboard")


# A8. `x` in an anchor was silently 0
def test_anchor_cannot_use_x(scene):
    p = _id(srv.add_plot(functions=["x**2"], x_range=[-3, 3], y_range=[0, 9]))
    assert "cannot use x" in srv.add_shape("dot", plot_id=p, at_x="2", at_y="x**2")["error"]


# A9. labels shifted the axes away from (x, y)
def test_axes_are_centred_at_the_plot_position(scene):
    _id(srv.add_plot(functions=["sin(x)"], x_label=r"\text{time (s)}", x=1, y=-1))
    assert "plot_1.shift(np.array([1, -1, 0]) - plot_1_ax.get_center())" in \
        generate_module(scene, "chalkboard")


# A11. a lone '=' was "too small" and fix_scene blew it up
def test_short_glyph_is_not_illegible(scene):
    e = _id(srv.add_equation("="))
    srv.add_step("write", targets=[e])
    assert not [o for o in inspect_scene(scene)["obligations"] if o["lane"] == "legibility"]


# A13. writing an object already on screen erases and redraws it
def test_rewriting_a_visible_object_warns(scene):
    t = _id(srv.add_text("A"))
    srv.add_step("write", targets=[t])
    srv.add_step("write", targets=[t])
    assert any("already on screen" in o["summary"] for o in _lanes(scene, "warning"))


# A14. content written outside a zoomed camera passed
def test_writing_outside_the_camera_view_is_flagged(scene):
    a = _id(srv.add_text("Title", y=3))
    b = _id(srv.add_text("Note", x=-5, y=-3))
    srv.add_step("write", targets=[a])
    srv.add_step("camera_move", zoom=3)
    srv.add_step("write", targets=[b])
    # never revealed by a later zoom-out: it is never seen at all
    assert any("outside the camera" in o["summary"] for o in _lanes(scene, "error"))


# A15. smooth='false' turned smoothing ON; dx='0.1' failed only later
def test_function_options_are_type_checked(scene):
    assert "true or false" in srv.add_plot(functions=[{"expr": "x", "smooth": "false"}])["error"]
    assert "must be a number" in srv.add_plot(functions=[{"expr": "x", "dx": "0.1"}])["error"]


# A17. small items
def test_light_theme_gets_dark_curves_and_ticks_are_checked():
    srv.open_project(theme="whiteboard")
    srv.add_scene()
    sc = session.get_project().get_scene(None)
    p = _id(srv.add_plot(functions=["x"], show_numbers=True))
    assert "#f6c76b" not in generate_module(sc, "whiteboard")
    srv.add_step("write", targets=[p])
    srv.update_object(p, scale=0.35)
    assert any("tick numbers" in o["summary"] for o in _lanes(sc, "warning"))


def test_closing_falls_back_to_another_open_project():
    first = srv.open_project()["result"]["project_id"]
    srv.open_project()
    assert srv.close_project()["result"]["current_project"] == first
    assert srv.add_scene()["ok"]


# ---- third audit ----------------------------------------------------------------

# B1. a shape's scale was stored but never rendered
def test_shape_scale_is_rendered(scene):
    s = _id(srv.add_shape("circle", size=1))
    srv.update_object(s, scale=3)
    assert "Circle(radius=0.5, color='#f4f1e8').scale(3)" in generate_module(scene, "chalkboard") \
        or ".scale(3)" in generate_module(scene, "chalkboard").split(f"{s} = ")[1].split("\n")[0]


# B2. segments are placed by their endpoints: no silent scale/position fix
def test_segment_cannot_be_scaled(scene):
    s = _id(srv.add_shape("line", x1=-9, y1=0, x2=9, y2=1))
    assert "x1, y1, x2, y2" in srv.update_object(s, scale=0.5)["error"]


# B3. update_object skipped the whole-number check of add_riemann
def test_update_riemann_keeps_whole_rectangles(scene):
    p = _id(srv.add_plot(functions=["x**2/4"], x_range=[0, 3], y_range=[0, 3]))
    r = _id(srv.add_riemann(p, 0, 2, dx=0.5))
    assert "whole number" in srv.update_object(r, dx=0.3)["error"]
    assert "whole number" in srv.update_object(r, x_max=1.9)["error"]


# B4. a Python keyword as a parameter name broke every plot
@pytest.mark.parametrize("name", ["lambda", "None", "for", "match", "sum"])
def test_keyword_parameter_names_rejected(scene, name):
    assert srv.add_parameter(name, 1.0)["ok"] is False


# B5. animating a parameter-driven object during a tween froze it, then it jumped
def test_animating_a_driven_object_during_its_tween_rejected(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["sin(x - t)"]))
    srv.add_step("write", targets=[p], duration=3)
    srv.add_step("highlight", targets=[p], duration=2)
    r = srv.add_step("tween", param="t", value=3.14, duration=2, with_previous=True)
    assert r["ok"] is False and "freeze" in r["error"]
    assert len(scene.steps) == 2  # nothing half-added


# B7/B9. a plot's curve was drawn only at the end; highlight ignored the theme
def test_plot_drawn_at_once_and_highlight_uses_accent():
    srv.open_project(theme="whiteboard")
    srv.add_scene()
    sc = session.get_project().get_scene(None)
    p = _id(srv.add_plot(functions=["sin(x)"]))
    srv.add_step("write", targets=[p], duration=3)
    srv.add_step("highlight", targets=[p])
    code = generate_module(sc, "whiteboard")
    assert "Create(plot_1, run_time=3, lag_ratio=0)" in code
    assert "Indicate(plot_1, color='#2a6df5'" in code


# B8. fix_scene's fidelity repair flip-flopped forever
def test_fidelity_repair_does_not_loop(scene):
    from chalkcast.fix import apply_fixes
    p = _id(srv.add_plot(functions=["sin(1000*x)"]))
    obl = [{"lane": "fidelity", "severity": "error", "target": p, "fn": 1, "summary": "x"}]
    attempts: dict = {}
    seen = []
    for _ in range(4):
        changes, _ = apply_fixes(scene, obl, None, attempts)
        seen.append(len(changes))
    assert seen[:1] == [1] and seen[2:] == [0, 0]


# B10. a zoomed-out camera shrinks text below legibility
def test_legibility_accounts_for_zoom(scene):
    t = _id(srv.add_text("small print", font_size=24))
    srv.add_step("camera_move", zoom=0.4)
    srv.add_step("write", targets=[t])
    assert any(o["lane"] == "legibility" for o in _lanes(scene, "warning"))


# B11. a tween of a parameter nothing uses
def test_unused_tween_is_reported(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["sin(x)"]))
    srv.add_step("write", targets=[p])
    r = srv.add_step("tween", param="t", value=5)
    assert "changes nothing" in r["result"]["warning"]
    assert any("no object uses" in o["summary"] for o in _lanes(scene, "warning"))


# B13. a probe timeout is not a compile error and does not block rendering
def test_probe_timeout_is_a_measure_warning(scene):
    t = _id(srv.add_text("A"))
    srv.add_step("write", targets=[t])
    insp = inspect_scene(scene, measurement={"ok": None, "timeout": True,
                                             "error": "geometry probe timed out after 120s"})
    lanes = {(o["lane"], o["severity"]) for o in insp["obligations"]}
    assert ("measure", "warning") in lanes and not any(l == "compile" for l, _ in lanes)


# B14. duplicate targets were accepted
def test_duplicate_targets_rejected(scene):
    t = _id(srv.add_text("A"))
    assert "duplicate" in srv.add_step("write", targets=[t, t])["error"]


# ---- fourth audit ---------------------------------------------------------------

# C1. float rounding in np.arange added a Riemann rectangle past x_max
def test_riemann_stops_before_x_max(scene):
    p = _id(srv.add_plot(functions=["x+1"], x_range=[0, 1.6], y_range=[0, 3]))
    _id(srv.add_riemann(p, 1, 1.6, dx=0.2))
    assert "x_range=[1.0, 1.5], dx=0.2" in generate_module(scene, "chalkboard")


# C2/C3. dx no longer disables jump splitting; small jumps on a flat line are jumps
def test_jumps_split_with_dx_and_when_small():
    pytest.importorskip("manim")
    import numpy as np
    from chalkcast.codegen import _PRELUDE
    ns: dict = {}
    exec(_PRELUDE, ns)  # noqa: S102 - our own generated code
    ax = ns["Axes"](x_range=[-3, 3], y_range=[-5, 5], x_length=6, y_length=4)
    with_dx = ns["_cc_plot"](ax, lambda x: np.sign(x), "#fff", dx=0.01)
    assert len(with_dx.submobjects) == 2
    floor = ns["_cc_plot"](ax, lambda x: np.floor(x), "#fff")   # unit jumps = 10% of span
    assert len(floor.submobjects) == 6


# C4. legibility repairs account for the zoom (and never report a no-op)
def test_legibility_fix_accounts_for_zoom(scene):
    from chalkcast.fix import apply_fixes
    t = _id(srv.add_text("hello", font_size=30))
    srv.add_step("camera_move", zoom=0.5)
    srv.add_step("write", targets=[t])
    obl = [{"lane": "legibility", "severity": "warning", "target": t, "summary": "x"}]
    changes, _ = apply_fixes(scene, obl, None)
    assert changes and changes[0]["font_size"] * 0.5 >= 20
    assert not any(o["lane"] == "legibility" for o in _lanes(scene, "warning"))


# C5. two tweens of one parameter in a beat: the first was silently ignored
def test_double_tween_in_a_beat_rejected(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["0*x"], x_range=[-6, 6]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="0"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=5, duration=1)
    r = srv.add_step("tween", param="t", value=-5, duration=3, with_previous=True)
    assert r["ok"] is False and "twice" in r["error"]


# C6. a zoom-out after the write also counts
def test_legibility_after_a_later_zoom_out(scene):
    t = _id(srv.add_text("x", font_size=30))
    srv.add_step("write", targets=[t])
    srv.add_step("camera_move", zoom=0.4)
    assert any(o["lane"] == "legibility" for o in _lanes(scene, "warning"))


# C7. axis labels are checked for legibility
def test_axis_labels_legibility(scene):
    p = _id(srv.add_plot(functions=["x"], x_label="t", y_label="v"))
    srv.update_object(p, scale=0.3)
    srv.add_step("write", targets=[p])
    assert any("axis labels" in o["summary"] for o in _lanes(scene, "warning"))


# C8. blank text / LaTeX made invisible objects
def test_blank_text_rejected(scene):
    assert srv.add_text("   ")["ok"] is False
    e = _id(srv.add_equation("a+b"))
    assert srv.update_object(e, tex="")["ok"] is False


# C9. estimated geometry places explicit segments by their endpoints
def test_estimated_segment_box_uses_endpoints(scene):
    s = _id(srv.add_shape("line", x1=5, y1=0, x2=20, y2=0))
    srv.add_step("write", targets=[s])
    assert any(o["lane"] == "framing" for o in inspect_scene(scene)["obligations"])


# ---- fifth audit ----------------------------------------------------------------

def _measured(scene):
    pytest.importorskip("manim")
    r = srv.inspect_render()
    assert r["coverage"]["measured"] is True
    return r


# D1. where a tween leaves a moving curve/dot is checked against later writes
@pytest.mark.parametrize("label_with_previous", [False, True])
def test_collision_after_a_tween(scene, label_with_previous):
    srv.add_parameter("a", 0)
    p = _id(srv.add_plot(functions=["a*x"], x_range=[-4, 4], y_range=[-4, 4], height=6))
    t = _id(srv.add_text("slope", x=2.0, y=2.0, font_size=36))
    srv.add_step("write", targets=[p])
    srv.add_step("tween", param="a", value=1, duration=2)
    srv.add_step("write", targets=[t], with_previous=label_with_previous)
    r = _measured(scene)
    assert any(o["lane"] == "collision" and "tween" in o["summary"] for o in r["obligations"])


# D2. segments can be moved by their endpoints
def test_update_segment_endpoints(scene):
    s = _id(srv.add_shape("arrow", x1=0, y1=0, x2=2, y2=1))
    r = srv.update_object(s, x2=3, y2=2)
    assert r["ok"] and r["result"]["object"]["payload"]["x2"] == 3
    assert srv.update_object(_id(srv.add_shape("dot")), x1=1)["ok"] is False


# D3. fix_scene moving a plot keeps its overlays' stored position in sync
def test_fix_scene_moves_overlays_with_plot(scene):
    p = _id(srv.add_plot(functions=["x**2/4"], x_range=[-4, 4], y_range=[0, 4], x=4))
    a = _id(srv.add_area(p, 0, 2))
    srv.add_step("write", targets=[p])
    srv.add_step("write", targets=[a])
    pytest.importorskip("manim")
    srv.fix_scene()
    assert scene.get_obj(a).position == scene.get_obj(p).position


# D4. a new server process never reuses a project id whose directory exists
def test_project_ids_skip_existing_directories(tmp_path):
    (tmp_path / "proj-0100").mkdir()
    import itertools
    session._counter = itertools.count(100)
    assert session.create_project().id == "proj-0101"


# ---- sixth audit ----------------------------------------------------------------

# E1. framing is judged in the camera's view while the object is shown
def test_framing_follows_the_camera(scene):
    a, b = _id(srv.add_text("Left", x=-9)), _id(srv.add_text("Right", x=9))
    srv.add_step("camera_move", zoom=0.5)
    srv.add_step("write", targets=[a, b])
    r = _measured(scene)
    assert not [o for o in r["obligations"] if o["lane"] == "framing" and o["severity"] == "error"]


def test_fix_scene_moves_into_the_camera_view(scene):
    t = _id(srv.add_text("Over here", x=12))
    srv.add_step("camera_move", zoom=1, focus_x=12, focus_y=0)
    srv.add_step("write", targets=[t])
    _measured(scene)  # in the view: nothing to fix
    srv.update_object(t, x=19)
    srv.fix_scene()
    x = scene.get_obj(t).position[0]
    assert 12 - 7.1 < x < 12 + 7.1  # moved into the camera's view, not the default frame


# E2. fix_scene never reports a move it did not make
def test_fix_scene_reports_no_noop_moves(scene):
    pytest.importorskip("manim")
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["sin(x)"], x_range=[-5, 5]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="0"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=20)
    r = srv.fix_scene()
    assert not [c for c in r["result"]["changes"] if c.get("object") == d]


# E3. a curve undefined on its whole x_range is an error, not blank axes that pass
def test_curve_undefined_everywhere(scene):
    p = _id(srv.add_plot(functions=["sqrt(x)"], x_range=[-5, -1], y_range=[0, 3]))
    srv.add_step("write", targets=[p])
    r = _measured(scene)
    assert any("draws nothing" in o["summary"] for o in r["obligations"] if o["severity"] == "error")


# E4. axis labels are ink a label must not sit on
def test_text_on_axis_label_collides(scene):
    p = _id(srv.add_plot(functions=["0.2*x"], x_label=r"\text{time (s)}", y_label=r"\text{velocity}"))
    t = _id(srv.add_text("NOTE", x=4.9, y=0.45, font_size=40))
    srv.add_step("write", targets=[p, t])
    r = _measured(scene)
    assert any(o["lane"] == "collision" for o in r["obligations"])


# E5. a faded object may move off-screen unseen
def test_faded_object_not_framed(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["sin(x - t)"], x_range=[-5, 5]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="0"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=3)
    srv.add_step("fade", targets=[d])
    srv.add_step("tween", param="t", value=20)
    r = _measured(scene)
    assert not [o for o in r["obligations"] if o["lane"] == "framing" and o.get("target") == d
                and o["severity"] == "error"]


# E6. export is judged on the render's verdict, not on a later inspection
def test_export_uses_render_time_verdict(scene, monkeypatch):
    t = _id(srv.add_text("far", x=10))
    srv.add_step("write", targets=[t])
    proj = session.get_project()
    session.set_last_render(proj.id, scene.id, {
        "ok": True, "artifact": "/nonexistent.mp4", "fingerprint": srv._fingerprint(scene),
        "blocking": [{"severity": "error", "summary": "off-screen"}]})
    proj.obligations.clear()  # as if a later inspect_render of another state passed
    r = srv.export_gif()
    assert r["ok"] is False and "rendered" in r["error"]


# E7. explicit colors that vanish on the theme background are flagged
def test_low_contrast_color(scene):
    session.get_project().theme = "whiteboard"
    t = _id(srv.add_text("Key idea", color="WHITE"))
    srv.add_step("write", targets=[t])
    obls = inspect_scene(scene, theme_name="whiteboard")["obligations"]
    assert any("contrast" in o["summary"] for o in obls)
    assert not any("contrast" in o["summary"]
                   for o in inspect_scene(scene, theme_name="chalkboard")["obligations"])


# E8. a dot with an undefined anchor is reported, not a probe crash
def test_undefined_anchor_reported(scene):
    p = _id(srv.add_plot(functions=["x"]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="2", at_y="log(-2)"))
    srv.add_step("write", targets=[p, d])
    r = _measured(scene)
    assert not any(o["lane"] == "compile" for o in r["obligations"])
    assert any("no finite position" in o["summary"] for o in r["obligations"])


# E9. a comprehension variable named like a parameter is not that parameter
def test_comprehension_variable_is_not_a_parameter(scene):
    from chalkcast.codegen import uses_params
    assert not uses_params("sum(sin(k*x)/k for k in range(1, 9, 2))", {"k": 1.0})
    assert uses_params("sum(sin(j*x)/k for j in range(1, 9, 2))", {"k": 1.0})


# minor: stale obligations and readable camera steps
def test_inspect_project_marks_stale_and_decodes_camera(scene):
    t = _id(srv.add_text("far", x=10))
    srv.add_step("camera_move", zoom=2)
    srv.add_step("write", targets=[t])
    srv.inspect_render()
    srv.update_object(t, x=0)
    r = srv.inspect_project()
    assert r["obligations"] and all(o.get("stale") for o in r["obligations"])
    cam = r["result"]["scenes"][0]["steps"][0]
    assert cam["zoom"] == 2.0 and cam["focus"] is None


# ---- seventh audit --------------------------------------------------------------

# F1a. a dot first shown after a tween is judged where the tween left it
def test_dot_shown_after_tween_not_framed_at_start(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["0*x"], x_range=[-5, 5], y_range=[-3, 3]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t-30", at_y="0"))
    srv.add_step("write", targets=[p])
    srv.add_step("tween", param="t", value=30)
    srv.add_step("write", targets=[d])
    r = _measured(scene)
    assert not [o for o in r["obligations"] if o["severity"] == "error"]


# F1b. a curve first shown once it is defined, or undefined only after its fade
@pytest.mark.parametrize("order", ["write_after", "fade_before"])
def test_curve_state_only_while_shown(scene, order):
    srv.add_parameter("a", 10 if order == "write_after" else 0)
    p = _id(srv.add_plot(functions=["sqrt(x - a)"], x_range=[0, 5], y_range=[0, 3]))
    t = _id(srv.add_text("Title", y=3.3))
    srv.add_step("write", targets=[t])
    if order == "write_after":
        srv.add_step("tween", param="a", value=0)
        srv.add_step("write", targets=[p])
    else:
        srv.add_step("write", targets=[p])
        srv.add_step("fade", targets=[p])
        srv.add_step("tween", param="a", value=10)
    r = _measured(scene)
    assert not any("draws nothing" in o["summary"] for o in r["obligations"])


# F1c. no collision with where a curve was before either object is shown
def test_no_collision_with_unseen_initial_curve(scene):
    srv.add_parameter("a", 1)
    p = _id(srv.add_plot(functions=["0.3*sin(x)+a"], x_range=[-5, 5], y_range=[-3, 3]))
    t = _id(srv.add_text("label", x=2.5, y=0.83, font_size=28))
    srv.add_step("tween", param="a", value=-2)
    srv.add_step("write", targets=[p, t])
    r = _measured(scene)
    assert not any(o["lane"] == "collision" for o in r["obligations"])


# F2. collision repair searches inside the camera view
def test_free_spot_inside_camera_view(scene):
    a = _id(srv.add_text("first label", x=14, y=0))
    b = _id(srv.add_text("second label", x=14, y=0.1))
    srv.add_step("camera_move", zoom=2, focus_x=14, focus_y=0)
    srv.add_step("write", targets=[a])
    srv.add_step("write", targets=[b])
    pytest.importorskip("manim")
    r = srv.fix_scene(max_rounds=1)
    moved = [c for c in r["result"]["changes"] if c["lane"] == "collision"]
    assert moved and all(14 - 3.6 < c["position"][0] < 14 + 3.6 for c in moved)


# F3. an area pushed off its axes by a redeclared parameter or a tween is an error
@pytest.mark.parametrize("how", ["redeclare", "tween"])
def test_area_rechecked_at_parameter_states(scene, how):
    srv.add_parameter("a", 0.5)
    p = _id(srv.add_plot(functions=["a*x**2"], x_range=[-3, 3], y_range=[0, 4]))
    ar = _id(srv.add_area(p, 0, 2))
    srv.add_step("write", targets=[p, ar])
    if how == "redeclare":
        srv.add_parameter("a", 5)
    else:
        srv.add_step("tween", param="a", value=3)
    obls = inspect_scene(scene)["obligations"]
    assert any(o["target"] == ar and "integral" in o["summary"] for o in obls
               if o["severity"] == "error")


# F4. both copies of the open obligations carry the stale flag
def test_stale_flag_in_both_copies(scene):
    t = _id(srv.add_text("far", x=10))
    srv.add_step("write", targets=[t])
    srv.inspect_render()
    srv.update_object(t, x=0)
    r = srv.inspect_project()
    assert all(o.get("stale") for o in r["result"]["open_obligations"])
