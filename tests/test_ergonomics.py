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
    assert "self.play(Write(text_3, run_time=1.0), Indicate(text_1, color='#ff7eb6', run_time=1.0))" in code


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


# ---- eighth audit ---------------------------------------------------------------

# G1. a transformed object re-written after a fade shows its new shape
def test_rewrite_after_transform_keeps_new_shape(scene):
    a = _id(srv.add_text("ORIGINAL", y=2))
    b = _id(srv.add_text("REPLACED", y=-2))
    c = _id(srv.add_text("caption", y=-2))
    srv.add_step("write", targets=[a])
    srv.add_step("transform", targets=[a], to=b)
    srv.add_step("fade", targets=[a])
    srv.add_step("write", targets=[a])
    srv.add_step("write", targets=[c])
    r = _measured(scene)
    assert any(o["lane"] == "collision" for o in r["obligations"])


# G2. a camera that follows a moving dot keeps it framed
def test_camera_tracking_a_moving_dot(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["0*x"]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="1"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=20, duration=3)
    srv.add_step("camera_move", zoom=1, focus_x=16, focus_y=0, duration=3, with_previous=True)
    r = _measured(scene)
    assert not [o for o in r["obligations"] if o["lane"] == "framing" and o.get("target") == d
                and o["severity"] == "error"]


# G3. text on a moving plot's axes is caught even when first shown after a tween
def test_static_axes_checked_after_tween(scene):
    srv.add_parameter("a", 1)
    p = _id(srv.add_plot(functions=["a*x"]))
    t = _id(srv.add_text("label here", x=0, y=2))
    srv.add_step("tween", param="a", value=0.2)
    srv.add_step("write", targets=[p, t])
    r = _measured(scene)
    assert any(o["lane"] == "collision" for o in r["obligations"])


# G4. edits are not refused on parameter values the overlay is never shown at
def test_overlay_edits_follow_shown_values(scene):
    srv.add_parameter("a", 0.5)
    p = _id(srv.add_plot(functions=["a*x**2"], x_range=[-3, 3], y_range=[0, 4]))
    ar = _id(srv.add_area(p, 0, 2))
    srv.add_parameter("a", 5)
    srv.add_step("write", targets=[p])
    srv.add_step("tween", param="a", value=0.5)
    srv.add_step("write", targets=[ar])
    assert srv.update_object(ar, color="RED")["ok"]
    assert not any(o.get("target") == ar for o in inspect_scene(scene)["obligations"]
                   if o["severity"] == "error")
    assert srv.add_area(p, 0, 2)["ok"]  # y = 0 visible: still checked at add time


# G5. moving-curve collisions follow the shape a label shows after a transform
def test_moving_collision_uses_transformed_label(scene):
    srv.add_parameter("a", -2)
    p = _id(srv.add_plot(functions=["0*x+a"], x_range=[-5, 5], y_range=[-3, 3]))
    a = _id(srv.add_text("A", x=2.5, y=3.3))
    b = _id(srv.add_text("B", x=2.5, y=0.8))
    srv.add_step("write", targets=[p, a])
    srv.add_step("transform", targets=[a], to=b)
    srv.add_step("tween", param="a", value=2)
    r = _measured(scene)
    hits = [o for o in r["obligations"] if o["lane"] == "collision"]
    assert hits and all(o["target"] == b for o in hits)


# G6. zoom applies to a transform destination's legibility
def test_zoom_applies_to_transform_destination(scene):
    a = _id(srv.add_text("A", font_size=48))
    b = _id(srv.add_text("B", font_size=30))
    srv.add_step("camera_move", zoom=0.5)
    srv.add_step("write", targets=[a])
    srv.add_step("transform", targets=[a], to=b)
    assert any(o["lane"] == "legibility" and o["target"] == b for o in _lanes(scene, "warning"))


# G7. an anchored dot leaving its axes is flagged
def test_anchor_leaving_axes(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["x**2"], x_range=[0, 3], y_range=[0, 4]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="t**2"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=2.8)
    assert any("leaves the axes" in o["summary"] for o in _lanes(scene, "warning"))


# ---- ninth audit ----------------------------------------------------------------

# H1. moving curves/dots are checked against equation labels too
def test_moving_dot_crosses_equation(scene):
    srv.add_parameter("a", -4)
    eq = _id(srv.add_equation("v(t)", x=1.6, y=0.833))
    p = _id(srv.add_plot(functions=["0*x"]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="a", at_y="1", size=2))
    srv.add_step("write", targets=[eq, p, d])
    srv.add_step("tween", param="a", value=4, duration=2)
    r = _measured(scene)
    assert any(o["lane"] == "collision" and o["target"] == eq for o in r["obligations"])


# H2. a transform destination is reached through the object that shows it
def test_transform_destination_targets_point_to_source(scene):
    a, b = _id(srv.add_text("x^2-1")), _id(srv.add_text("(x-1)(x+1)"))
    srv.add_step("write", targets=[a])
    srv.add_step("transform", targets=[a], to=b)
    srv.add_step("fade", targets=[b])
    errs = [o["suggested_fix"] for o in _lanes(scene) if o["lane"] == "timing"]
    assert any(f"target '{a}'" in f for f in errs)


def test_writing_a_shown_destination_warns(scene):
    a, b = _id(srv.add_text("x^2-1")), _id(srv.add_text("(x-1)(x+1)"))
    srv.add_step("write", targets=[a])
    srv.add_step("transform", targets=[a], to=b)
    srv.add_step("write", targets=[b])
    assert any("duplicate" in o["summary"] for o in _lanes(scene, "warning"))


# H3. overlays and anchored shapes left without their plot
def test_overlay_without_its_plot(scene):
    p = _id(srv.add_plot(functions=["x**2/4"], x_range=[-3, 3], y_range=[0, 3]))
    ar = _id(srv.add_area(p, 0, 2))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="2", at_y="1"))
    srv.add_step("write", targets=[p])
    srv.add_step("write", targets=[ar, d])
    srv.add_step("fade", targets=[p])
    floating = {o["target"] for o in _lanes(scene, "warning") if "without its plot" in o["summary"]}
    assert floating == {ar, d}


# H4. camera moves are eased like Manim's (smooth), tweens linear
def test_camera_easing_matches_manim():
    from chalkcast.verify import smooth
    assert smooth(0) == 0 and smooth(1) == 1 and abs(smooth(0.5) - 0.5) < 1e-9
    assert smooth(0.25) < 0.2  # eased: well behind a linear move early on


def test_long_tracking_pan_is_flagged(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["0*x"]))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="0", size=2))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=60, duration=4)
    srv.add_step("camera_move", zoom=1, focus_x=48, focus_y=0, duration=4, with_previous=True)
    r = _measured(scene)
    assert any(o["lane"] == "framing" and o.get("target") == d for o in r["obligations"]
               if o["severity"] == "error")


# H5. a curve following an un-tweened parameter is still refused at add time
def test_area_refused_for_untweened_parameter(scene):
    srv.add_parameter("a", 5)
    p = _id(srv.add_plot(functions=["a*x**2"], y_range=[0, 4]))
    assert srv.add_area(p, 0, 2)["ok"] is False


# ---- tenth audit ----------------------------------------------------------------

# J1. two animations of one object (or two camera moves) in one beat are refused
@pytest.mark.parametrize("second", ["highlight", "fade"])
def test_two_animations_of_one_object_in_a_beat(scene, second):
    t = _id(srv.add_text("Hello world"))
    srv.add_step("write", targets=[t], duration=2)
    r = srv.add_step(second, targets=[t], with_previous=True)
    assert r["ok"] is False and "only the last" in r["error"]
    assert len(scene.steps) == 1


def test_two_camera_moves_in_a_beat(scene):
    t = _id(srv.add_text("x"))
    srv.add_step("write", targets=[t])
    srv.add_step("camera_move", zoom=2, focus_x=-4, focus_y=0)
    r = srv.add_step("camera_move", zoom=3, with_previous=True)
    assert r["ok"] is False and "camera" in r["error"]


# J2. collision repair keeps the earlier label (the title) and leaves a gap
def test_fix_moves_the_later_label(scene):
    title = _id(srv.add_text("Chain rule", y=3.2, font_size=44))
    eq = _id(srv.add_equation(r"\frac{d}{dx}\sin(x^2)=2x\cos(x^2)", y=2.7))
    srv.add_step("write", targets=[title])
    srv.add_step("write", targets=[eq])
    pytest.importorskip("manim")
    r = srv.fix_scene()
    moved = {c["object"] for c in r["result"]["changes"] if c["lane"] == "collision"}
    assert moved == {eq} and scene.get_obj(title).position == (0, 3.2)


# J3. a curve undefined only where a tween lands says so
def test_undefined_at_tween_landing_names_the_value(scene):
    srv.add_parameter("h", 2)
    p = _id(srv.add_plot(functions=["x**2/2", "((1+h)**2/2-0.5)/h*(x-1)+0.5"]))
    srv.add_step("write", targets=[p])
    srv.add_step("tween", param="h", value=0)
    r = _measured(scene)
    msg = [o for o in r["obligations"] if "draws nothing" in o["summary"]]
    assert msg and "h=0" in msg[0]["summary"] and "tween" in msg[0]["suggested_fix"]


# J4. the missing tick at an axis tip is stated, not implied by the echoed range
def test_axis_max_tick_is_documented(scene):
    r = srv.add_plot(functions=["x**2"], x_range=[0, 2], y_range=[0, 4], show_numbers=True)
    assert any("x_max" in note for note in r["limitations"])


# J5. a curve touching a label (within half a stroke) collides
def test_inside_pads_by_half_a_stroke():
    from chalkcast.geometry import BBox
    from chalkcast.verify import _inside
    box = BBox(cx=4.5, cy=1.5, width=2.0, height=0.6)  # left edge at 3.5
    assert _inside([3.49, 1.5], box)


# ---- eleventh audit -------------------------------------------------------------

# K1. zooming in on a detail during a tween crops by design
def test_zoom_in_during_tween_is_not_an_error(scene):
    srv.add_parameter("w", 2)
    p = _id(srv.add_plot(functions=["cos(w*x)"], x_range=[0, 10], width=10))
    srv.add_step("write", targets=[p])
    srv.add_step("tween", param="w", value=6, duration=2)
    srv.add_step("camera_move", zoom=2, focus_x=-3, focus_y=0, with_previous=True)
    srv.add_step("camera_move", zoom=1)
    r = _measured(scene)
    assert not [o for o in r["obligations"] if o["lane"] == "framing" and o["severity"] == "error"]


# K2. a swap within one beat has no in-between state
def test_swap_in_one_beat_is_not_a_collision(scene):
    a, b = _id(srv.add_text("3", x=-3)), _id(srv.add_text("1", x=-1))
    a2, b2 = _id(srv.add_text("3", x=-1)), _id(srv.add_text("1", x=-3))
    srv.add_step("write", targets=[a, b])
    srv.add_step("transform", targets=[a], to=a2)
    srv.add_step("transform", targets=[b], to=b2, with_previous=True)
    assert not [o for o in inspect_scene(scene)["obligations"] if o["lane"] == "collision"]


# K3. a crossfade gives the same verdict in either step order
@pytest.mark.parametrize("fade_first", [True, False])
def test_crossfade_order_does_not_matter(scene, fade_first):
    n1, n2 = _id(srv.add_text("factor", y=-1.2)), _id(srv.add_text("simplify", y=-1.2))
    srv.add_step("write", targets=[n1])
    first, second = (("fade", n1), ("write", n2)) if fade_first else (("write", n2), ("fade", n1))
    srv.add_step(first[0], targets=[first[1]])
    srv.add_step(second[0], targets=[second[1]], with_previous=True)
    assert not [o for o in inspect_scene(scene)["obligations"] if o["lane"] == "collision"]


# K4. fading an already-faded object is an error
def test_fade_of_faded_object(scene):
    a, b = _id(srv.add_text("Step one", y=1)), _id(srv.add_text("Step two", y=-1))
    srv.add_step("write", targets=[a, b])
    srv.add_step("fade", targets=[a])
    srv.add_step("wait")
    srv.add_step("fade", targets=[a, b])
    assert any("already faded" in o["summary"] for o in _lanes(scene))


# K5. overlays stay on a plot's old shape after it is transformed
def test_overlay_after_plot_transform(scene):
    p1 = _id(srv.add_plot(functions=["x**2"], x_range=[0, 2], y_range=[0, 4]))
    p2 = _id(srv.add_plot(functions=["x**3"], x_range=[0, 2], y_range=[0, 8]))
    ar = _id(srv.add_area(p1, 0, 1.5))
    srv.add_step("write", targets=[p1])
    srv.add_step("write", targets=[ar])
    srv.add_step("transform", targets=[p1], to=p2)
    assert any(o["target"] == ar and "transformed" in o["summary"] for o in _lanes(scene, "warning"))


# K6. a dot sweeping across a small label between samples is caught
def test_swept_dot_path_crosses_label(scene):
    srv.add_parameter("t", -5)
    p = _id(srv.add_plot(functions=["0*x"], x_range=[-5, 5], y_range=[-2.5, 2.5], width=8, height=5))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="1.25"))
    e = _id(srv.add_equation("P", x=0.5, y=1.25, scale=0.7))
    srv.add_step("write", targets=[p, d, e])
    srv.add_step("tween", param="t", value=5, duration=4)
    r = _measured(scene)
    assert any(o["lane"] == "collision" and o["target"] == e for o in r["obligations"])


# K7. fills that barely tint the board are flagged; closed ±sqrt curves reach their edges
def test_fill_contrast(scene):
    p = _id(srv.add_plot(functions=["x"], x_range=[0, 2], y_range=[0, 2]))
    rr = _id(srv.add_riemann(p, 0, 2, dx=0.5, color="WHITE"))
    srv.add_step("write", targets=[p, rr])
    obls = inspect_scene(scene, theme_name="whiteboard")["obligations"]
    assert any(o["target"] == rr and "barely shows" in o["summary"] for o in obls)


def test_sqrt_curve_reaches_domain_edge():
    pytest.importorskip("manim")
    from chalkcast import codegen
    ns: dict = {}
    exec(codegen._PRELUDE, ns)
    ax = ns["Axes"](x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5])
    g = ns["_cc_plot"](ax, lambda x: ns["np"].sqrt(1 - x * x), ns["WHITE"])
    xs = [ax.point_to_coords(p)[0] for p in g.get_all_points()]
    assert min(xs) < -0.9999 and max(xs) > 0.9999


def test_scene_without_steps_still_measures(scene):
    _id(srv.add_text("NO STEPS"))
    r = _measured(scene)
    assert any("no steps" in o["summary"] for o in r["obligations"])


# K1b. a label written while zoomed in is judged in that view (held-out 4)
def test_label_written_during_zoom_is_framed_in_that_view(scene):
    t = _id(srv.add_text("intensity drops fast", x=-1.0, y=0.0, font_size=28))
    srv.add_step("camera_move", zoom=2.5, focus_x=-3, focus_y=-1)
    srv.add_step("write", targets=[t])
    r = _measured(scene)
    assert any(o["lane"] == "framing" and o["severity"] == "error" for o in r["obligations"])


# ---- twelfth audit --------------------------------------------------------------

# L1. LaTeX special characters are refused with the fix, not a misleading error
@pytest.mark.parametrize("tex", [r"r = 5% \text{ per year}", r"\text{50% off}", "a # b"])
def test_tex_specials_refused(scene, tex):
    r = srv.add_equation(tex)
    assert r["ok"] is False and "\\" in r["error"] and "unescaped" in r["error"]
    assert srv.add_equation(r"r = 5\% \text{ per year}")["ok"]


def test_undefined_macro_names_the_equation(scene):
    pytest.importorskip("manim")
    e = _id(srv.add_equation(r"\undefinedcmd x"))
    srv.add_step("write", targets=[e])
    r = srv.inspect_render()  # a compile failure: geometry is not measured
    comp = [o for o in r["obligations"] if o["lane"] == "compile"]
    assert comp and e in comp[0]["summary"] and "Undefined control sequence" in comp[0]["summary"]


# L2. periodic motion does not alias with the tween samples
def test_periodic_dot_leaving_frame_is_caught(scene):
    import math
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["3*sin(4*x)"], x_range=[-4, 4], y_range=[-3, 3],
                         width=8, height=6, x=-2))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="3.5", at_y="5*sin(t)"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=8 * math.pi, duration=4)
    r = _measured(scene)
    assert any(o["lane"] == "framing" and o.get("target") == d for o in r["obligations"]
               if o["severity"] == "error")


# L3. a label struck through by an axis line between ticks
def test_label_on_axis_line(scene):
    p = _id(srv.add_plot(functions=["0.5*x+2"], x_range=[-6, 6, 3], y_range=[-1, 5, 1],
                         width=12, height=5, y=-0.5))
    a = _id(srv.add_text("a", x=-2.2, y=-2.1667))
    srv.add_step("write", targets=[p, a])
    r = _measured(scene)
    assert any(o["lane"] == "collision" and a in (o.get("target"), o.get("other"))
               for o in r["obligations"])


# L5. a fill written after a label paints over it
def test_fill_over_label(scene):
    p = _id(srv.add_plot(functions=["4 - x**2/4"], x_range=[0, 4], y_range=[0, 5]))
    lab = _id(srv.add_equation(r"\int_0^4 f\,dx \approx 10.7", x=-1.2, y=-0.5, scale=0.8))
    rr = _id(srv.add_riemann(p, 0, 4, 0.25))
    srv.add_step("write", targets=[p])
    srv.add_step("write", targets=[lab])
    srv.add_step("write", targets=[rr])
    r = _measured(scene)
    assert any("painted over" in o["summary"] for o in r["obligations"])


# L6. tick numbers crowded on a short axis
def test_crowded_tick_numbers(scene):
    p = _id(srv.add_plot(x_range=[0, 16, 1], y_range=[-1, 1], width=12, height=0.4,
                         show_numbers=True))
    srv.add_step("write", targets=[p])
    assert any("overlap on its y-axis" in o["summary"] for o in _lanes(scene, "warning"))


# L7. tick numbers: no thousands separators, integral steps stay integers
def test_tick_number_format(scene):
    p = _id(srv.add_plot(functions=["x"], x_range=[2000, 2030, 5], y_range=[0, 5],
                         show_numbers=True))
    assert srv.inspect_project()["result"]["scenes"][0]["objects"][0]["payload"]["y_range"][2] == 1
    code = generate_module(scene, "chalkboard")
    assert "'group_with_commas': False" in code and "'num_decimal_places': 0" in code


# L8. close_project says what it discards
def test_close_project_reports_discarded(scene):
    r = srv.close_project()
    assert r["result"]["discarded"]["scenes"] == 1 and "files_kept" in r["result"]


# ---- thirteenth audit -----------------------------------------------------------

# M1. an edit that keeps the module the same size is measured, not a stale .pyc
def test_same_size_edit_is_measured_fresh(scene):
    e = _id(srv.add_equation("x^2+1", scale=1.5))
    srv.add_step("write", targets=[e])
    _measured(scene)
    srv.update_object(e, scale=2.5)  # same source length, same second
    from chalkcast import session as s
    proj = s.get_project()
    w1 = srv.inspect_render()
    box = scene.get_obj(e)
    from chalkcast import render
    meas = render.probe_scene(proj, scene, s.workdir(proj.id, scene.id))
    l, r, _, _ = meas["objects"][e]["bbox"]
    assert r - l > 2.0  # at scale 2.5 (1.96 wide at 1.5)
    assert w1["ok"] and box.scale == 2.5


# M2. misspelled argument names are rejected, not dropped
def test_unknown_argument_names_are_rejected(scene):
    import asyncio
    for name, args in (("add_shape", {"shape": "circle", "radius": 2}),
                       ("add_text", {"text": "hi", "position": [2, 2]}),
                       ("add_plot", {"functions": ["x"], "xrange": [0, 10]})):
        with pytest.raises(Exception) as exc:
            asyncio.run(srv.mcp.call_tool(name, args))
        assert list(args)[-1] in str(exc.value)
    schema = srv.mcp._tool_manager._tools["add_text"].parameters
    assert schema.get("additionalProperties") is False


# M3. framing repair stops at the readable size instead of fighting legibility
def test_fix_does_not_shrink_text_below_readable(scene):
    t = _id(srv.add_text("A very long paragraph " * 9, font_size=22))
    srv.add_step("write", targets=[t])
    pytest.importorskip("manim")
    r = srv.fix_scene()
    o = scene.get_obj(t)
    assert 22 * o.scale >= 20 - 1e-6 and o.payload["font_size"] == 22
    assert any("too long to fit" in why for why in r["result"]["not_auto_fixable"])


# M4. side-by-side labels whose letters run into each other
def test_side_by_side_labels_touching(scene):
    a = _id(srv.add_text("Quite long label A", x=-3, y=-2))
    b = _id(srv.add_text("Label B", x=-0.5, y=-2))
    srv.add_step("write", targets=[a, b])
    r = _measured(scene)
    assert any(o["lane"] == "collision" for o in r["obligations"])


# M5. wrong scene ids list the existing ones; transforming into itself is refused
def test_wrong_scene_id_lists_scenes(scene):
    r = srv.add_text("x", scene_id="scene:1")
    assert r["ok"] is False and "scene:0001" in r["error"]


def test_transform_into_itself_refused(scene):
    c = _id(srv.add_text("c"))
    srv.add_step("write", targets=[c])
    assert srv.add_step("transform", targets=[c], to=c)["ok"] is False


# ---- fourteenth audit (over the real MCP protocol) --------------------------------

# N1. a scene ending on an animation holds its final state instead of cutting it
def test_scene_ending_on_a_play_holds_the_end(scene):
    t = _id(srv.add_text("END"))
    srv.add_step("write", targets=[t])
    assert generate_module(scene, "chalkboard").rstrip().splitlines()[-1].strip() == "self.wait(0.5)"
    assert scene.duration() == 1.5
    srv.add_step("wait", duration=1)
    assert "self.wait(0.5)" not in generate_module(scene, "chalkboard")


# N2. zoom=1 keeps the centre: ending zoomed away from an object is flagged
def test_scene_ending_cropped_by_the_camera(scene):
    a = _id(srv.add_text("LEFT", x=-5))
    b = _id(srv.add_text("RIGHT", x=3))
    srv.add_step("write", targets=[a, b])
    srv.add_step("camera_move", zoom=2, focus_x=3, focus_y=0)
    assert not any("final view" in o["summary"] for o in _lanes(scene, "warning"))  # a choice
    srv.add_step("camera_move", zoom=1)  # back to frame size, but still centred on x=3
    assert any("final view" in o["summary"] and o["target"] == a for o in _lanes(scene, "warning"))
    srv.add_step("camera_move", zoom=1, focus_x=0, focus_y=0)
    assert not any("final view" in o["summary"] for o in _lanes(scene, "warning"))


# N3/N4. axis labels: LaTeX specials and plain words are refused at add time
def test_axis_labels_checked(scene):
    assert srv.add_plot(functions=["x"], x_label="50% load")["ok"] is False
    r = srv.add_plot(functions=["x"], x_label="time (s)")
    assert r["ok"] is False and "\\text{time (s)}" in r["error"]
    p = _id(srv.add_plot(functions=["x"], x_label="\\text{time (s)}", y_label="v_x"))
    assert srv.update_object(p, y_label="speed of light")["ok"] is False


def test_axis_label_latex_errors_are_measured(scene):
    p = _id(srv.add_plot(functions=["x"], y_label="\\frac{a}"))
    srv.add_step("write", targets=[p])
    pytest.importorskip("manim")
    r = srv.inspect_render()
    assert any("y_label" in o["summary"] for o in r["obligations"] if o["severity"] == "error")


# N5. refused calls are errors on the wire, dicts in Python
def test_refusals_are_mcp_errors(scene):
    import asyncio
    res = asyncio.run(srv.mcp.call_tool("remove_object", {"object_id": "nope"}))
    res = res if not isinstance(res, tuple) else res[0]
    assert getattr(res, "is_error", False) is True
    assert srv.remove_object("nope")["ok"] is False


# N6. render_frame says it is a preview, not a verification
def test_render_frame_is_not_a_verification(scene):
    pytest.importorskip("manim")
    r = srv.render_frame()
    assert any("not verified" in n for n in r["limitations"])
    assert any("no steps" in n for n in r["limitations"])


# N7. an object id from another scene points there
def test_unknown_object_names_its_scene(scene):
    t = _id(srv.add_text("A"))
    srv.add_scene("second")
    r = srv.update_object(t, x=1)
    assert r["ok"] is False and "scene:0001" in r["error"] and "scene_id" in r["error"]



# ---- fifteenth audit --------------------------------------------------------------

# P1. a wide axis label pushed over the axis tip, struck through by its own curve
def test_curve_strikes_its_own_wide_label(scene):
    p = _id(srv.add_plot(functions=[{"expr": "x - x**2/10"}], x_range=[0, 10], y_range=[0, 3],
                         width=10, height=3.5, x_label="\\text{distance (m)}"))
    srv.add_step("write", targets=[p])
    r = _measured(scene)
    assert any("own axis label" in o["summary"] for o in r["obligations"] if o["severity"] == "error")


# P2. a refusal naming something another scene has says where it is
def test_wrong_scene_hint_everywhere(scene):
    p = _id(srv.add_plot(functions=["x"]))
    srv.add_parameter("t", 0)
    srv.add_scene("second")
    for r in (srv.add_area(plot_id=p),
              srv.add_shape("dot", plot_id=p, at_x="1", at_y="1"),
              srv.add_step("write", targets=[p]),
              srv.add_plot(functions=["sin(x - t)"]),
              srv.add_step("tween", param="t", value=1)):
        assert r["ok"] is False and "scene:0001" in r["error"] and "pass scene_id" in r["error"], r


# P3. an exported GIF is never overwritten by a later export
def test_export_does_not_overwrite(tmp_path):
    pytest.importorskip("manim")
    from chalkcast import render
    if not render.have_ffmpeg():
        pytest.skip("no ffmpeg")
    import subprocess
    vid = tmp_path / "v.mp4"
    subprocess.run([render.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.3",
                    str(vid)], capture_output=True)
    a = render.export_gif(vid, tmp_path)["artifact"]
    b = render.export_gif(vid, tmp_path, width=32)["artifact"]
    assert a != b and (tmp_path / "v.gif").exists()


# P4/P5. durations are seconds between 0.2 and 60
def test_duration_bounds(scene):
    t = _id(srv.add_text("A"))
    assert srv.add_step("write", targets=[t], duration=0.05)["ok"] is False
    r = srv.add_step("write", targets=[t], duration=1500)
    assert r["ok"] is False and "milliseconds" in r["error"]
    assert srv.add_step("write", targets=[t], duration=0.2)["ok"]


# P6. the default highlight colour is not a default curve colour
def test_highlight_differs_from_curves():
    from chalkcast.codegen import CURVE_COLORS
    from chalkcast.themes import theme
    assert theme("chalkboard")["accent"] not in CURVE_COLORS


# P7. '$' in an equation is refused with the reason
def test_dollar_in_equation(scene):
    r = srv.add_equation("$x^2$ is the area")
    assert r["ok"] is False and "already math" in r["error"]
    assert srv.add_equation("\\$5")["ok"]


# ---- sixteenth audit --------------------------------------------------------------

# Q1. tick numbers are drawn in the theme ink, not Manim's white
def test_tick_numbers_in_theme_ink():
    srv.open_project(theme="whiteboard")
    srv.add_scene("t")
    sc = session.get_project().get_scene(None)
    _id(srv.add_plot(functions=["x"], x_range=[0, 4], y_range=[0, 4], show_numbers=True))
    from chalkcast.themes import theme
    code = generate_module(sc, "whiteboard")
    assert code.count(f"'color': '{theme('whiteboard')['ink']}'") >= 2 or "'group_with_commas': False, 'color'" in code


# Q2. the highlight pulse is judged for framing
def test_highlight_pulse_cut_at_edge(scene):
    t = _id(srv.add_text("Highlight me", x=5.95, y=-3.4, font_size=30))
    srv.add_step("write", targets=[t])
    srv.add_step("highlight", targets=[t], duration=2)
    r = _measured(scene)
    assert any("highlight at step 2" in o["summary"] for o in r["obligations"] if o["severity"] == "error")


# Q3. an anchored dot leaving the frame names the cause; fix_scene says why it does not move it
def test_anchored_offscreen_names_cause(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=[{"expr": "exp(-0.3*x)*cos(3*x)"}], x_range=[0, 10],
                         y_range=[-1.2, 1.2], width=10, height=4, y=-0.8))
    d = _id(srv.add_shape("dot", plot_id=p, at_x="t", at_y="exp(-0.3*t)*cos(3*t)"))
    srv.add_step("write", targets=[p, d])
    srv.add_step("tween", param="t", value=13, duration=4)
    r = _measured(scene)
    err_ = [o for o in r["obligations"] if o["severity"] == "error" and o.get("target") == d]
    assert err_ and "leaves the axes" in err_[0]["summary"]
    assert "fix_scene can do" not in err_[0]["suggested_fix"]
    f = srv.fix_scene()["result"]
    assert any(d in why for why in f["not_auto_fixable"])


# Q4. explicit segments drawn on a plot's axes move with it
def test_fix_moves_vectors_with_their_plot(scene):
    p = _id(srv.add_plot(x_range=[-4, 4], y_range=[-3, 3], width=10, height=6, x=2.5))
    a = _id(srv.add_shape("arrow", x1=2.5, y1=0, x2=5.0, y2=2))
    srv.add_step("write", targets=[p, a])
    pytest.importorskip("manim")
    r = srv.fix_scene()["result"]
    moved = scene.get_obj(p).position[0] - 2.5
    if abs(moved) > 1e-3:
        assert scene.get_obj(a).payload["x1"] == pytest.approx(2.5 + moved, abs=1e-3)
        assert any(c["object"] == a for c in r["changes"])


# Q5. a closed project id says so and lists the open ones
def test_closed_project_id():
    a = srv.open_project()["result"]["project_id"]
    b = srv.open_project()["result"]["project_id"]
    srv.close_project(b)
    r = srv.add_text("x", project_id=b)
    assert r["ok"] is False and "closed or unknown" in r["error"] and a in r["error"]
    assert srv.close_project(a)["result"]["files_kept"] is None


# Q6. overlays take their plot's new size
def test_overlay_size_follows_plot(scene):
    p = _id(srv.add_plot(functions=["x"], width=6, height=4))
    ar = _id(srv.add_area(plot_id=p, x_min=0, x_max=1))
    srv.update_object(p, width=3, height=2)
    assert scene.get_obj(ar).payload.get("width", 3) == 3


# Q7. a tween whose only user is never shown is named
def test_idle_tween_named(scene):
    srv.add_parameter("t", 0)
    p = _id(srv.add_plot(functions=["sin(x - t)"]))
    tx = _id(srv.add_text("hi", y=3))
    srv.add_step("write", targets=[tx])
    srv.add_step("tween", param="t", value=1)
    assert any("change nothing on screen" in o["summary"] and o["target"] == p
               for o in _lanes(scene, "warning"))
