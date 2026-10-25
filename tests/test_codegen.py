"""Codegen produces syntactically valid, expectation-matching Manim modules."""

import pytest

from chalkcast.codegen import generate_module
from chalkcast.demo import build_demo_project, build_equation_scene
from chalkcast.model import Obj, Project


def _scene():
    return Project(id="p").new_scene()


def test_demo_module_is_valid_python():
    _, scene = build_demo_project()
    code = generate_module(scene, "chalkboard")
    # Must compile as Python (proves no broken codegen).
    compile(code, "<demo>", "exec")


def test_demo_module_contains_expected_constructs():
    _, scene = build_demo_project()
    code = generate_module(scene, "chalkboard")
    assert "from manim import *" in code
    assert f"class {scene.class_name()}(MovingCameraScene):" in code
    assert "Axes(" in code             # the plot
    assert "return x**2" in code       # the function, sampled by _cc_plot
    assert "_cc_plot(plot_1_ax, plot_1_f1" in code
    assert "Write(text_1" in code      # a write step
    assert "Indicate(" in code         # the highlight step
    assert "background_color" in code  # theme applied
    assert "def _cc_build():" in code  # objects built by the function the probe reuses


def test_equation_codegen():
    scene = build_equation_scene()
    code = generate_module(scene, "chalkboard")
    compile(code, "<eq>", "exec")
    assert "MathTex(" in code
    assert "config.tex_template = _cc_tex" in code  # Dirac macros available


def test_area_and_riemann_codegen():
    scene = _scene()
    scene.objects.append(Obj(id="plot_1", kind="plot", payload={
        "functions": [{"expr": "x**2", "color": "YELLOW"}], "x_range": [0, 2, 1],
        "y_range": [0, 4, 1], "width": 6, "height": 4}))
    scene.objects.append(Obj(id="area_2", kind="area", color="BLUE",
                             payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 1, "opacity": 0.6}))
    scene.objects.append(Obj(id="riemann_3", kind="riemann", color="TEAL",
                             payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 1, "dx": 0.1}))
    scene.add_step("write", targets=["plot_1"])
    scene.add_step("write", targets=["riemann_3"])
    scene.add_step("transform", targets=["riemann_3"], to="area_2")
    code = generate_module(scene, "chalkboard")
    compile(code, "<area>", "exec")
    # fills are built on an unclipped helper graph of the same function
    assert "plot_1_ax.get_area(_cc_helper_graph(plot_1_ax, plot_1_f1" in code
    assert "plot_1_ax.get_riemann_rectangles(_cc_helper_graph(plot_1_ax, plot_1_f1" in code
    assert "Create(riemann_3" in code  # overlays use Create, not Write


def test_plot_math_functions_usable_bare():
    # regression: generated code lacked numpy, so expr='sin(x)' crashed the render.
    scene = _scene()
    scene.objects.append(Obj(id="plot_1", kind="plot", payload={
        "functions": [{"expr": "sin(x)"}], "x_range": [-3, 3, 1], "y_range": [-1, 1, 1]}))
    scene.add_step("write", targets=["plot_1"])
    code = generate_module(scene, "dark")
    compile(code, "<p>", "exec")
    assert "from numpy import" in code
    assert "return sin(x)" in code


def test_plot_smooth_false_and_dx():
    # explicit overrides still reach the plot call
    scene = _scene()
    scene.objects.append(Obj(id="plot_1", kind="plot", payload={
        "functions": [{"expr": "np.sign(sin(x))", "smooth": False, "dx": 0.02}],
        "x_range": [-3, 3, 1], "y_range": [-1.5, 1.5, 0.5]}))
    scene.add_step("write", targets=["plot_1"])
    code = generate_module(scene, "dark")
    compile(code, "<p>", "exec")
    assert "smooth=False, dx=0.02" in code


def test_plot_empty_functions_gives_bare_axes():
    # regression: an explicit empty functions list must not inject a default parabola
    scene = _scene()
    scene.objects.append(Obj(id="plot_1", kind="plot", payload={
        "functions": [], "x_range": [-5, 5, 1], "y_range": [-3, 3, 1]}))
    code = generate_module(scene, "dark")
    compile(code, "<p>", "exec")
    assert "plot_1_g" not in code           # no graph drawn
    assert "VGroup(plot_1_ax)" in code      # axes only


def test_equation_backslashes_not_doubled():
    # regression: combining an r-prefix with repr() doubled every backslash,
    # turning \int into \\int (a LaTeX line break) and garbling the math.
    tex = r"\int_0^1 x^2 \, dx"
    scene = _scene()
    scene.objects.append(Obj(id="equation_1", kind="equation", payload={"tex": tex}))
    code = generate_module(scene, "dark")
    assert "MathTex(r" not in code   # no raw-string prefix on the tex
    assert repr(tex) in code         # the correctly-escaped literal is present


def test_shape_and_camera_codegen():
    scene = _scene()
    scene.objects.append(Obj(id="shape_1", kind="shape", position=(0, 0),
                             payload={"shape": "square", "size": 2}))
    scene.add_step("write", targets=["shape_1"])
    scene.add_step("camera_move", to="2@1,1", duration=1.0)
    code = generate_module(scene, "dark")
    compile(code, "<shapes>", "exec")
    assert "Square(side_length=2" in code
    assert "self.camera.frame.animate" in code
    assert "Create(shape_1" in code  # shapes use Create, not Write


def test_with_previous_plays_steps_together():
    scene = _scene()
    scene.objects += [Obj(id="text_1", kind="text", payload={"text": "a"}),
                      Obj(id="text_2", kind="text", payload={"text": "b"}),
                      Obj(id="text_3", kind="text", payload={"text": "c"})]
    scene.add_step("write", targets=["text_1"])
    scene.add_step("transform", targets=["text_1"], to="text_2", duration=2)
    scene.add_step("write", targets=["text_3"], duration=0.5, with_previous=True)
    code = generate_module(scene, "dark")
    compile(code, "<wp>", "exec")
    assert "self.play(Transform(text_1, text_2, run_time=2), Write(text_3, run_time=0.5))" in code


def test_parameter_tween_redraws_dependents():
    scene = _scene()
    scene.params["t"] = 0.0
    scene.objects.append(Obj(id="plot_1", kind="plot", payload={
        "functions": [{"expr": "sin(x - t)"}], "x_range": [0, 10], "y_range": [-1, 1]}))
    scene.objects.append(Obj(id="area_2", kind="area", payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 2}))
    scene.objects.append(Obj(id="shape_3", kind="shape", payload={
        "shape": "dot", "size": 1, "anchor": {"plot": "plot_1", "x": "t", "y": "sin(0)"}}))
    scene.add_step("write", targets=["plot_1", "area_2", "shape_3"])
    scene.add_step("tween", param="t", value=6.28, duration=3)
    code = generate_module(scene, "dark")
    compile(code, "<tw>", "exec")
    assert "P['t'] = ValueTracker(0.0)" in code
    assert "t = P['t'].get_value()" in code
    assert "plot_1_g1 = always_redraw(" in code
    assert "area_2 = always_redraw(" in code
    assert "shape_3.add_updater(" in code
    assert "P['t'].animate(run_time=3, rate_func=linear).set_value(6.28)" in code


def test_axis_labels_and_numbers():
    scene = _scene()
    scene.objects.append(Obj(id="plot_1", kind="plot", payload={
        "functions": [{"expr": "x"}], "x_range": [0, 1], "y_range": [0, 1],
        "x_label": "x", "y_label": r"|\psi|^2", "show_numbers": True}))
    code = generate_module(scene, "dark")
    compile(code, "<lbl>", "exec")
    assert "get_axis_labels(" in code and "include_numbers" in code
    assert "VGroup(plot_1_ax, plot_1_g1, plot_1_labels)" in code


def test_line_segment_endpoints():
    scene = _scene()
    scene.objects.append(Obj(id="shape_1", kind="shape", payload={
        "shape": "line", "size": 1, "x1": 0, "y1": -1, "x2": 0, "y2": 1}))
    code = generate_module(scene, "dark")
    assert "Line([0, -1, 0], [0, 1, 0]" in code


def test_cc_plot_is_faithful_without_latex():
    """Runs the generated prelude: jumps split the curve, poles split, values clip."""
    pytest.importorskip("manim")
    import numpy as np

    from chalkcast.codegen import _PRELUDE
    ns: dict = {}
    exec(_PRELUDE, ns)  # noqa: S102 - our own generated code
    Axes = ns["Axes"]
    ax = Axes(x_range=[-4, 4], y_range=[-3, 3], x_length=8, y_length=5)
    sq = ns["_cc_plot"](ax, lambda x: np.sign(np.sin(x)), "#ffffff")
    assert len(sq.submobjects) == 4                  # 3 jumps (x = -pi, 0, pi) -> 4 steps
    assert all(p.use_smoothing is False for p in sq.submobjects)
    forced = ns["_cc_plot"](ax, lambda x: np.sign(np.sin(x)), "#ffffff", smooth=True, dx=0.1)
    assert forced.use_smoothing is True              # explicit overrides are honoured
    pole = ns["_cc_plot"](ax, lambda x: 1 / x, "#ffffff")
    assert len(pole.submobjects) == 2                # split at x = 0, clipped to the axes
    top = ax.c2p(0, 3)[1]
    assert pole.get_top()[1] <= top + 0.2
    with pytest.raises(NameError):                   # bugs surface, not blank plots
        ns["_cc_plot"](ax, lambda x: undefined_name(x), "#ffffff")  # noqa: F821
