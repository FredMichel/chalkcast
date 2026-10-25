"""Scientific correctness: reference scenes whose drawn geometry is checked numerically.

Each scene is built exactly as it would be rendered (the generated module's
``_cc_build()``), then the Mobjects are read back in axes coordinates and
compared with known values: shaded areas against exact integrals, Riemann
rectangles against the sum, drawn curves against the function, and the Gibbs
overshoot against its theoretical limit (2/pi) Si(pi) ~= 1.17898.
"""

import importlib.util

import numpy as np
import pytest

pytest.importorskip("manim", reason="needs Manim to build the Mobjects")

from chalkcast.codegen import generate_module  # noqa: E402
from chalkcast.model import Obj, Project  # noqa: E402


def _build(tmp_path, objects):
    project = Project(id="sci", quality="low")
    scene = project.new_scene(title="sci")
    scene.objects += objects
    path = tmp_path / "sci_scene.py"
    path.write_text(generate_module(scene, project.theme), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_cc_sci", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._cc_build()


def _plot(oid, functions, xr, yr, **kw):
    return Obj(id=oid, kind="plot", position=(0, 0),
               payload={"functions": functions, "x_range": list(xr), "y_range": list(yr),
                        "width": kw.get("w", 10), "height": kw.get("h", 5)})


def _coords(ax, mob):
    pts = np.concatenate([m.points for m in mob.get_family() if len(m.points)])
    return np.array([ax.point_to_coords(p)[:2] for p in pts])


def _shoelace(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _drawn_vs_true(ax, graph, f):
    xy = _coords(ax, graph)
    return np.max(np.abs(xy[:, 1] - np.array([f(v) for v in xy[:, 0]])))


def test_area_equals_integral(tmp_path):
    # x^3 - x on [0, 1]: geometric area 1/4 (signed integral -1/4)
    ns = _build(tmp_path, [
        _plot("plot_1", [{"expr": "x**3 - x"}], (-1.3, 1.3), (-1, 1)),
        Obj(id="area_2", kind="area", position=(0, 0), payload={"plot": "plot_1", "x_min": 0, "x_max": 1}),
    ])
    area = _shoelace(_coords(ns["plot_1_ax"], ns["area_2"]))
    assert area == pytest.approx(0.25, rel=5e-3)


def test_born_rule_probability(tmp_path):
    # ground state of the infinite well on [0,1]: P(1/3 < x < 2/3) = 1/3 + sqrt(3)/(2 pi)
    ns = _build(tmp_path, [
        _plot("plot_1", [{"expr": "2*sin(pi*x)**2"}], (0, 1), (0, 2.2)),
        Obj(id="area_2", kind="area", position=(0, 0),
            payload={"plot": "plot_1", "x_min": 1 / 3, "x_max": 2 / 3}),
    ])
    p = _shoelace(_coords(ns["plot_1_ax"], ns["area_2"]))
    assert p == pytest.approx(1 / 3 + np.sqrt(3) / (2 * np.pi), rel=5e-3)  # 0.6090


def test_riemann_rectangles_sum(tmp_path):
    ns = _build(tmp_path, [
        _plot("plot_1", [{"expr": "sqrt(x)"}], (0, 4), (0, 2.2)),
        Obj(id="riemann_2", kind="riemann", position=(0, 0),
            payload={"plot": "plot_1", "x_min": 0, "x_max": 4, "dx": 0.5}),
    ])
    ax = ns["plot_1_ax"]
    rects = [_coords(ax, r)[::4] for r in ns["riemann_2"]]
    assert len(rects) == 8
    # each height is f at the left edge, exactly
    for x, c in zip(np.arange(0, 4, 0.5), rects):
        assert c[:, 1].max() == pytest.approx(np.sqrt(x), abs=1e-9)
        assert c[:, 0].min() == pytest.approx(x, abs=1e-9)
    # Manim draws each rectangle 0.1 % wider than dx (0.5005 for 0.5), so the drawn
    # total sits 0.1 % above the left sum: allow that, not more
    total = sum(_shoelace(c) for c in rects)
    left = sum(np.sqrt(x) * 0.5 for x in np.arange(0, 4, 0.5))
    assert total == pytest.approx(left, rel=1.5e-3)


@pytest.mark.parametrize("expr,f,xr,yr", [
    ("sin(x)", np.sin, (-6.3, 6.3), (-1.5, 1.5)),
    ("exp(-x**2/2)", lambda x: np.exp(-x * x / 2), (-3, 3), (0, 1.2)),
    ("sin(30*x)", lambda x: np.sin(30 * x), (0, 6), (-1.2, 1.2)),  # high frequency
    ("2*sin(3*pi*x)**2", lambda x: 2 * np.sin(3 * np.pi * x) ** 2, (0, 1), (0, 2.2)),
])
def test_drawn_curve_matches_function(tmp_path, expr, f, xr, yr):
    ns = _build(tmp_path, [_plot("plot_1", [{"expr": expr}], xr, yr)])
    err = _drawn_vs_true(ns["plot_1_ax"], ns["plot_1_g1"], f)
    assert err < 0.005 * (yr[1] - yr[0])


def test_gibbs_overshoot_is_the_true_one(tmp_path):
    # the partial sum's peak tends to (2/pi) Si(pi) = 1.17898; a spline would invent its own
    n = 25
    expr = f"4/pi*sum(sin((2*k+1)*x)/(2*k+1) for k in range({n}))"
    ns = _build(tmp_path, [_plot("plot_1", [{"expr": expr}], (-np.pi, np.pi), (-1.5, 1.5))])
    peak = _coords(ns["plot_1_ax"], ns["plot_1_g1"])[:, 1].max()
    xs = np.linspace(0, np.pi, 200001)
    true_peak = (4 / np.pi * sum(np.sin((2 * k + 1) * xs) / (2 * k + 1) for k in range(n))).max()
    assert peak == pytest.approx(true_peak, abs=2e-3)
    assert peak == pytest.approx(1.17898, abs=0.01)


def test_square_wave_has_no_spline_ringing(tmp_path):
    # square wave: the jumps are drawn as vertical edges, but away from them no drawn
    # point may leave the plateaus (|y| < 0.9) -- the fake "Gibbs" a smoothed spline adds
    ns = _build(tmp_path, [_plot("plot_1", [{"expr": "sign(sin(x))"}], (-6, 6), (-1.5, 1.5))])
    xy = _coords(ns["plot_1_ax"], ns["plot_1_g1"])
    mid = np.abs(xy[:, 1]) < 0.9
    near_jump = np.min(np.abs(xy[:, 0][:, None] - np.pi * np.arange(-2, 3)[None, :]), axis=1) < 0.05
    assert not np.any(mid & ~near_jump)
