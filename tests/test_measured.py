"""Measured verification (geometry probe), auto-fix, and the correction-loop budget.

These need Manim (+ LaTeX for equations); they are skipped elsewhere.
"""

import shutil

import pytest

manim = pytest.importorskip("manim", reason="needs Manim for the geometry probe")
if shutil.which("latex") is None:
    pytest.skip("needs LaTeX", allow_module_level=True)

import chalkcast.server as srv  # noqa: E402
from chalkcast import bench  # noqa: E402


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("CHALKCAST_HOME", str(tmp_path))


def test_benchmark_meets_verdict_targets(tmp_path):
    res = bench.run(measured=True, workdir=tmp_path)
    bad = [r for r in res["rows"] if r["false_positive"] or r["detected"] is False]
    assert res["detection_rate"] >= 0.95, bad
    assert res["false_positive_rate"] < 0.05, bad


@pytest.mark.parametrize("suffix", ["", "2", "3", "4"])
def test_holdout_benchmark_meets_verdict_targets(tmp_path, suffix):
    # cases labelled by authors who never read the verifier (bench_holdout*.py);
    # their first runs are the out-of-sample figures in the README
    import importlib
    cases = importlib.import_module(f"chalkcast.bench_holdout{suffix}").CASES
    res = bench.run(measured=True, workdir=tmp_path, cases=cases)
    bad = [r for r in res["rows"] if r["false_positive"] or r["detected"] is False]
    assert res["detection_rate"] >= 0.95, bad
    assert res["false_positive_rate"] < 0.05, bad


def test_real_false_positive_is_gone():
    # the equation that blocked export in real use is measured, not guessed
    srv.open_project()
    srv.add_scene()
    eq = srv.add_equation(r"\frac{d}{dx}\sin(x) = \cos(x)", x=3.6, y=-1, scale=0.7)["result"]["object_id"]
    srv.add_step("write", targets=[eq])
    r = srv.inspect_render()
    assert r["coverage"]["measured"] is True
    assert r["verdict"]["errors"] == 0 and r["verdict"]["warnings"] == 0


def test_compile_errors_caught_without_rendering():
    srv.open_project()
    srv.add_scene()
    eq = srv.add_equation(r"\frac{1}{2")["result"]["object_id"]
    srv.add_step("write", targets=[eq])
    r = srv.render_scene()
    assert r["ok"] is False and "render skipped" in r["error"]
    assert any(o["lane"] == "compile" for o in r["obligations"])
    assert "render_ms" not in r["evidence"]["timing"]  # no slow render was attempted


def test_fix_scene_repairs_mechanical_problems():
    srv.open_project()
    srv.add_scene()
    t1 = srv.add_text("Titre beaucoup trop à droite", x=6.5, y=3)["result"]["object_id"]
    t2 = srv.add_text("Énergie cinétique", y=0.1)["result"]["object_id"]
    t3 = srv.add_text("Énergie potentielle", y=-0.1)["result"]["object_id"]
    p = srv.add_plot(functions=[{"expr": "sign(sin(pi*x/5))", "smooth": True, "dx": 0.1}],
                     x_range=[-7.5, 7.5], y_range=[-1.5, 1.5], width=9, height=3, y=-2)["result"]["object_id"]
    srv.add_step("write", targets=[t1, t2, t3, p])
    before = srv.inspect_render()
    lanes = {o["lane"] for o in before["obligations"] if o["severity"] == "error"}
    assert {"framing", "collision", "fidelity"} <= lanes
    fixed = srv.fix_scene()
    assert fixed["verdict"]["errors"] == 0, fixed["result"]["needs_author"]
    assert {c["lane"] for c in fixed["result"]["changes"]} >= {"framing", "collision", "fidelity"}


def _simple_request(kind: str):
    """What an agent typically authors on a first pass for a simple request."""
    srv.open_project()
    srv.add_scene()
    if kind == "derivative":
        a = srv.add_text("La dérivée de sin(x)", y=3.3, font_size=40)["result"]["object_id"]
        b = srv.add_plot(functions=[{"expr": "sin(x)", "color": "YELLOW"}, {"expr": "cos(x)", "color": "BLUE"}],
                         x_range=[-6.3, 6.3], y_range=[-1.5, 1.5], width=8, height=4.5, x=-2.3, y=-0.4)["result"]["object_id"]
        c = srv.add_equation(r"\frac{d}{dx}\sin(x) = \cos(x)", x=4.6, y=-1, scale=0.85)["result"]["object_id"]
        ids = [a, b, c]
    elif kind == "area":
        a = srv.add_text("f(x) = x³ − x  sur [0, 1]", y=2.9)["result"]["object_id"]
        b = srv.add_plot(functions=[{"expr": "x**3 - x"}], x_range=[-1.3, 1.3], y_range=[-1, 1],
                         width=9, height=4.8, y=-0.4)["result"]["object_id"]
        c = srv.add_area(b, 0, 1)["result"]["object_id"]
        ids = [a, b, c]
    else:  # gaussian with a label dropped on the curve and an equation off the edge
        a = srv.add_plot(functions=[{"expr": "exp(-x**2)"}], x_range=[-3, 3], y_range=[0, 1.2],
                         width=9, height=4)["result"]["object_id"]
        b = srv.add_text("densité gaussienne", x=0, y=0.6, font_size=34)["result"]["object_id"]
        c = srv.add_equation(r"\int_{-\infty}^{\infty} e^{-x^2}\,dx=\sqrt{\pi}", x=6, y=-3, scale=1.0)["result"]["object_id"]
        ids = [a, b, c]
    for i in ids:
        srv.add_step("write", targets=[i])


@pytest.mark.parametrize("kind", ["derivative", "area", "messy_gaussian"])
def test_simple_request_exports_within_two_iterations(kind):
    """The agent loop: author -> inspect -> (fix_scene) -> render -> export.
    'Iterations' = renders needed before the GIF exports; the budget is 2."""
    _simple_request(kind)
    renders = 0
    insp = srv.inspect_render()
    if insp["verdict"]["errors"]:
        srv.fix_scene()
    while renders < 2:
        renders += 1
        r = srv.render_scene()
        assert r["ok"], r.get("error")
        g = srv.export_gif()
        if g["ok"]:
            break
        srv.fix_scene()
    assert g["ok"], g.get("obligations")
    assert renders <= 2
