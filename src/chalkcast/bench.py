"""Verdict benchmark: labelled scenes -> detection rate and false-positive rate.

Every case is a scene plus the set of lanes that *should* fire (empty = clean).
Clean cases include the real false positives met in PhD-level use (equation
framing, curves sharing axes, transform targets, annotations on plots); defect
cases cover each lane, including the silent visual bugs (spline overshoot on a
discontinuity, aliasing, a moving object leaving the frame).

    chalkcast bench            # measured (geometry probe; needs Manim + LaTeX)
    chalkcast bench --estimated

Scoring (per case, warnings and errors count; ``info`` does not):
* a defect case is *detected* if an obligation fires in one of its expected lanes;
* a case is a *false positive* if any obligation fires in a lane it does not expect.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .model import Obj, Project, Scene


@dataclass
class Case:
    name: str
    build: Callable[[Scene], None]
    expect: set[str] = field(default_factory=set)


def _t(i, text, x=0.0, y=0.0, fs=36, color=None, scale=1.0):
    return Obj(id=i, kind="text", position=(x, y), color=color, scale=scale,
               payload={"text": text, "font_size": fs})


def _eq(i, tex, x=0.0, y=0.0, scale=1.0, color=None):
    return Obj(id=i, kind="equation", position=(x, y), scale=scale, color=color, payload={"tex": tex})


def _plot(i, fns, xr, yr, x=0.0, y=0.0, w=8.0, h=5.0, **extra):
    return Obj(id=i, kind="plot", position=(x, y), payload={
        "functions": fns, "x_range": xr, "y_range": yr, "width": w, "height": h, **extra})


def _shape(i, shape, x=0.0, y=0.0, color=None, **payload):
    return Obj(id=i, kind="shape", position=(x, y), color=color, payload={"shape": shape, **payload})


def _show(s: Scene, *ids, d=1.0):
    s.add_step("write", targets=list(ids), duration=d)


# ---------------------------------------------------------------------------
# clean scenes
# ---------------------------------------------------------------------------

def c_demo(s):
    s.objects += [_t("text_1", "Area under a parabola", y=3.0, fs=40),
                  _plot("plot_1", [{"expr": "x**2", "color": "YELLOW"}], [0, 2, 1], [0, 4, 1], y=-0.8, w=6, h=4),
                  _shape("shape_1", "dot", 1.0, -0.8, "RED", size=1.0)]
    _show(s, "text_1", "plot_1", "shape_1")
    s.add_step("highlight", targets=["shape_1"])


def c_derivative(s):  # the equation that was wrongly blocked in real use
    s.objects += [_t("text_1", "La dérivée de sin(x)", y=2.9),
                  _plot("plot_2", [{"expr": "np.sin(x)", "color": "YELLOW"}, {"expr": "np.cos(x)", "color": "BLUE"}],
                        [-6.3, 6.3], [-1.5, 1.5], x=-2.5, y=-0.4, w=7, h=4),
                  _t("text_3", "sin(x)", 3.6, 1.2, 30, "YELLOW"), _t("text_4", "cos(x)", 3.6, 0.5, 30, "BLUE"),
                  _eq("equation_5", r"\frac{d}{dx}\sin(x) = \cos(x)", 3.6, -1, 0.7)]
    _show(s, "text_1", "plot_2", "text_3", "text_4", "equation_5")
    s.add_step("highlight", targets=["equation_5"])


def c_area_cubic(s):
    s.objects += [_t("text_1", "f(x) = x³ − x  sur [0, 1]", y=2.9),
                  _plot("plot_2", [{"expr": "x**3 - x", "color": "YELLOW"}], [-1.3, 1.3], [-1, 1], y=-0.4, w=9, h=4.8),
                  Obj(id="area_3", kind="area", position=(0, -0.4), color="BLUE",
                      payload={"plot": "plot_2", "fn": 1, "x_min": 0, "x_max": 1, "opacity": 0.5})]
    _show(s, "text_1", "plot_2", "area_3")


def _fourier(s, k, smooth=None, dx=None):
    f = {"expr": f"4/np.pi*sum(np.sin((2*k+1)*np.pi*x/5)/(2*k+1) for k in range({k}))", "color": "YELLOW"}
    if smooth is not None:
        f["smooth"] = smooth
    if dx is not None:
        f["dx"] = dx
    return f


def c_fourier_gibbs(s):  # overlays on shared axes + transforms + zoom
    s.objects += [_t("text_1", "Séries de Fourier → signal carré", y=3.1, fs=34),
                  _plot("plot_2", [{"expr": "np.sign(np.sin(np.pi*x/5))", "color": "GRAY"}], [-7.5, 7.5],
                        [-1.5, 1.5], y=-0.5, w=11, h=4)]
    for i, k in zip((3, 4, 5, 6), (1, 2, 5, 13)):
        s.objects.append(_plot(f"plot_{i}", [_fourier(s, k)], [-7.5, 7.5], [-1.5, 1.5], y=-0.5, w=11, h=4))
    for i, n in zip((7, 8, 9, 10), (1, 3, 9, 25)):
        s.objects.append(_t(f"text_{i}", f"N = {n}", 5, 2.1, 30, "YELLOW"))
    s.objects += [_shape("shape_11", "line", 1, 1.072, "RED", size=2), _t("text_12", "+9 %", 2.6, 1.072, 26, "RED"),
                  _t("text_13", "Phénomène de Gibbs : le dépassement (≈ 9 %) ne disparaît jamais", y=-3.1, fs=26)]
    _show(s, "text_1", "plot_2", "text_7", "plot_3")
    for lbl, pl in (("text_8", "plot_4"), ("text_9", "plot_5"), ("text_10", "plot_6")):
        s.add_step("transform", targets=["text_7"], to=lbl, duration=0.4)
        s.add_step("transform", targets=["plot_3"], to=pl, duration=1.5)
    _show(s, "shape_11", "text_12", "text_13")
    s.add_step("camera_move", to="2.5@0.4,0.3", duration=2)


def c_quantum_well(s):
    fns = [{"expr": f"{n*n} + 0*x", "color": "GRAY"} for n in (1, 2, 3)]
    fns += [{"expr": f"{n*n} + 0.8*np.sin({n}*np.pi*x/10)", "color": c} for n, c in ((1, "BLUE"), (2, "GREEN"), (3, "YELLOW"))]
    s.objects += [_t("text_1", "Particule dans un puits infini", y=3.2, fs=30),
                  _eq("equation_2", r"-\frac{\hbar^2}{2m}\frac{d^2\psi_n}{dx^2} = E_n\,\psi_n", y=2.3, scale=0.8),
                  _plot("plot_3", fns, [0, 10], [0, 11], x=-2.6, y=-1, w=6.5, h=4.4),
                  _t("text_4", "n = 1", 1.3, -2.8, 24, "BLUE"), _t("text_5", "n = 2", 1.3, -1.6, 24, "GREEN"),
                  _t("text_6", "n = 3", 1.3, 0.4, 24, "YELLOW"),
                  _eq("equation_7", r"E_n=\frac{n^2\pi^2\hbar^2}{2mL^2}", 3.6, -1, 0.6)]
    _show(s, "text_1", "equation_2", "plot_3", "text_4", "text_5", "text_6", "equation_7")


def c_born(s):
    s.objects += [_t("text_1", "Règle de Born : probabilité = aire sous |ψ|²", y=3.2, fs=30),
                  _eq("equation_2", r"P(0<x<L/3)=\int_0^{L/3}|\psi_2|^2\,dx\approx 0.402", y=2.2, scale=0.75),
                  _plot("plot_3", [{"expr": "0.2*np.sin(2*np.pi*x/10)**2", "color": "GREEN"}], [0, 10], [0, 0.25],
                        x=-1.5, y=-1, w=8, h=4),
                  _t("text_4", "quantique : 0,402", 4.6, 0.3, 24, "BLUE"),
                  _t("text_5", "classique : 0,333", 4.6, -0.3, 24, "GRAY"),
                  Obj(id="area_6", kind="area", position=(-1.5, -1), color="BLUE",
                      payload={"plot": "plot_3", "fn": 1, "x_min": 0, "x_max": 10 / 3, "opacity": 0.5})]
    _show(s, "text_1", "plot_3", "area_6", "equation_2", "text_4", "text_5")


def c_riemann(s):
    _riemann(s, eq_at=(4.6, 1.0))


def d_equation_touches_curve(s):
    # found by the probe: the curve's end runs into the integral's lower limit
    _riemann(s, eq_at=(4.5, 2.5))


def _riemann(s, eq_at):
    s.objects += [_plot("plot_1", [{"expr": "x**2", "color": "YELLOW"}], [0, 2, 1], [0, 4, 1], y=-0.5, w=7, h=5),
                  Obj(id="riemann_2", kind="riemann", color="TEAL",
                      payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 2, "dx": 0.25}),
                  Obj(id="riemann_3", kind="riemann", color="TEAL",
                      payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 2, "dx": 0.05}),
                  _eq("equation_4", r"\int_0^2 x^2\,dx = \tfrac{8}{3}", *eq_at, 0.8)]
    _show(s, "plot_1", "riemann_2", "equation_4")
    s.add_step("transform", targets=["riemann_2"], to="riemann_3", duration=2)


def c_square_auto(s):  # discontinuity handled automatically (no smooth flag)
    s.objects += [_plot("plot_1", [{"expr": "np.sign(np.sin(x))", "color": "GRAY"}, _fourier(s, 13)],
                        [-7.5, 7.5], [-1.5, 1.5], y=-0.5, w=11, h=4)]
    _show(s, "plot_1")


def c_pole_clipped(s):  # 1/x: split at the pole and clipped (info only)
    s.objects += [_plot("plot_1", [{"expr": "1/x", "color": "YELLOW"}], [-4, 4], [-3, 3], w=8, h=5)]
    _show(s, "plot_1")


def c_label_beside_curve(s):
    s.objects += [_plot("plot_1", [{"expr": "np.exp(-x**2)", "color": "YELLOW"}], [-3, 3], [0, 1.2], x=-1.5, w=8, h=4),
                  _t("text_2", "gaussienne", 4.5, 1.0, 30, "YELLOW")]
    _show(s, "plot_1", "text_2")


def c_quantum_beat(s):  # continuous parameter + anchored dot + simultaneous steps
    s.params["t"] = 0.0
    rho = ("0.1*(np.sin(np.pi*x/10)**2 + np.sin(2*np.pi*x/10)**2 "
           "+ 2*np.sin(np.pi*x/10)*np.sin(2*np.pi*x/10)*np.cos(t))")
    s.objects += [_t("text_1", "Battement quantique", y=3.2, fs=30),
                  _eq("equation_2", r"|\Psi(t)\rangle=\tfrac{1}{\sqrt{2}}\left(\ket{1}+e^{-i\omega_{21}t}\ket{2}\right)",
                      y=2.3, scale=0.7),
                  _plot("plot_3", [{"expr": rho, "color": "TEAL"}], [0, 10], [0, 0.35], y=-0.6, w=9, h=3.8),
                  Obj(id="shape_4", kind="shape", color="RED", payload={
                      "shape": "dot", "size": 1.2,
                      "anchor": {"plot": "plot_3", "x": "5 - 16*10/(9*np.pi**2)*np.cos(t)", "y": "0"}})]
    _show(s, "text_1", "equation_2", "plot_3", "shape_4")
    s.add_step("tween", param="t", value=6.283, duration=4)


def c_dirac(s):
    s.objects += [_eq("equation_1", r"\braket{\psi}{\phi} = \int \psi^*(x)\,\phi(x)\,dx", y=0.5),
                  _eq("equation_2", r"\hat{H}\ket{n} = E_n\ket{n}", y=-1)]
    _show(s, "equation_1", "equation_2")


def c_sqrt_domain(s):
    s.objects += [_plot("plot_1", [{"expr": "np.sqrt(x)", "color": "GREEN"}], [-1, 4], [0, 2.5], w=8, h=4)]
    _show(s, "plot_1")


def c_high_freq(s):
    s.objects += [_plot("plot_1", [{"expr": "np.sin(40*x)", "color": "BLUE"}], [0, 10], [-1.5, 1.5], w=11, h=4)]
    _show(s, "plot_1")


def c_title_subtitle(s):
    s.objects += [_t("text_1", "Mécanique quantique", y=2.8, fs=44), _t("text_2", "Chapitre 1", y=2.0, fs=30)]
    _show(s, "text_1", "text_2")


def c_callout_box(s):
    s.objects += [_t("text_1", "Résultat clé", y=0, fs=36),
                  _shape("shape_2", "rectangle", 0, 0, "YELLOW", width=4, height=1.2)]
    _show(s, "text_1", "shape_2")


def c_transform_labels(s):
    s.objects += [_t("text_1", "N = 1", 0, 0, 36), _t("text_2", "N = 3", 0, 0, 36), _t("text_3", "N = 9", 0, 0, 36)]
    _show(s, "text_1")
    s.add_step("transform", targets=["text_1"], to="text_2")
    s.add_step("transform", targets=["text_1"], to="text_3")


# ---------------------------------------------------------------------------
# defect scenes
# ---------------------------------------------------------------------------

def d_offscreen_text(s):
    s.objects += [_t("text_1", "hors champ", x=9.5, y=0)]
    _show(s, "text_1")


def d_wide_equation(s):
    s.objects += [_eq("equation_1", r"\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n} = \ln 2 \quad\text{et}\quad "
                                    r"\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}", x=3.5, scale=1.2)]
    _show(s, "equation_1")


def d_text_overlap(s):
    s.objects += [_t("text_1", "Énergie cinétique", y=0.1), _t("text_2", "Énergie potentielle", y=-0.1)]
    _show(s, "text_1", "text_2")


def d_label_on_curve(s):
    s.objects += [_plot("plot_1", [{"expr": "np.sin(x)", "color": "YELLOW"}], [-6, 6], [-1.5, 1.5], w=10, h=4),
                  _t("text_2", "onde stationnaire", 0, 0, 34)]
    _show(s, "plot_1", "text_2")


def d_tiny_text(s):
    s.objects += [_t("text_1", "note de bas de page", y=-3, fs=12)]
    _show(s, "text_1")


def d_tiny_equation(s):
    s.objects += [_eq("equation_1", r"E = mc^2", scale=0.25)]
    _show(s, "equation_1")


def d_highlight_before_write(s):
    s.objects += [_t("text_1", "attention")]
    s.add_step("highlight", targets=["text_1"])
    _show(s, "text_1")


def d_unknown_target(s):
    s.objects += [_t("text_1", "bonjour")]
    _show(s, "text_1", "plot_9")


def d_transform_undefined(s):
    s.objects += [_t("text_1", "avant")]
    _show(s, "text_1")
    s.add_step("transform", targets=["text_1"], to="text_7")


def d_latex_error(s):
    s.objects += [_eq("equation_1", r"\frac{1}{2")]
    _show(s, "equation_1")


def d_bad_expression(s):
    s.objects += [_plot("plot_1", [{"expr": "foo(x)"}], [-3, 3], [-1, 1])]
    _show(s, "plot_1")


def d_fake_gibbs(s):  # explicit smooth=True on a discontinuity: invented overshoot
    s.objects += [_plot("plot_1", [{"expr": "np.sign(np.sin(np.pi*x/5))", "smooth": True, "dx": 0.1}],
                        [-7.5, 7.5], [-1.5, 1.5], w=11, h=4)]
    _show(s, "plot_1")


def d_aliasing(s):  # coarse explicit sampling of a fast oscillation
    s.objects += [_plot("plot_1", [{"expr": "np.sin(40*x)", "dx": 0.5}], [0, 10], [-1.5, 1.5], w=11, h=4)]
    _show(s, "plot_1")


def d_never_shown(s):
    s.objects += [_t("text_1", "titre", y=2), _t("text_2", "oublié", y=-1)]
    _show(s, "text_1")


def d_text_on_equation(s):
    s.objects += [_eq("equation_1", r"\nabla\cdot\mathbf{E} = \frac{\rho}{\varepsilon_0}", y=0),
                  _t("text_2", "loi de Gauss", y=0.1, fs=36)]
    _show(s, "equation_1", "text_2")


def d_line_through_text(s):
    s.objects += [_t("text_1", "seuil critique", 0, 0, 40),
                  _shape("shape_2", "line", 0, 0, "RED", size=6)]
    _show(s, "text_1", "shape_2")


def d_plot_offscreen(s):
    s.objects += [_plot("plot_1", [{"expr": "x"}], [-5, 5], [-3, 3], x=3.5, w=12, h=5)]
    _show(s, "plot_1")


def d_dot_leaves_frame(s):  # only off-screen at the end of the tween
    s.params["t"] = 1.0
    s.objects += [_plot("plot_1", [{"expr": "0*x"}], [0, 10], [-1, 1], w=9, h=3),
                  Obj(id="shape_2", kind="shape", color="RED",
                      payload={"shape": "dot", "size": 1.2, "anchor": {"plot": "plot_1", "x": "t", "y": "0"}})]
    _show(s, "plot_1", "shape_2")
    s.add_step("tween", param="t", value=30, duration=3)


def d_undeclared_param(s):
    s.objects += [_t("text_1", "temps")]
    _show(s, "text_1")
    s.add_step("tween", param="omega", value=2.0)


def c_label_near_axis(s):
    s.objects += [_plot("plot_1", [{"expr": "np.exp(-x**2)", "color": "YELLOW"}], [-3, 3], [0, 1.2], w=9, h=4),
                  _t("text_2", "densité gaussienne", 2.6, 1.4, 30)]
    _show(s, "plot_1", "text_2")


def d_label_on_axis(s):  # found while reviewing fix_scene: label over the y-axis tip
    s.objects += [_plot("plot_1", [{"expr": "np.exp(-x**2)", "color": "YELLOW"}], [-3, 3], [0, 1.2], w=9, h=4),
                  _t("text_2", "densité gaussienne", 0, 2.1, 34)]
    _show(s, "plot_1", "text_2")


def d_equations_overlap(s):
    s.objects += [_eq("equation_1", r"i\hbar\frac{\partial\Psi}{\partial t} = \hat{H}\Psi", y=0.15),
                  _eq("equation_2", r"\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V", y=-0.15)]
    _show(s, "equation_1", "equation_2")


CASES: list[Case] = [
    Case("clean/demo", c_demo), Case("clean/derivative_equation", c_derivative),
    Case("clean/area_cubic", c_area_cubic), Case("clean/fourier_gibbs", c_fourier_gibbs),
    Case("clean/quantum_well", c_quantum_well), Case("clean/born_rule", c_born),
    Case("clean/riemann", c_riemann), Case("clean/square_auto", c_square_auto),
    Case("clean/pole_clipped", c_pole_clipped), Case("clean/label_beside_curve", c_label_beside_curve),
    Case("clean/quantum_beat", c_quantum_beat), Case("clean/dirac", c_dirac),
    Case("clean/sqrt_domain", c_sqrt_domain), Case("clean/high_freq", c_high_freq),
    Case("clean/title_subtitle", c_title_subtitle), Case("clean/callout_box", c_callout_box),
    Case("clean/transform_labels", c_transform_labels),
    Case("defect/offscreen_text", d_offscreen_text, {"framing"}),
    Case("defect/wide_equation", d_wide_equation, {"framing"}),
    Case("defect/text_overlap", d_text_overlap, {"collision"}),
    Case("defect/label_on_curve", d_label_on_curve, {"collision"}),
    Case("defect/tiny_text", d_tiny_text, {"legibility"}),
    Case("defect/tiny_equation", d_tiny_equation, {"legibility"}),
    Case("defect/highlight_before_write", d_highlight_before_write, {"timing"}),
    Case("defect/unknown_target", d_unknown_target, {"timing"}),
    Case("defect/transform_undefined", d_transform_undefined, {"timing"}),
    Case("defect/latex_error", d_latex_error, {"compile"}),
    Case("defect/bad_expression", d_bad_expression, {"compile"}),
    Case("defect/fake_gibbs", d_fake_gibbs, {"fidelity"}),
    Case("defect/aliasing", d_aliasing, {"fidelity"}),
    Case("defect/never_shown", d_never_shown, {"timing"}),
    Case("defect/text_on_equation", d_text_on_equation, {"collision"}),
    Case("defect/line_through_text", d_line_through_text, {"collision"}),
    Case("defect/plot_offscreen", d_plot_offscreen, {"framing"}),
    Case("defect/dot_leaves_frame", d_dot_leaves_frame, {"framing"}),
    Case("defect/undeclared_param", d_undeclared_param, {"timing"}),
    Case("defect/equations_overlap", d_equations_overlap, {"collision"}),
    Case("defect/equation_touches_curve", d_equation_touches_curve, {"collision"}),
    Case("clean/label_near_axis", c_label_near_axis),
    Case("defect/label_on_axis", d_label_on_axis, {"collision"}),
]


def build_case(case: Case) -> tuple[Project, Scene]:
    project = Project(id="bench", quality="low")
    scene = project.new_scene(title=case.name)
    case.build(scene)
    return project, scene


def run(measured: bool = True, workdir: Path | None = None, verbose: bool = False,
        cases: list[Case] | None = None) -> dict:
    from .render import probe_scene
    from .verify import inspect_scene

    base = Path(workdir or tempfile.mkdtemp(prefix="chalkcast-bench-"))
    rows = []
    for case in (CASES if cases is None else cases):
        project, scene = build_case(case)
        meas = probe_scene(project, scene, base / case.name.replace("/", "_")) if measured else None
        insp = inspect_scene(scene, measurement=meas)
        fired = [o for o in insp["obligations"] if o["severity"] in ("error", "warning")]
        lanes = {o["lane"] for o in fired}
        detected = bool(case.expect & lanes) if case.expect else None
        false_pos = sorted(lanes - case.expect)
        rows.append({"case": case.name, "expect": sorted(case.expect), "fired": sorted(lanes),
                     "detected": detected, "false_positive": false_pos,
                     "measured": insp["measured"],
                     "details": [f"[{o['severity']}] {o['lane']}: {o['summary']}" for o in fired]})
    defects = [r for r in rows if r["expect"]]
    det_rate = sum(r["detected"] for r in defects) / len(defects)
    fp_rate = sum(bool(r["false_positive"]) for r in rows) / len(rows)
    return {"mode": "measured" if measured else "estimated", "cases": len(rows),
            "defect_cases": len(defects), "detection_rate": round(det_rate, 4),
            "false_positive_rate": round(fp_rate, 4), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="chalkcast bench")
    ap.add_argument("--estimated", action="store_true", help="skip the probe (estimated geometry)")
    ap.add_argument("--holdout", type=int, nargs="?", const=1, choices=(1, 2, 3, 4),
                    help="run held-out set N (cases labelled by authors who never read the "
                         "verifier) instead of the development set")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cases = None
    if args.holdout:
        import importlib
        suffix = "" if args.holdout == 1 else str(args.holdout)
        cases = importlib.import_module(f".bench_holdout{suffix}", __package__).CASES
    res = run(measured=not args.estimated, cases=cases)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    for r in res["rows"]:
        flag = "ok " if (not r["false_positive"] and r["detected"] is not False) else "BAD"
        print(f"{flag} {r['case']:<34} expect={r['expect']} fired={r['fired']}")
        if flag == "BAD":
            for d in r["details"]:
                print("      ", d)
    print(f"\nmode={res['mode']}  cases={res['cases']}  defects={res['defect_cases']}  "
          f"detection={res['detection_rate'] * 100:.1f}%  false_positives={res['false_positive_rate'] * 100:.1f}%")
    return 0
