"""Third held-out verdict benchmark (independently labelled, run once).

Written without reading the verifier, the probe, the fixer, the earlier held-out
sets or the tests. Domains deliberately differ from the earlier sets: orbital
mechanics, statistics (CLT, regression), economics, optics, thermodynamics,
graph theory, probability trees, linear algebra, fluid flow, chemical kinetics,
special relativity, signal processing, electrostatics and number theory.

Every label was checked against a rendered final frame (or, for timing defects
that do not crash, reasoned from the step list).

    from chalkcast.bench import run
    from chalkcast.bench_holdout3 import CASES
    run(cases=CASES)
"""

from __future__ import annotations

import math

from .bench import Case, _eq, _plot, _shape, _show, _t
from .model import Obj, Scene


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _anchor_dot(i, plot, x, y, color="RED", size=1.0):
    return Obj(id=i, kind="shape", color=color, payload={
        "shape": "dot", "size": size, "anchor": {"plot": plot, "x": str(x), "y": str(y)}})


def _area(i, plot, a, b, color="BLUE", fn=1, opacity=0.4):
    return Obj(id=i, kind="area", color=color,
               payload={"plot": plot, "fn": fn, "x_min": a, "x_max": b, "opacity": opacity})


def _riem(i, plot, a, b, dx, color="TEAL", fn=1):
    return Obj(id=i, kind="riemann", color=color,
               payload={"plot": plot, "fn": fn, "x_min": a, "x_max": b, "dx": dx})


def _seg(i, shape, p, q, color=None):
    return _shape(i, shape, color=color, x1=p[0], y1=p[1], x2=q[0], y2=q[1])


def _also(s: Scene, *ids, d=1.0):
    """A write step played together with the previous one."""
    s.add_step("write", targets=list(ids), duration=d, with_previous=True)


# ---------------------------------------------------------------------------
# clean scenes
# ---------------------------------------------------------------------------

VIS_VIVA = "np.sqrt(2/x - 1/4)"


def _vis_viva_plot():
    return _plot("plot_2", [{"expr": VIS_VIVA, "color": "YELLOW"}], [0.5, 7, 1], [0, 2.2, 0.5],
                 x=-2.2, y=-0.6, w=7, h=4.5)


def c_vis_viva(s):
    s.objects += [_t("text_1", "Orbital speed: the vis-viva equation", y=3.3, fs=34),
                  _vis_viva_plot(),
                  _eq("equation_3", r"v^2 = GM\left(\frac{2}{r} - \frac{1}{a}\right)", 4.4, 1.3, 0.8),
                  _t("text_4", "perihelion: fastest", 4.4, 0.0, 26),
                  _t("text_5", "aphelion: slowest", 4.4, -0.7, 26),
                  _anchor_dot("shape_6", "plot_2", 1, "np.sqrt(1.75)")]
    _show(s, "text_1", "plot_2")
    _show(s, "equation_3")
    _show(s, "text_4", "text_5", "shape_6")
    s.add_step("highlight", targets=["shape_6"])


def _clt_objects():
    g = "np.exp(-{k}*x**2)*{c}/np.sqrt(2*np.pi)"
    plots = []
    for i, (k, c) in zip((2, 3, 4), ((0.5, 1), (2, 2), (8, 4))):
        plots.append(_plot(f"plot_{i}", [{"expr": g.format(k=k, c=c), "color": "TEAL"}],
                           [-3, 3, 1], [0, 1.8, 0.5], x=-1.5, y=-0.5, w=8, h=4.8))
    return plots


def c_clt(s):
    s.objects += [_t("text_1", "Central limit theorem", y=3.3, fs=36), *_clt_objects(),
                  _t("text_5", "n = 1", 4.5, 1.0, 32), _t("text_6", "n = 4", 4.5, 1.0, 32),
                  _t("text_7", "n = 16", 4.5, 1.0, 32),
                  _eq("equation_8", r"\bar{X}_n \sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)", 4.5, -0.6, 0.8)]
    _show(s, "text_1", "plot_2", "text_5", "equation_8")
    s.add_step("transform", targets=["text_5"], to="text_6", duration=0.5)
    s.add_step("transform", targets=["plot_2"], to="plot_3", duration=1.5, with_previous=True)
    s.add_step("transform", targets=["text_5"], to="text_7", duration=0.5)
    s.add_step("transform", targets=["plot_2"], to="plot_4", duration=1.5, with_previous=True)


REG_DATA = ((1, 1.4), (2, 2.5), (3, 2.6), (4, 3.9), (5, 3.7), (6, 4.8), (7, 5.0))


def _regression(s, caption=None):
    s.objects += [_t("text_1", "Least-squares regression", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "0.6*x + 1", "color": "ORANGE"}], [0, 8, 1], [0, 7, 1],
                        x=-2, y=-0.6, w=7, h=4.6)]
    dots = []
    for k, (x, y) in enumerate(REG_DATA, start=3):
        s.objects.append(_anchor_dot(f"shape_{k}", "plot_2", x, y, "BLUE", 0.8))
        dots.append(f"shape_{k}")
    s.objects += [_eq("equation_10", r"\hat{y} = 0.6\,x + 1", 4.6, 1.4, 0.85),
                  _t("text_11", "R² = 0.95", 4.6, 0.4, 30)]
    _show(s, "text_1", "plot_2")
    _show(s, *dots)
    _show(s, "equation_10", "text_11")
    if caption is not None:
        s.objects.append(caption)
        _show(s, caption.id)


def c_regression(s):
    _regression(s)


def _market(s, label_pos=((3.6, 1.0), (3.6, 0.3))):
    s.objects += [_t("text_1", "Market equilibrium", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "10 - x", "color": "BLUE"}, {"expr": "1 + 0.8*x", "color": "ORANGE"}],
                        [0, 10, 1], [0, 10, 2], x=-2, y=-0.5, w=6.5, h=5.2),
                  _anchor_dot("shape_3", "plot_2", 5, 5),
                  _t("text_4", "demand", *label_pos[0], 30, "BLUE"),
                  _t("text_5", "supply", *label_pos[1], 30, "ORANGE"),
                  _t("text_6", "equilibrium: Q = 5, P = 5", 4.2, -0.8, 24)]
    _show(s, "text_1", "plot_2")
    _show(s, "text_4", "text_5")
    _show(s, "shape_3", "text_6")


def c_market(s):
    _market(s)


def c_thin_lens(s):
    s.objects += [_t("text_1", "Thin lens: image distance", y=3.3, fs=34),
                  _eq("equation_2", r"\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}", 4.3, 1.4, 0.9),
                  _plot("plot_3", [{"expr": "2*x/(x - 2)", "color": "YELLOW"}], [0, 10, 1], [-10, 10, 5],
                        x=-2.2, y=-0.5, w=7, h=5),
                  _t("text_4", "f = 2", 4.3, -0.2, 30),
                  _t("text_5", "virtual image when d < f", 4.3, -1.0, 24)]
    _show(s, "text_1", "plot_3", "equation_2")
    _show(s, "text_4", "text_5")


def _pv(s, y=-0.6):
    s.objects += [_t("text_1", "Isothermal work on a PV diagram", y=3.3, fs=32),
                  _plot("plot_2", [{"expr": "6/x", "color": "RED"}, {"expr": "3/x", "color": "BLUE"}],
                        [0.5, 6, 1], [0, 8, 2], x=-2, y=y, w=7, h=4.6),
                  _area("area_3", "plot_2", 1, 4, "RED", opacity=0.35),
                  _eq("equation_4", r"W = nRT\ln\frac{V_2}{V_1}", 4.4, 1.2, 0.8),
                  _t("text_5", "hot isotherm", 4.4, 0.0, 26, "RED"),
                  _t("text_6", "cold isotherm", 4.4, -0.6, 26, "BLUE")]
    _show(s, "text_1", "plot_2")
    _show(s, "area_3", "equation_4")
    _show(s, "text_5", "text_6")


def c_pv(s):
    _pv(s)


O = (-1.0, -1.4)


def _eigen(s, label1=(2.8, 1.0)):
    s.objects += [_t("text_1", "Eigenvectors keep their direction", y=3.3, fs=34),
                  _eq("equation_2", r"A\mathbf{v} = \lambda\mathbf{v}", 0, 2.3),
                  _seg("shape_3", "arrow", O, (0.2, -0.2), "BLUE"),
                  _seg("shape_4", "arrow", O, (1.4, 1.0), "BLUE"),
                  _t("text_5", "Av = 2v", *label1, 30, "BLUE"),
                  _seg("shape_6", "arrow", O, (-2.6, 0.2), "GREEN"),
                  _seg("shape_7", "arrow", O, (-1.8, -0.6), "GREEN"),
                  _t("text_8", "Av = ½v", -3.6, -0.6, 30, "GREEN")]
    _show(s, "text_1", "equation_2")
    _show(s, "shape_3", "shape_6")
    s.add_step("transform", targets=["shape_3"], to="shape_4", duration=1.5)
    s.add_step("transform", targets=["shape_6"], to="shape_7", duration=1.5, with_previous=True)
    _show(s, "text_5", "text_8")


def c_eigen(s):
    _eigen(s)


NODES = {"A": (-3.0, 1.0), "B": (0.0, 2.0), "C": (3.0, 1.0), "D": (-1.5, -1.5), "E": (1.5, -1.5)}
EDGES = (("A", "B"), ("B", "C"), ("C", "E"), ("E", "D"), ("D", "A"), ("B", "D"), ("B", "E"))
R_NODE = 0.45


def _graph(s, extra_edge=None):
    s.objects.append(_t("text_1", "An Euler path on a small graph", y=3.4, fs=32))
    ids = []
    for k, (name, (x, y)) in enumerate(NODES.items()):
        s.objects += [_shape(f"shape_{10 + k}", "circle", x, y, "BLUE", size=2 * R_NODE),
                      _t(f"text_{20 + k}", name, x, y, 30)]
        ids += [f"shape_{10 + k}", f"text_{20 + k}"]
    edges = []
    for k, (a, b) in enumerate(EDGES + ((extra_edge,) if extra_edge else ())):
        (xa, ya), (xb, yb) = NODES[a], NODES[b]
        d = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / d, (yb - ya) / d
        p = (round(xa + R_NODE * ux, 3), round(ya + R_NODE * uy, 3))
        q = (round(xb - R_NODE * ux, 3), round(yb - R_NODE * uy, 3))
        s.objects.append(_seg(f"shape_{40 + k}", "line", p, q, "GRAY"))
        edges.append(f"shape_{40 + k}")
    s.objects.append(_t("text_2", "Two odd vertices (D, E): an Euler path exists", y=-3.0, fs=26))
    _show(s, "text_1")
    _show(s, *ids)
    _show(s, *edges)
    _show(s, "text_2")


def c_graph(s):
    _graph(s)


def _tree(s, p_label_fail=(-3.6, -1.35)):
    root, up, dn = (-5.0, 0.0), (-1.5, 1.5), (-1.5, -1.5)
    leaves = ((2.5, 2.5), (2.5, 0.5), (2.5, -0.5), (2.5, -2.5))
    s.objects += [_t("text_1", "Quality control: two inspections", y=3.4, fs=32),
                  _shape("shape_2", "dot", *root, size=1.0),
                  _seg("shape_3", "line", root, up, "GREEN"), _seg("shape_4", "line", root, dn, "RED"),
                  _seg("shape_5", "line", up, leaves[0], "GREEN"), _seg("shape_6", "line", up, leaves[1], "RED"),
                  _seg("shape_7", "line", dn, leaves[2], "GREEN"), _seg("shape_8", "line", dn, leaves[3], "RED"),
                  _t("text_9", "0.9", -3.6, 1.35, 26, "GREEN"), _t("text_10", "0.1", *p_label_fail, 26, "RED"),
                  _t("text_11", "0.95", 0.3, 2.55, 26, "GREEN"), _t("text_12", "0.05", 0.3, 0.55, 26, "RED"),
                  _t("text_13", "0.6", 0.3, -0.55, 26, "GREEN"), _t("text_14", "0.4", 0.3, -2.55, 26, "RED"),
                  _t("text_15", "PP  0.855", 3.8, 2.5, 28), _t("text_16", "PF  0.045", 3.8, 0.5, 28),
                  _t("text_17", "FP  0.06", 3.8, -0.5, 28), _t("text_18", "FF  0.04", 3.8, -2.5, 28)]
    _show(s, "text_1", "shape_2")
    _show(s, "shape_3", "shape_4", "text_9", "text_10")
    _show(s, "shape_5", "shape_6", "shape_7", "shape_8")
    _show(s, "text_11", "text_12", "text_13", "text_14")
    _show(s, "text_15", "text_16", "text_17", "text_18")


def c_tree(s):
    _tree(s)


def _gamma(s, v_end=0.95):
    s.params["v"] = 0.1
    s.objects += [_t("text_1", "The Lorentz factor", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "1/np.sqrt(1 - x**2)", "color": "YELLOW"}], [0, 1, 0.2], [0, 8, 2],
                        x=-1.8, y=-0.6, w=7, h=4.6),
                  _anchor_dot("shape_3", "plot_2", "v", "1/np.sqrt(1 - v**2)"),
                  _eq("equation_4", r"\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}", 4.6, 1.4, 0.8),
                  _t("text_5", "time dilation", 4.6, 0.1, 28)]
    _show(s, "text_1", "plot_2")
    _show(s, "shape_3", "equation_4")
    s.add_step("tween", param="v", value=v_end, duration=3)
    _show(s, "text_5")


def c_gamma(s):
    _gamma(s)


def c_sampling(s):
    s.objects += [_t("text_1", "Sampling a 1 Hz tone at 8 Hz", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "np.sin(2*np.pi*x)", "color": "YELLOW"}], [0, 2, 0.25],
                        [-1.5, 1.5, 0.5], y=-0.3, w=10, h=4)]
    dots = []
    for k in range(17):
        s.objects.append(_anchor_dot(f"shape_{10 + k}", "plot_2", k / 8, f"np.sin(2*np.pi*{k / 8})", "RED", 0.7))
        dots.append(f"shape_{10 + k}")
    s.objects.append(_t("text_3", "Nyquist: sample faster than twice the highest frequency", y=-3.1, fs=26))
    _show(s, "text_1", "plot_2")
    _show(s, *dots, d=2)
    _show(s, "text_3")


def c_coulomb(s):
    s.objects += [_t("text_1", "Field and potential of a point charge", y=3.3, fs=32),
                  _plot("plot_2", [{"expr": "1/x", "color": "YELLOW"}, {"expr": "1/x**2", "color": "BLUE"}],
                        [0, 5, 1], [0, 4, 1], x=-2, y=-0.6, w=7, h=4.6),
                  _eq("equation_3", r"V(r) = \frac{kq}{r}", 4.5, 1.3, 0.8, "YELLOW"),
                  _eq("equation_4", r"E(r) = \frac{kq}{r^2}", 4.5, -0.2, 0.8, "BLUE")]
    _show(s, "text_1", "plot_2")
    _show(s, "equation_3")
    _also(s, "equation_4")


def _collatz(s, tex=r"f(n)=\begin{cases} n/2 & n \text{ even}\\ 3n+1 & n \text{ odd}\end{cases}"):
    s.objects += [_t("text_1", "The Collatz map", y=3.2, fs=40),
                  _eq("equation_2", tex, y=1.4),
                  _t("text_3", "6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1", y=-0.8, fs=32),
                  _t("text_4", "7 → 22 → 11 → 34 → 17 → 52 → 26 → 13 → …", y=-0.8, fs=32),
                  _t("text_5", "Conjecture: every orbit reaches 1", y=-2.4, fs=30)]
    _show(s, "text_1", "equation_2")
    _show(s, "text_3", d=2)
    s.add_step("transform", targets=["text_3"], to="text_4", duration=1.5)
    _show(s, "text_5")


def c_collatz(s):
    _collatz(s)


def c_poiseuille(s):
    s.objects += [_t("text_1", "Poiseuille flow in a pipe", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "1 - x**2", "color": "TEAL"}], [-1, 1, 0.5], [0, 1.2, 0.2],
                        x=-2, y=-0.8, w=7, h=4.2),
                  _area("area_3", "plot_2", -1, 1, "TEAL", opacity=0.35),
                  _eq("equation_4", r"Q = \frac{\pi R^4\,\Delta P}{8\mu L}", 4.5, 0.9, 0.85),
                  _t("text_5", "parabolic velocity profile", 4.5, -0.4, 22)]
    _show(s, "text_1", "plot_2")
    _show(s, "area_3")
    _show(s, "equation_4", "text_5")


def _kinetics(s, eq_pos=((4.5, 1.2), (4.5, -0.3))):
    s.objects += [_t("text_1", "First- vs second-order kinetics", y=3.3, fs=32),
                  _plot("plot_2", [{"expr": "np.exp(-x)", "color": "TEAL"}, {"expr": "1/(1 + x)", "color": "ORANGE"}],
                        [0, 6, 1], [0, 1.1, 0.25], x=-2, y=-0.6, w=7, h=4.6),
                  _eq("equation_3", r"[A] = [A]_0\,e^{-kt}", *eq_pos[0], 0.75, "TEAL"),
                  _eq("equation_4", r"\frac{1}{[A]} = \frac{1}{[A]_0} + kt", *eq_pos[1], 0.75, "ORANGE")]
    _show(s, "text_1", "plot_2")
    _show(s, "equation_3")
    _show(s, "equation_4")


def c_kinetics(s):
    _kinetics(s)


def c_isotherm_tween(s):
    s.params["T"] = 1.0
    s.objects += [_t("text_1", "Isotherms of an ideal gas", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "T/x", "color": "RED"}], [0.5, 5, 1], [0, 6, 1],
                        x=-1.8, y=-0.6, w=7, h=4.6),
                  _eq("equation_3", r"PV = nRT", 4.5, 1.0),
                  _t("text_4", "heating raises the isotherm", 4.5, -0.2, 22)]
    _show(s, "text_1", "plot_2", "equation_3")
    s.add_step("tween", param="T", value=2.5, duration=3)
    _show(s, "text_4")


def c_exp_riemann(s):
    s.objects += [_t("text_1", "Lifetime of an exponential component", y=3.3, fs=32),
                  _plot("plot_2", [{"expr": "np.exp(-x)", "color": "YELLOW"}], [0, 5, 1], [0, 1.2, 0.25],
                        x=-2, y=-0.6, w=7, h=4.6),
                  _riem("riemann_3", "plot_2", 0, 4, 0.5), _riem("riemann_4", "plot_2", 0, 4, 0.1),
                  _eq("equation_5", r"\int_0^\infty e^{-t}\,dt = 1", 4.5, 1.0, 0.8)]
    _show(s, "text_1", "plot_2", "riemann_3", "equation_5")
    s.add_step("transform", targets=["riemann_3"], to="riemann_4", duration=2)


def _lorentz(s, ys=(1.3, -0.2, -1.8)):
    s.objects += [_t("text_1", "Lorentz transformation", y=3.2, fs=40),
                  _eq("equation_2", r"t' = \gamma\left(t - \frac{vx}{c^2}\right)", y=ys[0]),
                  _eq("equation_3", r"x' = \gamma\,(x - vt)", y=ys[1]),
                  _eq("equation_4", r"\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}", y=ys[2])]
    _show(s, "text_1")
    _show(s, "equation_2", "equation_3")
    _show(s, "equation_4")


def c_lorentz_footnote(s):  # corner footnote close to the frame edge but fully visible
    _lorentz(s)
    s.objects.append(_t("text_5", "units: c = 1 from now on", 4.3, -3.5, 24))
    _show(s, "text_5")


# ---------------------------------------------------------------------------
# defect scenes -- framing
# ---------------------------------------------------------------------------

def d_orbit_label_off_right(s):
    s.objects += [_t("text_1", "Orbital speed along an ellipse", y=3.3, fs=34), _vis_viva_plot(),
                  _t("text_3", "aphelion: the slowest point", 6.0, 0.6, 30)]
    _show(s, "text_1", "plot_2")
    _show(s, "text_3")


def d_lorentz_wide_equation(s):
    s.objects += [_t("text_1", "Lorentz boost", y=3.2, fs=40),
                  _eq("equation_2", r"x' = \gamma\,(x - vt), \qquad t' = \gamma\left(t - \frac{vx}{c^2}\right), "
                                    r"\qquad \gamma = \frac{1}{\sqrt{1 - v^2/c^2}}", x=2.5, y=0.5, scale=1.1)]
    _show(s, "text_1", "equation_2")


def d_pv_plot_too_low(s):
    _pv(s, y=-2.3)


def d_gamma_dot_runs_off(s):  # the dot climbs the asymptote and leaves through the top
    _gamma(s, v_end=0.999)


def d_title_clipped_top(s):
    s.objects += [_t("text_1", "Supply, demand and the market price", y=3.92, fs=40)]
    s.objects += [_plot("plot_2", [{"expr": "10 - x", "color": "BLUE"}, {"expr": "1 + 0.8*x", "color": "ORANGE"}],
                        [0, 10, 1], [0, 10, 2], x=0, y=-0.5, w=7, h=5)]
    _show(s, "text_1", "plot_2")


# ---------------------------------------------------------------------------
# defect scenes -- collision
# ---------------------------------------------------------------------------

def d_market_labels_on_curves(s):
    # labels placed "at the curves" -- right on top of the lines they name
    _market(s, label_pos=((-3.3, 0.55), (-3.95, -1.75)))


def d_vis_viva_equation_on_curve(s):
    s.objects += [_t("text_1", "Orbital speed: the vis-viva equation", y=3.3, fs=34), _vis_viva_plot(),
                  _eq("equation_3", r"v^2 = GM\left(\frac{2}{r} - \frac{1}{a}\right)", -3.2, -1.3, 0.8)]
    _show(s, "text_1", "plot_2", "equation_3")


def d_lorentz_crowded(s):
    _lorentz(s, ys=(0.5, 0.0, -1.6))


def d_axis_title_on_tip(s):
    _kinetics(s)
    s.objects.append(_t("text_5", "concentration [A]", -5.5, 1.75, 28))
    _show(s, "text_5")


def d_eigen_arrow_through_label(s):
    _eigen(s, label1=(0.6, 0.2))


def d_tree_label_on_branch(s):
    _tree(s, p_label_fail=(-3.25, -0.75))


# ---------------------------------------------------------------------------
# defect scenes -- legibility
# ---------------------------------------------------------------------------

def d_regression_tiny_caption(s):
    _regression(s, caption=_t("text_12", "synthetic data, n = 7, ordinary least squares", y=-3.4, fs=14))


def d_clt_scaled_label(s):
    s.objects += [_t("text_1", "Central limit theorem", y=3.3, fs=36), *_clt_objects()[:1],
                  _t("text_5", "sample size n = 1", 4.5, 1.0, 36, scale=0.45)]
    _show(s, "text_1", "plot_2", "text_5")


def d_tiny_matrix_equation(s):
    s.objects += [_t("text_1", "Diagonalisation", y=3.2, fs=40),
                  _eq("equation_2", r"A = P D P^{-1}", y=1.2),
                  _eq("equation_3", r"\begin{pmatrix}2 & 1\\ 1 & 2\end{pmatrix} = "
                                    r"\begin{pmatrix}1 & -1\\ 1 & 1\end{pmatrix}"
                                    r"\begin{pmatrix}3 & 0\\ 0 & 1\end{pmatrix}"
                                    r"\begin{pmatrix}1 & -1\\ 1 & 1\end{pmatrix}^{-1}", y=-0.8, scale=0.28)]
    _show(s, "text_1", "equation_2", "equation_3")


# ---------------------------------------------------------------------------
# defect scenes -- timing
# ---------------------------------------------------------------------------

def d_highlight_after_fade(s):
    s.objects += [_t("text_1", "Field and potential of a point charge", y=3.3, fs=32),
                  _eq("equation_2", r"V(r) = \frac{kq}{r}", 0, 0.8),
                  _eq("equation_3", r"E(r) = -\frac{dV}{dr} = \frac{kq}{r^2}", 0, -1.0)]
    _show(s, "text_1", "equation_2")
    s.add_step("fade", targets=["equation_2"])
    _show(s, "equation_3")
    s.add_step("highlight", targets=["equation_2"])


def d_transform_typo_target(s):
    s.objects += [_t("text_1", "Central limit theorem", y=3.3, fs=36), *_clt_objects()[:2],
                  _t("text_5", "n = 1", 4.5, 1.0, 32), _t("text_6", "n = 4", 4.5, 1.0, 32)]
    _show(s, "text_1", "plot_2", "text_5")
    s.add_step("transform", targets=["text_5"], to="text_6", duration=0.5)
    s.add_step("transform", targets=["plot_2"], to="plot_5", duration=1.5, with_previous=True)


def d_tween_undeclared_param(s):
    s.params["v"] = 0.1
    s.objects += [_t("text_1", "The Lorentz factor", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "1/np.sqrt(1 - x**2)", "color": "YELLOW"}], [0, 1, 0.2], [0, 8, 2],
                        x=-1.8, y=-0.6, w=7, h=4.6),
                  _anchor_dot("shape_3", "plot_2", "v", "1/np.sqrt(1 - v**2)")]
    _show(s, "text_1", "plot_2", "shape_3")
    s.add_step("tween", param="beta", value=0.9, duration=3)


def d_legend_never_shown(s):
    s.objects += [_t("text_1", "First- vs second-order kinetics", y=3.3, fs=32),
                  _plot("plot_2", [{"expr": "np.exp(-x)", "color": "TEAL"}, {"expr": "1/(1 + x)", "color": "ORANGE"}],
                        [0, 6, 1], [0, 1.1, 0.25], x=-2, y=-0.6, w=7, h=4.6),
                  _t("text_3", "first order", 4.5, 1.0, 28, "TEAL"),
                  _t("text_4", "second order", 4.5, 0.2, 28, "ORANGE")]
    _show(s, "text_1", "plot_2", "text_3")


def d_no_steps(s):
    s.objects += [_t("text_1", "Poiseuille flow in a pipe", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "1 - x**2", "color": "TEAL"}], [-1, 1, 0.5], [0, 1.2, 0.2],
                        x=-2, y=-0.8, w=7, h=4.2),
                  _eq("equation_3", r"Q = \frac{\pi R^4\,\Delta P}{8\mu L}", 4.5, 0.9, 0.85)]


def d_transform_before_write(s):
    s.objects += [_t("text_1", "The Collatz map", y=3.2, fs=40),
                  _t("text_3", "6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1", y=-0.3, fs=32),
                  _t("text_4", "7 → 22 → 11 → 34 → 17 → 52 → 26 → 13 → …", y=-0.3, fs=32)]
    _show(s, "text_1")
    s.add_step("transform", targets=["text_3"], to="text_4", duration=1.5)
    _show(s, "text_3")


# ---------------------------------------------------------------------------
# defect scenes -- compile
# ---------------------------------------------------------------------------

def d_latex_typo_lamda(s):
    s.objects += [_t("text_1", "Eigenvectors keep their direction", y=3.3, fs=34),
                  _eq("equation_2", r"A\mathbf{v} = \lamda\mathbf{v}", 0, 0.5)]
    _show(s, "text_1", "equation_2")


def d_collatz_missing_cases(s):
    # \end{cases} forgotten: the environment is closed by Manim's align* instead
    _collatz(s, tex=r"f(n)=\begin{cases} n/2 & n \text{ even}\\ 3n+1 & n \text{ odd}")


def d_erf_unknown(s):
    s.objects += [_t("text_1", "CDF of the sample mean", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "0.5*(1 + erf(x/np.sqrt(2)))", "color": "TEAL"}], [-3, 3, 1], [0, 1, 0.25],
                        y=-0.5, w=8, h=4.5)]
    _show(s, "text_1", "plot_2")


# ---------------------------------------------------------------------------
# defect scenes -- fidelity
# ---------------------------------------------------------------------------

def d_am_signal_aliased(s):
    s.objects += [_t("text_1", "Amplitude modulation", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "(1 + 0.5*np.cos(x))*np.sin(25*x)", "color": "YELLOW", "dx": 0.4}],
                        [0, 12, 2], [-1.6, 1.6, 0.5], y=-0.4, w=11, h=4.5)]
    _show(s, "text_1", "plot_2")


PRIMES = "sum(np.heaviside(x - p, 1) for p in (2, 3, 5, 7, 11, 13, 17, 19))"


def d_prime_count_smoothed(s):
    s.objects += [_t("text_1", "The prime-counting function", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": PRIMES, "color": "YELLOW", "smooth": True, "dx": 0.25}],
                        [0, 20, 2], [0, 9, 1], y=-0.5, w=10, h=5),
                  _eq("equation_3", r"\pi(x) \sim \frac{x}{\ln x}", -2.2, 1.2, 0.8)]
    _show(s, "text_1", "plot_2", "equation_3")


def d_resonance_undersampled(s):
    s.objects += [_t("text_1", "A sharp resonance", y=3.3, fs=36),
                  _plot("plot_2", [{"expr": "1/(1 + 100*(x - 5)**2)", "color": "TEAL", "dx": 0.7}],
                        [0, 10, 1], [0, 1.2, 0.2], y=-0.5, w=10, h=5)]
    _show(s, "text_1", "plot_2")


# ---------------------------------------------------------------------------

CASES: list[Case] = [
    Case("holdout3/clean/vis_viva", c_vis_viva),
    Case("holdout3/clean/clt_transforms", c_clt),
    Case("holdout3/clean/regression", c_regression),
    Case("holdout3/clean/market_equilibrium", c_market),
    Case("holdout3/clean/thin_lens_pole", c_thin_lens),
    Case("holdout3/clean/pv_isotherms_area", c_pv),
    Case("holdout3/clean/eigenvectors", c_eigen),
    Case("holdout3/clean/euler_graph", c_graph),
    Case("holdout3/clean/probability_tree", c_tree),
    Case("holdout3/clean/lorentz_factor_tween", c_gamma),
    Case("holdout3/clean/sampling_dots", c_sampling),
    Case("holdout3/clean/coulomb", c_coulomb),
    Case("holdout3/clean/collatz", c_collatz),
    Case("holdout3/clean/poiseuille_area", c_poiseuille),
    Case("holdout3/clean/kinetics", c_kinetics),
    Case("holdout3/clean/isotherm_tween", c_isotherm_tween),
    Case("holdout3/clean/exp_riemann", c_exp_riemann),
    Case("holdout3/clean/lorentz_footnote", c_lorentz_footnote),
    Case("holdout3/defect/orbit_label_off_right", d_orbit_label_off_right, {"framing"}),
    Case("holdout3/defect/lorentz_wide_equation", d_lorentz_wide_equation, {"framing"}),
    Case("holdout3/defect/pv_plot_too_low", d_pv_plot_too_low, {"framing"}),
    Case("holdout3/defect/gamma_dot_runs_off", d_gamma_dot_runs_off, {"framing"}),
    Case("holdout3/defect/title_clipped_top", d_title_clipped_top, {"framing"}),
    Case("holdout3/defect/market_labels_on_curves", d_market_labels_on_curves, {"collision"}),
    Case("holdout3/defect/vis_viva_equation_on_curve", d_vis_viva_equation_on_curve, {"collision"}),
    Case("holdout3/defect/lorentz_crowded", d_lorentz_crowded, {"collision"}),
    Case("holdout3/defect/axis_title_on_tip", d_axis_title_on_tip, {"collision"}),
    Case("holdout3/defect/eigen_arrow_through_label", d_eigen_arrow_through_label, {"collision"}),
    Case("holdout3/defect/tree_label_on_branch", d_tree_label_on_branch, {"collision"}),
    Case("holdout3/defect/regression_tiny_caption", d_regression_tiny_caption, {"legibility"}),
    Case("holdout3/defect/clt_scaled_label", d_clt_scaled_label, {"legibility"}),
    Case("holdout3/defect/tiny_matrix_equation", d_tiny_matrix_equation, {"legibility"}),
    Case("holdout3/defect/highlight_after_fade", d_highlight_after_fade, {"timing"}),
    Case("holdout3/defect/transform_typo_target", d_transform_typo_target, {"timing"}),
    Case("holdout3/defect/tween_undeclared_param", d_tween_undeclared_param, {"timing"}),
    Case("holdout3/defect/legend_never_shown", d_legend_never_shown, {"timing"}),
    Case("holdout3/defect/no_steps", d_no_steps, {"timing"}),
    Case("holdout3/defect/transform_before_write", d_transform_before_write, {"timing"}),
    Case("holdout3/defect/latex_typo_lamda", d_latex_typo_lamda, {"compile"}),
    Case("holdout3/defect/collatz_missing_cases", d_collatz_missing_cases, {"compile"}),
    Case("holdout3/defect/erf_unknown", d_erf_unknown, {"compile"}),
    Case("holdout3/defect/am_signal_aliased", d_am_signal_aliased, {"fidelity"}),
    Case("holdout3/defect/prime_count_smoothed", d_prime_count_smoothed, {"fidelity"}),
    Case("holdout3/defect/resonance_undersampled", d_resonance_undersampled, {"fidelity"}),
]
