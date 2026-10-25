"""Fourth held-out verdict benchmark (independently labelled, run once).

Written without reading the verifier, probe, fixer, earlier held-out sets or
tests. Domains: SIR epidemiology, population genetics (Hardy-Weinberg,
selection sweeps), music / harmonics and standing waves, finance (compound
interest, present value), the pendulum, the logistic map, game theory (Hawk-Dove,
prisoner's dilemma), map projections (Mercator), algorithmic complexity, and
acoustics (inverse-square law, sound level). Scenes lean on the newer features:
parameters with tweens, anchored dots tracking curves, ``with_previous`` beats,
transforms, fades and camera zooms. Every label was checked against rendered
frames (default chalkboard theme, low quality).

    from chalkcast.bench import run
    from chalkcast.bench_holdout4 import CASES
"""

from __future__ import annotations

from .bench import Case, _eq, _plot, _shape, _show, _t
from .model import Obj, Scene

# default curve colours on the chalkboard theme (codegen.CURVE_COLORS)
C1, C2, C3 = "#f6c76b", "#6bc5f6", "#f6866b"


def _dot_on(i, plot, x, y, color=None, size=1.0):
    return Obj(id=i, kind="shape", color=color,
               payload={"shape": "dot", "size": size, "anchor": {"plot": plot, "x": x, "y": y}})


def _area(i, plot, a, b, fn=1, opacity=0.5, color=None):
    return Obj(id=i, kind="area", color=color,
               payload={"plot": plot, "fn": fn, "x_min": a, "x_max": b, "opacity": opacity})


def _riemann(i, plot, a, b, dx, fn=1, color=None):
    return Obj(id=i, kind="riemann", color=color,
               payload={"plot": plot, "fn": fn, "x_min": a, "x_max": b, "dx": dx})


# ---------------------------------------------------------------------------
# SIR epidemiology
# ---------------------------------------------------------------------------

_S = "1 - 0.9/(1 + exp(-(x - 40)/7))"
_I = "0.35*exp(-((x - 40)/12)**2)"
_R = "0.9/(1 + exp(-(x - 45)/8))"


def _sir_objects(s: Scene, legend_inside: bool = False, width: float = 8.0, px: float = -1.5):
    s.objects += [
        _t("text_1", "SIR epidemic model", 0, 3.4, fs=40),
        _plot("plot_2", [_S, _I, _R], [0, 100, 20], [0, 1, 0.25], x=px, y=-0.4, w=width, h=5),
        _t("text_3", "Susceptible", 4.6, 1.0, fs=30, color=C1),
        (_t("text_4", "Infected peak", -2.3, -1.15, fs=30, color=C2) if legend_inside
         else _t("text_4", "Infected", 4.6, 0.2, fs=30, color=C2)),
        _t("text_5", "Recovered", 4.6, -0.6, fs=30, color=C3),
    ]


def _sir_steps(s: Scene):
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["text_3", "text_4", "text_5"], duration=1.0)
    s.add_step("highlight", targets=["text_4"], duration=1.0)
    s.add_step("wait", duration=1.0)


def sir_curves(s: Scene):
    _sir_objects(s)
    _sir_steps(s)


def sir_peak_dot(s: Scene):
    s.params["t"] = 0.0
    s.objects += [
        _t("text_1", "Infections rise, peak, and fall", 0, 3.4, fs=36),
        _plot("plot_2", [_I], [0, 100, 20], [0, 0.4, 0.1], x=-1.5, y=-0.4, w=8, h=5),
        _dot_on("shape_3", "plot_2", "t", "0.35*exp(-((t - 40)/12)**2)", color="#ffffff"),
        _eq("equation_4", r"R_0 = \frac{\beta}{\gamma} = 2.5", 4.6, 1.2, scale=0.9),
        _t("text_5", "infected fraction", 4.6, -0.2, fs=28, color=C1),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["equation_4"], duration=1.0)
    s.add_step("write", targets=["text_5"], duration=1.0, with_previous=True)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("tween", param="t", value=100.0, duration=4.0)
    s.add_step("wait", duration=0.5)


def sir_legend_on_curve(s: Scene):
    _sir_objects(s, legend_inside=True)
    _sir_steps(s)


def sir_r0_undefined_macro(s: Scene):
    s.objects += [
        _t("text_1", "Basic reproduction number", 0, 3.0, fs=40),
        _eq("equation_2", r"R_0 = \frac{\Beta}{\gamma}", 0, 0.5, scale=1.3),
        _t("text_3", "above 1: the epidemic grows", 0, -1.5, fs=32),
    ]
    _show(s, "text_1")
    _show(s, "equation_2")
    _show(s, "text_3")


def sir_no_steps(s: Scene):
    _sir_objects(s)


def sir_plot_too_wide(s: Scene):
    s.objects += [
        _t("text_1", "SIR epidemic model", 0, 3.4, fs=40),
        _plot("plot_2", [_S, _I, _R], [0, 100, 20], [0, 1, 0.25], x=2.0, y=-0.4, w=12, h=5),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.5)
    s.add_step("wait", duration=1.0)


# ---------------------------------------------------------------------------
# population genetics
# ---------------------------------------------------------------------------

def _hw_base(s: Scene):
    s.objects += [
        _eq("equation_1", r"p^2 + 2pq + q^2 = 1", 0, 3.3),
        _plot("plot_2", ["x**2", "2*x*(1-x)", "(1-x)**2"], [0, 1, 0.2], [0, 1, 0.25],
              x=-1.8, y=-0.5, w=7, h=5, x_label="p"),
    ]


def hw_genotypes(s: Scene):
    _hw_base(s)
    s.objects += [
        _eq("equation_3", r"AA:\ p^2", 4.4, 1.2, color=C1),
        _eq("equation_4", r"Aa:\ 2pq", 4.4, 0.0, color=C2),
        _eq("equation_5", r"aa:\ q^2", 4.4, -1.2, color=C3),
    ]
    _show(s, "equation_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3", "equation_4", "equation_5"], duration=1.5)
    s.add_step("highlight", targets=["equation_4"], duration=1.0)


def hw_labels_overlap(s: Scene):
    _hw_base(s)
    s.objects += [
        _t("text_3", "Homozygous AA", 4.1, 0.6, fs=34, color=C1),
        _t("text_4", "Heterozygous Aa", 4.3, 0.35, fs=34, color=C2),
    ]
    _show(s, "equation_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["text_3", "text_4"], duration=1.5)
    s.add_step("wait", duration=1.0)


def hw_tiny_equation(s: Scene):
    _hw_base(s)
    s.objects += [
        _t("text_3", "Genotype frequencies", 4.4, 1.0, fs=30),
        _eq("equation_4", r"p = 0.6,\ q = 0.4 \Rightarrow 2pq = 0.48", 4.4, -0.2, scale=0.3),
    ]
    _show(s, "equation_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["text_3"], duration=1.0)
    s.add_step("write", targets=["equation_4"], duration=1.0)
    s.add_step("wait", duration=1.0)


def _selection(s: Scene, caption_fs: float):
    s.params["s"] = 0.1
    s.objects += [
        _t("text_1", "Selection sweeps a new allele", 0, 3.4, fs=36),
        _plot("plot_2", ["1/(1 + 9*exp(-s*x))"], [0, 40, 10], [0, 1, 0.25], x=0, y=0.1, w=9, h=5,
              x_label="t", y_label="p"),
        _t("text_3", "stronger selection, faster sweep", 0, -3.3, fs=caption_fs),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["text_3"], duration=1.0)
    s.add_step("tween", param="s", value=0.6, duration=4.0)
    s.add_step("wait", duration=0.5)


def selection_sweep(s: Scene):
    _selection(s, 30)


def selection_tiny_caption(s: Scene):
    _selection(s, 12)


# ---------------------------------------------------------------------------
# music / harmonics
# ---------------------------------------------------------------------------

def harmonics_transform(s: Scene):
    kw = dict(x=0, y=-0.3, w=9, h=4)
    s.objects += [
        _eq("equation_1", r"f_n = n\,f_1", 0, 3.2),
        _plot("plot_2", ["sin(pi*x)"], [0, 1, 0.25], [-1.2, 1.2, 0.5], **kw),
        _plot("plot_3", ["sin(2*pi*x)"], [0, 1, 0.25], [-1.2, 1.2, 0.5], **kw),
        _plot("plot_4", ["sin(3*pi*x)"], [0, 1, 0.25], [-1.2, 1.2, 0.5], **kw),
        _t("text_5", "Fundamental", 0, -3.1, fs=34),
        _t("text_6", "2nd harmonic", 0, -3.1, fs=34),
        _t("text_7", "3rd harmonic", 0, -3.1, fs=34),
    ]
    _show(s, "equation_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["text_5"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=0.5)
    s.add_step("transform", targets=["plot_2"], to="plot_3", duration=1.5)
    s.add_step("transform", targets=["text_5"], to="text_6", duration=1.5, with_previous=True)
    s.add_step("wait", duration=0.5)
    s.add_step("transform", targets=["plot_2"], to="plot_4", duration=1.5)
    s.add_step("transform", targets=["text_5"], to="text_7", duration=1.5, with_previous=True)
    s.add_step("wait", duration=0.5)


def standing_wave(s: Scene):
    s.params["ph"] = 0.0
    s.objects += [
        _t("text_1", "A standing wave on a string", 0, 3.3, fs=36),
        _plot("plot_2", ["sin(2*pi*x)*cos(ph)"], [0, 1, 0.25], [-1.5, 1.5, 0.5],
              x=0, y=-0.3, w=10, h=4.5),
        _dot_on("shape_3", "plot_2", "0.5", "0", color=C3),
        _t("text_4", "the middle point never moves: a node", 0, -3.3, fs=30, color=C3),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("tween", param="ph", value=12.566, duration=4.0)
    s.add_step("write", targets=["text_4"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=0.5)


def harmonic_aliased(s: Scene):
    s.objects += [
        _t("text_1", "The 12th harmonic", 0, 3.3, fs=36),
        _plot("plot_2", [{"expr": "sin(12*2*pi*x)", "dx": 0.1}], [0, 1, 0.25], [-1.2, 1.2, 0.5],
              x=0, y=-0.3, w=10, h=4.5),
        _eq("equation_3", r"y = \sin(24\pi x)", 0, -3.2),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3"], duration=1.0)
    s.add_step("wait", duration=1.0)


def tween_unused_param(s: Scene):
    s.params["f"] = 1.0
    s.objects += [
        _t("text_1", "Raising the pitch", 0, 3.3, fs=36),
        _plot("plot_2", ["sin(2*pi*x)"], [0, 2, 0.5], [-1.2, 1.2, 0.5], x=0, y=-0.3, w=10, h=4.5),
        _t("text_3", "frequency rises from 1 to 3 Hz", 0, -3.2, fs=30),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["text_3"], duration=1.0)
    s.add_step("tween", param="f", value=3.0, duration=3.0)
    s.add_step("wait", duration=0.5)


# ---------------------------------------------------------------------------
# acoustics
# ---------------------------------------------------------------------------

def beats_clipped(s: Scene):
    s.objects += [
        _t("text_1", "Beats: 10 Hz + 11 Hz", 0, 3.3, fs=36),
        _plot("plot_2", ["cos(2*pi*11*x) + cos(2*pi*10*x)"], [0, 2, 0.5], [-0.5, 0.5, 0.25],
              x=0, y=-0.3, w=11, h=4.5),
        _t("text_3", "the loudness swells once per second", 0, -3.3, fs=30),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["text_3"], duration=1.0)
    s.add_step("wait", duration=1.0)


def _inverse_square(s: Scene):
    s.params["d"] = 0.6
    s.objects += [
        _t("text_1", "Inverse-square law", 0, 3.4, fs=40),
        _plot("plot_2", ["1/x**2"], [0, 5, 1], [0, 4, 1], x=-1.5, y=-0.4, w=8, h=5,
              x_label="r"),
        _dot_on("shape_3", "plot_2", "d", "1/d**2", color="#ffffff"),
        _eq("equation_4", r"I \propto \frac{1}{r^2}", 4.6, 0.6),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["equation_4"], duration=1.0, with_previous=True)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("tween", param="d", value=4.8, duration=4.0)


def inverse_square(s: Scene):
    _inverse_square(s)
    s.add_step("wait", duration=0.5)


def zoom_label_cut(s: Scene):
    _inverse_square(s)
    s.objects.append(_t("text_5", "intensity drops fast", -1.0, 0.0, fs=28, color=C1))
    s.add_step("tween", param="d", value=1.2, duration=1.5)
    s.add_step("camera_move", to="2.5@-3,-1", duration=1.5)
    s.add_step("write", targets=["text_5"], duration=1.0)
    s.add_step("wait", duration=1.0)


def _spl(s: Scene, y_range):
    s.objects += [
        _t("text_1", "Sound level falls with distance", 0, 3.4, fs=36),
        _plot("plot_2", ["100 - 20*log10(x)"], [0, 100, 20], y_range, x=-1.5, y=-0.4, w=8, h=5,
              show_numbers=True, x_label="r"),
        _eq("equation_3", r"L = L_0 - 20\log_{10} r", 4.6, 0.8, scale=0.8),
        _t("text_4", "-6 dB per doubling", 4.6, -0.4, fs=28, color=C1),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3"], duration=1.0)
    s.add_step("write", targets=["text_4"], duration=1.0)
    s.add_step("wait", duration=1.0)


def spl_curve(s: Scene):
    _spl(s, [0, 120, 20])


def spl_empty_curve(s: Scene):
    _spl(s, [0, 50, 10])


# ---------------------------------------------------------------------------
# finance
# ---------------------------------------------------------------------------

def compound_interest(s: Scene):
    s.objects += [
        _t("text_1", "Compound interest", 0, 3.4, fs=40),
        _plot("plot_2", ["1000*(1 + 0.05)**x"], [0, 30, 5], [0, 5000, 1000], x=-1.8, y=-0.5,
              w=8, h=5, show_numbers=True, x_label="t", y_label="A"),
        _eq("equation_3", r"A = P(1+r)^t", 4.8, 1.0),
        _t("text_4", "P = 1000, r = 5%", 4.8, -0.2, fs=28, color=C1),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3"], duration=1.0)
    s.add_step("write", targets=["text_4"], duration=1.0, with_previous=True)
    s.add_step("highlight", targets=["equation_3"], duration=1.0)


def rate_tween(s: Scene):
    s.params["r"] = 0.02
    s.objects += [
        _t("text_1", "A higher rate compounds faster", 0, 3.4, fs=36),
        _plot("plot_2", ["1000*(1 + r)**x"], [0, 30, 5], [0, 12000, 2000], x=-1.5, y=-0.4,
              w=8, h=5, y_label="A"),
        _dot_on("shape_3", "plot_2", "25", "1000*(1 + r)**25", color="#ffffff"),
        _eq("equation_4", r"r:\ 2\% \to 8\%", 4.6, 0.5),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("write", targets=["equation_4"], duration=0.5, with_previous=True)
    s.add_step("tween", param="r", value=0.08, duration=4.0)
    s.add_step("wait", duration=0.5)


def doubling_smoothed(s: Scene):
    s.objects += [
        _t("text_1", "Money doubles once per period", 0, 3.4, fs=36),
        _plot("plot_2", [{"expr": "1000*2**floor(x)", "smooth": True, "dx": 1.0}],
              [0, 5, 1], [0, 32000, 8000], x=-1.0, y=-0.4, w=8, h=5, x_label="n", y_label="A"),
        _eq("equation_3", r"A_n = 1000 \cdot 2^{\lfloor n \rfloor}", 4.9, 0.5, scale=0.8),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3"], duration=1.0)
    s.add_step("wait", duration=1.0)


def continuous_clipped(s: Scene):
    s.objects += [
        _t("text_1", "Continuous compounding over 40 years", 0, 3.4, fs=36),
        _plot("plot_2", ["1000*exp(0.1*x)"], [0, 40, 10], [0, 5000, 1000], x=-1.5, y=-0.4,
              w=8, h=5, show_numbers=True, x_label="t", y_label="A"),
        _eq("equation_3", r"A = P e^{rt}", 4.8, 0.5),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3"], duration=1.0)
    s.add_step("wait", duration=1.0)


def title_through_axis(s: Scene):
    s.objects += [
        _plot("plot_1", ["1000*(1 + 0.05)**x"], [0, 30, 5], [0, 5000, 1000], x=2.0, y=-0.5,
              w=9, h=6),
        _t("text_2", "Growth of 1000 at 5%", 0, 2.0, fs=40),
    ]
    s.add_step("write", targets=["plot_1"], duration=2.0)
    _show(s, "text_2")
    s.add_step("wait", duration=1.0)


def present_value(s: Scene):
    s.objects += [
        _t("text_1", "Present value of a cash stream", 0, 3.4, fs=36),
        _plot("plot_2", ["100*exp(-0.08*x)"], [0, 10, 2], [0, 120, 20], x=-1.8, y=-0.5, w=8, h=5,
              x_label="t"),
        _riemann("riemann_3", "plot_2", 0, 10, 1.0),
        _area("area_4", "plot_2", 0, 10, opacity=0.5),
        _eq("equation_5", r"PV = \sum_k C e^{-rk}", 4.7, 0.8, scale=0.8),
        _eq("equation_6", r"PV = \int_0^{T} C e^{-rt}\,dt", 4.7, 0.8, scale=0.8),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["riemann_3"], duration=1.5)
    s.add_step("write", targets=["equation_5"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=0.5)
    s.add_step("transform", targets=["riemann_3"], to="area_4", duration=1.5)
    s.add_step("transform", targets=["equation_5"], to="equation_6", duration=1.5,
               with_previous=True)
    s.add_step("wait", duration=1.0)


# ---------------------------------------------------------------------------
# pendulum
# ---------------------------------------------------------------------------

def pendulum_track(s: Scene):
    s.params["t"] = 0.0
    s.objects += [
        _eq("equation_1", r"\theta(t) = \theta_0 \cos(\omega t)", 0, 3.2),
        _plot("plot_2", ["0.5*cos(pi*x/2)"], [0, 10, 2], [-1, 1, 0.5], x=-1.0, y=-0.6, w=9, h=4,
              x_label="t", y_label=r"\theta"),
        _dot_on("shape_3", "plot_2", "t", "0.5*cos(pi*t/2)", color="#ffffff"),
        _eq("equation_4", r"T = 2\pi\sqrt{L/g}", 0, -3.3, scale=0.8, color=C1),
    ]
    _show(s, "equation_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("tween", param="t", value=10.0, duration=5.0)
    s.add_step("write", targets=["equation_4"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=0.5)


def damped_zoom(s: Scene):
    s.objects += [
        _t("text_1", "A damped pendulum", 0, 3.4, fs=40),
        _plot("plot_2", ["exp(-0.2*x)*cos(3*x)", "exp(-0.2*x)"], [0, 15, 3], [-1, 1, 0.5],
              x=0, y=-0.3, w=11, h=4.5),
        _t("text_3", "amplitude decays as exp(-0.2t)", 0, -3.3, fs=28, color=C2),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("fade", targets=["text_1"], duration=0.5)
    s.add_step("camera_move", to="2@-3.5,0", duration=1.5)
    s.add_step("wait", duration=1.0)
    s.add_step("camera_move", to="1@0,0", duration=1.5)
    s.add_step("write", targets=["text_3"], duration=1.0)
    s.add_step("wait", duration=0.5)


def pendulum_dot_degrees(s: Scene):
    s.params["t"] = 0.0
    s.objects += [
        _t("text_1", "Pendulum angle over time", 0, 3.4, fs=36),
        _plot("plot_2", ["0.5*cos(2*x)"], [0, 10, 2], [-1, 1, 0.5], x=0, y=-0.3, w=10, h=4,
              y_label=r"\theta"),
        # the bob's angle typed in degrees on radian axes
        _dot_on("shape_3", "plot_2", "t", "30*cos(2*t)", color="#ffffff"),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("tween", param="t", value=10.0, duration=4.0)


def period_formula_off_right(s: Scene):
    s.objects += [
        _t("text_1", "Large swings take longer", -2.0, 3.0, fs=36),
        _plot("plot_2", ["1 + x**2/16"], [0, 1.5, 0.5], [0.9, 1.2, 0.1], x=-2.5, y=-0.6, w=6, h=4,
              x_label=r"\theta_0", y_label="T/T_0"),
        _eq("equation_3", r"T \approx 2\pi\sqrt{\frac{L}{g}}\left(1 + \frac{\theta_0^2}{16}\right)",
            5.5, 0.5),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["equation_3"], duration=1.5)
    s.add_step("wait", duration=1.0)


def small_angle_swap(s: Scene):
    s.objects += [
        _t("text_1", "The small-angle approximation", 0, 3.4, fs=36),
        _plot("plot_2", ["sin(x)", "x"], [0, 1.5, 0.5], [0, 1.6, 0.4], x=-2.0, y=-0.5, w=6.5, h=5,
              x_label=r"\theta"),
        _eq("equation_3", r"\sin\theta \approx \theta", 4.3, 0.8),
        _eq("equation_4", r"\text{error} < 1\%", 4.3, -0.4, scale=0.9),
        _eq("equation_5", r"\text{for } \theta < 0.24", 4.3, -1.4, scale=0.9),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3"], duration=1.5, with_previous=True)
    s.add_step("wait", duration=0.5)
    s.add_step("fade", targets=["equation_3"], duration=1.0)
    s.add_step("write", targets=["equation_4", "equation_5"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=1.0)


def pendulum_label_on_path(s: Scene):
    s.params["t"] = 0.0
    s.objects += [
        _eq("equation_1", r"\theta(t) = \theta_0 \cos(\omega t)", 0, 3.2),
        _plot("plot_2", ["0.5*cos(2*x)"], [0, 10, 2], [-1, 1, 0.5], x=-1.0, y=-0.6, w=9, h=4),
        _dot_on("shape_3", "plot_2", "t", "0.5*cos(2*t)", color="#ffffff"),
        # sits on the first trough of the curve, so the curve and the dot run through it
        _t("text_4", "turning point", -4.1, -1.6, fs=30, color=C2),
    ]
    _show(s, "equation_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["shape_3", "text_4"], duration=0.5)
    s.add_step("tween", param="t", value=10.0, duration=5.0)


# ---------------------------------------------------------------------------
# logistic map
# ---------------------------------------------------------------------------

def _logistic(s: Scene, expr: str, highlight_first: bool = False):
    s.params["r"] = 2.5
    s.objects += [
        _t("text_1", "The logistic map", 0, 3.4, fs=40),
        _plot("plot_2", [expr, "x"], [0, 1, 0.25], [0, 1, 0.25], x=-2.0, y=-0.5, w=5.5, h=5.5,
              y_label="x_{n+1}"),
        _dot_on("shape_3", "plot_2", "1 - 1/r", "1 - 1/r", color="#ffffff"),
        _eq("equation_4", r"x^* = 1 - \frac{1}{r}", 4.2, 0.8),
        _t("text_5", "r from 2.5 to 3.8", 4.2, -0.6, fs=28, color=C1),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    if highlight_first:
        s.add_step("highlight", targets=["equation_4"], duration=1.0)
    s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("write", targets=["equation_4"], duration=1.0)
    s.add_step("write", targets=["text_5"], duration=1.0, with_previous=True)
    s.add_step("tween", param="r", value=3.8, duration=4.0)
    s.add_step("wait", duration=0.5)


def logistic_fixed_point(s: Scene):
    _logistic(s, "r*x*(1-x)")


def logistic_missing_star(s: Scene):
    _logistic(s, "r*x(1-x)")


def logistic_highlight_early(s: Scene):
    _logistic(s, "r*x*(1-x)", highlight_first=True)


def logistic_substitute(s: Scene):
    s.objects += [
        _t("text_1", "Logistic map", 0, 3.0, fs=40),
        _eq("equation_2", r"x_{n+1} = r\,x_n(1 - x_n)", 0, 1.0, scale=1.2),
        _eq("equation_3", r"x_{n+1} = 3.2\,x_n(1 - x_n)", 0, 1.0, scale=1.2),
        _t("text_4", "set r = 3.2", 0, -1.0, fs=32, color=C1),
        _t("text_5", "a period-2 cycle appears", 0, -1.0, fs=32, color=C2),
    ]
    _show(s, "text_1")
    _show(s, "equation_2")
    _show(s, "text_4")
    s.add_step("transform", targets=["equation_2"], to="equation_3", duration=1.5)
    s.add_step("fade", targets=["text_4"], duration=0.8)
    _show(s, "text_5")
    s.add_step("wait", duration=1.0)


# ---------------------------------------------------------------------------
# game theory
# ---------------------------------------------------------------------------

def _hawk_dove(s: Scene, write_dot: bool = True):
    s.objects += [
        _t("text_1", "Hawk-Dove game", 0, 3.4, fs=40),
        _plot("plot_2", ["2 - 3*x", "1 - x"], [0, 1, 0.25], [-1.5, 2.5, 0.5], x=-1.8, y=-0.4,
              w=7, h=5),
        _dot_on("shape_3", "plot_2", "0.5", "0.5", color="#ffffff"),
        _eq("equation_4", r"E[H] = 2 - 3p", 4.4, 1.2, color=C1),
        _eq("equation_5", r"E[D] = 1 - p", 4.4, 0.1, color=C2),
        _t("text_6", "ESS: p = 1/2", 4.4, -1.1, fs=30),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["equation_4", "equation_5"], duration=1.0)
    if write_dot:
        s.add_step("write", targets=["shape_3"], duration=0.5)
    s.add_step("write", targets=["text_6"], duration=1.0)
    s.add_step("wait", duration=1.0)


def hawk_dove(s: Scene):
    _hawk_dove(s)


def hawk_dove_dot_never_shown(s: Scene):
    _hawk_dove(s, write_dot=False)


def _payoff_matrix(s: Scene, cells, fs: float, size: float):
    off = size / 2
    s.objects += [
        _t("text_1", "Prisoner's dilemma", 0, 3.4, fs=40),
        _shape("shape_2", "square", -off, off - 0.4, size=size),
        _shape("shape_3", "square", off, off - 0.4, size=size),
        _shape("shape_4", "square", -off, -off - 0.4, size=size),
        _shape("shape_5", "square", off, -off - 0.4, size=size),
        _t("text_6", cells[0], -off, off - 0.4, fs=fs),
        _t("text_7", cells[1], off, off - 0.4, fs=fs),
        _t("text_8", cells[2], -off, -off - 0.4, fs=fs),
        _t("text_9", cells[3], off, -off - 0.4, fs=fs, color=C1),
        _t("text_10", "Cooperate", -off - size / 2 - 1.2, off - 0.4, fs=28),
        _t("text_11", "Defect", -off - size / 2 - 1.2, -off - 0.4, fs=28),
        _t("text_12", "Cooperate", -off, size - 0.4 + 0.45, fs=28),
        _t("text_13", "Defect", off, size - 0.4 + 0.45, fs=28),
    ]


def payoff_matrix(s: Scene):
    _payoff_matrix(s, ["-1, -1", "-3, 0", "0, -3", "-2, -2"], fs=32, size=1.8)
    _show(s, "text_1")
    s.add_step("write", targets=["shape_2", "shape_3", "shape_4", "shape_5"], duration=1.0)
    s.add_step("write", targets=["text_10", "text_11", "text_12", "text_13"], duration=1.0)
    s.add_step("write", targets=["text_6", "text_7", "text_8", "text_9"], duration=1.5)
    s.add_step("highlight", targets=["text_9"], duration=1.0)


def payoff_text_overflows(s: Scene):
    _payoff_matrix(s, ["-1 yr, -1 yr", "-3 yr, 0 yr", "0 yr, -3 yr", "-2 yr, -2 yr"],
                   fs=36, size=1.6)
    _show(s, "text_1")
    s.add_step("write", targets=["shape_2", "shape_3", "shape_4", "shape_5"], duration=1.0)
    s.add_step("write", targets=["text_10", "text_11", "text_12", "text_13"], duration=1.0)
    s.add_step("write", targets=["text_6", "text_7", "text_8", "text_9"], duration=1.5)
    s.add_step("wait", duration=1.0)


def payoff_fade_unwritten(s: Scene):
    _payoff_matrix(s, ["-1, -1", "-3, 0", "0, -3", "-2, -2"], fs=32, size=1.8)
    s.objects.append(_t("text_14", "Defect dominates", 0, -3.3, fs=32, color=C1))
    _show(s, "text_1")
    s.add_step("write", targets=["shape_2", "shape_3", "shape_4", "shape_5"], duration=1.0)
    s.add_step("write", targets=["text_10", "text_11", "text_12", "text_13"], duration=1.0)
    s.add_step("write", targets=["text_6", "text_7", "text_8", "text_9"], duration=1.5)
    s.add_step("highlight", targets=["text_9"], duration=1.0)
    s.add_step("fade", targets=["text_14"], duration=1.0)
    s.add_step("wait", duration=0.5)


# ---------------------------------------------------------------------------
# map projections
# ---------------------------------------------------------------------------

def mercator(s: Scene):
    s.objects += [
        _t("text_1", "Mercator stretches the poles", 0, 3.4, fs=36),
        _plot("plot_2", ["log(tan(pi/4 + x/2))", "x"], [0, 1.4, 0.2], [0, 3, 0.5], x=-1.8, y=-0.4,
              w=7, h=5, x_label=r"\phi", y_label="y"),
        _t("text_3", "Mercator", 4.5, 0.8, fs=30, color=C1),
        _t("text_4", "Equirectangular", 4.5, -0.2, fs=30, color=C2),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["text_3", "text_4"], duration=1.0)
    s.add_step("wait", duration=1.0)


def _sec_area(s: Scene, x_max_plot: float, area_max: float):
    s.objects += [
        _t("text_1", "Mercator: integrate the secant", 0, 3.4, fs=36),
        _plot("plot_2", ["1/cos(x)"], [0, x_max_plot, 0.2], [0, 6, 1], x=-1.8, y=-0.4, w=7, h=5,
              x_label=r"\phi"),
        _area("area_3", "plot_2", 0, area_max, opacity=0.5),
        _eq("equation_4", r"y(\phi) = \int_0^{\phi} \sec\theta \, d\theta", 4.4, 0.6, scale=0.8),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["area_3"], duration=1.0)
    s.add_step("write", targets=["equation_4"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=1.0)


def mercator_secant_area(s: Scene):
    _sec_area(s, 1.4, 1.0)


def mercator_south_area_empty(s: Scene):
    # southern-hemisphere stretch shaded on axes whose y-range starts at 0: the
    # integral is negative, so the area (and half the curve) has nowhere to go
    s.objects += [
        _t("text_1", "Stretch in the southern hemisphere", 0, 3.4, fs=36),
        _plot("plot_2", ["log(tan(pi/4 + x/2))"], [-1.4, 1.4, 0.2], [0, 3, 0.5], x=-1.6, y=-0.4,
              w=7.5, h=5),
        _area("area_3", "plot_2", -1.2, 0.0, opacity=0.5),
        _eq("equation_4", r"y(-1.2) = \int_0^{-1.2} \sec\theta \, d\theta", 4.6, 0.6, scale=0.7),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=1.5)
    s.add_step("write", targets=["area_3"], duration=1.0)
    s.add_step("write", targets=["equation_4"], duration=1.0, with_previous=True)
    s.add_step("wait", duration=1.0)


# ---------------------------------------------------------------------------
# algorithmic complexity
# ---------------------------------------------------------------------------

def _complexity(s: Scene, legend_color=None):
    s.objects += [
        _t("text_1", "How running time grows", 0, 3.4, fs=40),
        _plot("plot_2", ["x", "x*log2(x)", "x**2"], [0, 20, 5], [0, 400, 100], x=-1.8, y=-0.4,
              w=7, h=5),
        _eq("equation_3", r"O(n)", 4.4, 1.2, color=legend_color or C1),
        _eq("equation_4", r"O(n \log n)", 4.4, 0.1, color=legend_color or C2),
        _eq("equation_5", r"O(n^2)", 4.4, -1.0, color=legend_color or C3),
    ]
    _show(s, "text_1")
    s.add_step("write", targets=["plot_2"], duration=2.0)
    s.add_step("write", targets=["equation_3", "equation_4", "equation_5"], duration=1.5)
    s.add_step("highlight", targets=["equation_5"], duration=1.0)


def complexity_curves(s: Scene):
    _complexity(s)


def complexity_black_legend(s: Scene):
    _complexity(s, legend_color="BLACK")


# ---------------------------------------------------------------------------

CASES: list[Case] = [
    # clean
    Case("holdout4/clean/sir-curves", sir_curves),
    Case("holdout4/clean/sir-peak-dot", sir_peak_dot),
    Case("holdout4/clean/hardy-weinberg", hw_genotypes),
    Case("holdout4/clean/selection-sweep", selection_sweep),
    Case("holdout4/clean/harmonics-transform", harmonics_transform),
    Case("holdout4/clean/standing-wave", standing_wave),
    Case("holdout4/clean/inverse-square", inverse_square),
    Case("holdout4/clean/sound-level", spl_curve),
    Case("holdout4/clean/compound-interest", compound_interest),
    Case("holdout4/clean/rate-tween", rate_tween),
    Case("holdout4/clean/present-value", present_value),
    Case("holdout4/clean/pendulum-track", pendulum_track),
    Case("holdout4/clean/damped-zoom", damped_zoom),
    Case("holdout4/clean/small-angle-swap", small_angle_swap),
    Case("holdout4/clean/logistic-fixed-point", logistic_fixed_point),
    Case("holdout4/clean/logistic-substitute", logistic_substitute),
    Case("holdout4/clean/hawk-dove", hawk_dove),
    Case("holdout4/clean/payoff-matrix", payoff_matrix),
    Case("holdout4/clean/mercator", mercator),
    Case("holdout4/clean/mercator-secant-area", mercator_secant_area),
    Case("holdout4/clean/complexity-curves", complexity_curves),
    # defects
    Case("holdout4/defect/sir-legend-on-curve", sir_legend_on_curve, {"collision"}),
    Case("holdout4/defect/hw-labels-overlap", hw_labels_overlap, {"collision"}),
    Case("holdout4/defect/payoff-text-overflows", payoff_text_overflows, {"collision"}),
    Case("holdout4/defect/title-through-axis", title_through_axis, {"collision"}),
    Case("holdout4/defect/pendulum-label-on-path", pendulum_label_on_path, {"collision"}),
    Case("holdout4/defect/sir-plot-too-wide", sir_plot_too_wide, {"framing"}),
    Case("holdout4/defect/pendulum-dot-degrees", pendulum_dot_degrees, {"framing"}),
    Case("holdout4/defect/period-formula-off-right", period_formula_off_right, {"framing"}),
    Case("holdout4/defect/zoom-label-cut", zoom_label_cut, {"framing"}),
    Case("holdout4/defect/selection-tiny-caption", selection_tiny_caption, {"legibility"}),
    Case("holdout4/defect/hw-tiny-equation", hw_tiny_equation, {"legibility"}),
    Case("holdout4/defect/complexity-black-legend", complexity_black_legend, {"legibility"}),
    Case("holdout4/defect/logistic-highlight-early", logistic_highlight_early, {"timing"}),
    Case("holdout4/defect/payoff-fade-unwritten", payoff_fade_unwritten, {"timing"}),
    Case("holdout4/defect/sir-no-steps", sir_no_steps, {"timing"}),
    Case("holdout4/defect/tween-unused-param", tween_unused_param, {"timing"}),
    Case("holdout4/defect/hawk-dove-dot-never-shown", hawk_dove_dot_never_shown, {"timing"}),
    Case("holdout4/defect/sir-r0-undefined-macro", sir_r0_undefined_macro, {"compile"}),
    Case("holdout4/defect/logistic-missing-star", logistic_missing_star, {"compile"}),
    Case("holdout4/defect/harmonic-aliased", harmonic_aliased, {"fidelity"}),
    Case("holdout4/defect/doubling-smoothed", doubling_smoothed, {"fidelity"}),
    Case("holdout4/defect/beats-clipped", beats_clipped, {"range"}),
    Case("holdout4/defect/continuous-clipped", continuous_clipped, {"range"}),
    Case("holdout4/defect/mercator-south-area-empty", mercator_south_area_empty, {"range"}),
    Case("holdout4/defect/spl-empty-curve", spl_empty_curve, {"range"}),
]
