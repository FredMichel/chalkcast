"""Second independent held-out benchmark (labelled without looking at the verifier).

Scenes are drawn from STEM domains not used in ``bench.py`` / ``bench_holdout.py``
(radioactive decay, RC circuits, oscillators, population dynamics, molecular
potentials, enzyme kinetics, electromagnetism, ML activations, heat flow,
acoustics, optics, blackbody radiation, thermodynamics, numerical methods).
Every label was checked by eye on a rendered final frame.
"""

from __future__ import annotations

from .bench import Case, _eq, _plot, _shape, _show, _t
from .model import Obj


def _dot(i, plot, x, y, color="RED", size=1.0):
    return Obj(id=i, kind="shape", color=color,
               payload={"shape": "dot", "size": size, "anchor": {"plot": plot, "x": x, "y": y}})


def _arrow(i, x1, y1, x2, y2, color="YELLOW", shape="arrow"):
    return Obj(id=i, kind="shape", color=color,
               payload={"shape": shape, "x1": x1, "y1": y1, "x2": x2, "y2": y2})


def _riem(i, plot, a, b, dx, color="TEAL"):
    return Obj(id=i, kind="riemann", color=color,
               payload={"plot": plot, "fn": 1, "x_min": a, "x_max": b, "dx": dx})


# ---------------------------------------------------------------------------
# reusable scene bodies
# ---------------------------------------------------------------------------

def _decay_plot(y=-0.6, h=4.6):
    return _plot("plot_2", [{"expr": "np.exp(-0.0862*x)", "color": "GREEN"}], [0, 40, 8], [0, 1.1, 0.25],
                 x=-1.8, y=y, w=8, h=h)


def _logistic(s):
    s.objects += [_t("text_1", "Logistic population growth", y=3.3),
                  _plot("plot_2", [{"expr": "100/(1+9*np.exp(-0.8*x))", "color": "GREEN"},
                                   {"expr": "100+0*x", "color": "GRAY"}],
                        [0, 10, 2], [0, 120, 20], x=-1.6, y=-0.7, w=8, h=4.6)]


def _michaelis(s, title="Michaelis–Menten kinetics", fs=36):
    s.objects += [_t("text_1", title, y=3.3, fs=fs),
                  _plot("plot_2", [{"expr": "10*x/(2+x)", "color": "GREEN"}, {"expr": "10+0*x", "color": "GRAY"}],
                        [0, 20, 5], [0, 12, 2], x=-2.0, y=-0.6, w=8, h=4.6)]


def _planck_plot():
    fns = [{"expr": f"10/(x**5*(np.exp(14388/(x*{T}))-1))", "color": c}
           for T, c in ((5500, "BLUE"), (5000, "YELLOW"), (4500, "RED"))]
    return _plot("plot_2", fns, [0.2, 3, 0.5], [0, 2, 0.5], x=-1.5, y=-0.6, w=8, h=4.6)


def _snell(s, glass_at=(-4.5, -1.0)):
    s.objects += [_t("text_1", "Refraction: Snell's law", y=3.3),
                  _shape("shape_2", "line", 0, 0, "GRAY", size=12),
                  _arrow("shape_3", 0, 2.5, 0, -2.5, "GRAY", shape="line"),
                  _arrow("shape_4", -2.5, 2.5, 0, 0, "YELLOW"),
                  _arrow("shape_5", 0, 0, 1.2, -2.5, "YELLOW"),
                  _t("text_6", "air  n₁ = 1.00", -4.5, 1.0, 28),
                  _t("text_7", "glass  n₂ = 1.50", *glass_at, 28),
                  _eq("equation_8", r"n_1\sin\theta_1 = n_2\sin\theta_2", 4.0, 1.5, 0.8)]
    _show(s, "text_1", "shape_2", "shape_3", "shape_4", "shape_5", "text_6", "text_7", "equation_8")


_MAXWELL = [r"\nabla\cdot\mathbf{E} = \frac{\rho}{\varepsilon_0}",
            r"\nabla\cdot\mathbf{B} = 0",
            r"\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}",
            r"\nabla\times\mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial\mathbf{E}}{\partial t}"]


def _maxwell(s, ys):
    s.objects.append(_t("text_1", "Maxwell's equations", y=3.3))
    for k, (tex, y) in enumerate(zip(_MAXWELL, ys), start=2):
        s.objects.append(_eq(f"equation_{k}", tex, y=y, scale=0.9))
    _show(s, "text_1", "equation_2", "equation_3", "equation_4", "equation_5")


# ---------------------------------------------------------------------------
# clean scenes
# ---------------------------------------------------------------------------

def c_radioactive_decay(s):
    s.objects += [_t("text_1", "Radioactive decay of iodine-131", y=3.3),
                  _decay_plot(),
                  _dot("shape_3", "plot_2", "8.04", "0.5"),
                  _t("text_4", "half-life ≈ 8 days", 4.6, 1.0, 26),
                  _eq("equation_5", r"N(t) = N_0\,e^{-\lambda t}", 4.6, -0.2, 0.8)]
    _show(s, "text_1", "plot_2", "shape_3", "text_4", "equation_5")
    s.add_step("highlight", targets=["shape_3"])


def c_rc_charging(s):
    s.params["tau"] = 0.5
    s.objects += [_t("text_1", "Charging an RC circuit", y=3.3),
                  _eq("equation_2", r"V(t) = V_0\left(1 - e^{-t/\tau}\right)", y=2.3, scale=0.8),
                  _plot("plot_3", [{"expr": "1-np.exp(-x/tau)", "color": "YELLOW"}], [0, 5, 1], [0, 1.2, 0.5],
                        y=-0.9, w=9, h=4)]
    _show(s, "text_1", "equation_2", "plot_3")
    s.add_step("tween", param="tau", value=2.0, duration=3)


def c_damped_oscillator(s):
    s.objects += [_t("text_1", "Underdamped harmonic oscillator", y=3.3, fs=34),
                  _plot("plot_2", [{"expr": "np.exp(-0.25*x)*np.cos(3*x)", "color": "YELLOW"},
                                   {"expr": "np.exp(-0.25*x)", "color": "GRAY"},
                                   {"expr": "-np.exp(-0.25*x)", "color": "GRAY"}],
                        [0, 12, 2], [-1.2, 1.2, 0.5], x=-1.5, y=-0.5, w=9, h=4.5),
                  _t("text_3", "envelope ±e^(−γt)", 5.2, 1.0, 24, "GRAY"),
                  _eq("equation_4", r"x(t) = e^{-\gamma t}\cos(\omega_d t)", 5.0, -1.6, 0.7)]
    _show(s, "text_1", "plot_2", "text_3", "equation_4")


def c_logistic_growth(s):
    _logistic(s)
    s.objects += [_t("text_3", "carrying capacity K", 4.7, 0.83, 26, "GRAY"),
                  _eq("equation_4", r"\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)", 4.6, -1.5, 0.7)]
    _show(s, "text_1", "plot_2", "text_3", "equation_4")


def c_lennard_jones(s):
    s.objects += [_t("text_1", "Lennard-Jones potential", y=3.3),
                  _plot("plot_2", [{"expr": "4*(x**-12 - x**-6)", "color": "TEAL"}], [0.9, 3, 0.5], [-1.2, 2, 0.5],
                        x=-2.0, y=-0.5, w=8, h=5),
                  _dot("shape_3", "plot_2", "2**(1/6)", "-1"),
                  _t("text_4", "equilibrium r₀ = 2^(1/6) σ", 4.7, -2.0, 24),
                  _eq("equation_5", r"V(r) = 4\varepsilon\left[\left(\tfrac{\sigma}{r}\right)^{12}"
                                    r" - \left(\tfrac{\sigma}{r}\right)^{6}\right]", 4.5, 1.4, 0.75)]
    _show(s, "text_1", "plot_2", "shape_3", "text_4", "equation_5")


def c_michaelis_menten(s):
    _michaelis(s)
    s.objects += [_dot("shape_3", "plot_2", "2", "5"),
                  _eq("equation_4", r"v = \frac{V_{max}[S]}{K_m + [S]}", 4.6, 1.0, 0.75),
                  _t("text_5", "v = Vmax/2 at [S] = Km", 4.6, -1.0, 24)]
    _show(s, "text_1", "plot_2", "shape_3", "equation_4", "text_5")


def c_maxwell(s):
    _maxwell(s, (2.0, 0.7, -0.6, -1.9))


def c_activations(s):
    s.objects += [_t("text_1", "Activation functions", y=3.3),
                  _t("text_2", "ReLU", -3.6, 2.5, 30, "YELLOW"),
                  _t("text_3", "sigmoid", 3.6, 2.5, 30, "BLUE"),
                  _plot("plot_4", [{"expr": "np.maximum(0, x)", "color": "YELLOW"}], [-3, 3, 1], [-0.5, 3, 1],
                        x=-3.6, y=-0.6, w=5.5, h=4),
                  _plot("plot_5", [{"expr": "1/(1+np.exp(-x))", "color": "BLUE"}], [-6, 6, 2], [0, 1.2, 0.5],
                        x=3.6, y=-0.6, w=5.5, h=4)]
    _show(s, "text_1", "plot_4", "text_2", "plot_5", "text_3")


def c_heat_equation(s):
    s.params["t"] = 0.0
    s.objects += [_t("text_1", "Heat equation: decay of modes", y=3.3, fs=34),
                  _eq("equation_2", r"u_t = \alpha\,u_{xx}", y=2.4, scale=0.8),
                  _plot("plot_3", [{"expr": "np.exp(-t)*np.sin(np.pi*x) + 0.3*np.exp(-9*t)*np.sin(3*np.pi*x)",
                                    "color": "ORANGE"}], [0, 1, 0.25], [-0.5, 1.5, 0.5], y=-0.9, w=9, h=4)]
    _show(s, "text_1", "equation_2", "plot_3")
    s.add_step("tween", param="t", value=1.5, duration=4)


def c_beats(s):
    s.objects += [_t("text_1", "Acoustic beats", y=3.3),
                  _plot("plot_2", [{"expr": "np.sin(20*x)+np.sin(22*x)", "color": "BLUE"},
                                   {"expr": "2*np.cos(x)", "color": "GRAY"},
                                   {"expr": "-2*np.cos(x)", "color": "GRAY"}],
                        [0, 10, 2], [-2.5, 2.5, 1], y=-0.5, w=11, h=4.5),
                  _t("text_3", "f_beat = |f₁ − f₂|", y=-3.4, fs=28)]
    _show(s, "text_1", "plot_2", "text_3")


def c_snell(s):
    _snell(s)


def c_planck(s):
    s.objects += [_t("text_1", "Blackbody spectrum", y=3.3), _planck_plot(),
                  _t("text_3", "5500 K", 4.6, 1.2, 26, "BLUE"),
                  _t("text_4", "5000 K", 4.6, 0.6, 26, "YELLOW"),
                  _t("text_5", "4500 K", 4.6, 0.0, 26, "RED"),
                  _eq("equation_6", r"\lambda_{max}\,T = b", 4.6, -1.5, 0.8)]
    _show(s, "text_1", "plot_2", "text_3", "text_4", "text_5", "equation_6")


def c_sawtooth_auto(s):  # jump discontinuities, no smooth/dx override
    s.objects += [_t("text_1", "Sawtooth wave", y=3.3),
                  _plot("plot_2", [{"expr": "x - np.floor(x)", "color": "YELLOW"}], [-3, 3, 1], [-0.5, 1.5, 0.5],
                        y=-0.5, w=10, h=4),
                  _t("text_3", "period T = 1", y=-3.3, fs=28)]
    _show(s, "text_1", "plot_2", "text_3")


def c_ideal_gas_box(s):
    s.objects += [_t("text_1", "Thermodynamics", y=3.3),
                  _eq("equation_2", r"PV = nRT", 0, 0.3, 1.2),
                  _shape("shape_3", "rectangle", 0, 0.3, "YELLOW", width=5, height=1.8),
                  _t("text_4", "ideal gas law", 0, -1.5, 30),
                  _t("text_5", "isothermal: PV = const", 0, -1.5, 30)]
    _show(s, "text_1", "equation_2", "shape_3", "text_4")
    s.add_step("transform", targets=["text_4"], to="text_5", duration=1.5)


def c_newton_method(s):
    s.objects += [_t("text_1", "Newton's method for √2", y=3.3),
                  _plot("plot_2", [{"expr": "x**2-2", "color": "YELLOW"}, {"expr": "4*(x-2)+2", "color": "BLUE"}],
                        [0, 3, 1], [-3, 7, 2], x=-1.5, y=-0.6, w=7, h=5),
                  _dot("shape_3", "plot_2", "2", "2"),
                  _dot("shape_4", "plot_2", "1.5", "0", "GREEN"),
                  _eq("equation_5", r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}", 4.6, 1.0, 0.8),
                  _t("text_6", "x₁ = 1.5", 4.6, -1.0, 28, "GREEN")]
    _show(s, "text_1", "plot_2", "shape_3", "shape_4", "equation_5", "text_6")


def c_spring_work_riemann(s):
    s.objects += [_t("text_1", "Work done by a spring", y=3.3),
                  _plot("plot_2", [{"expr": "2*x", "color": "YELLOW"}], [0, 3.5, 1], [0, 7, 2],
                        x=-1.8, y=-0.6, w=7, h=4.6),
                  _riem("riemann_3", "plot_2", 0, 3, 0.5),
                  _riem("riemann_4", "plot_2", 0, 3, 0.1),
                  _eq("equation_5", r"W = \int_0^3 kx\,dx = 9\,\mathrm{J}", 4.5, 0.8, 0.7)]
    _show(s, "text_1", "plot_2", "riemann_3", "equation_5")
    s.add_step("transform", targets=["riemann_3"], to="riemann_4", duration=2)


# ---------------------------------------------------------------------------
# defect scenes -- framing
# ---------------------------------------------------------------------------

def d_title_too_wide(s):
    _michaelis(s, title="Michaelis–Menten kinetics: reaction velocity saturates with substrate", fs=40)
    _show(s, "text_1", "plot_2")


def d_plot_too_low(s):
    s.objects += [_t("text_1", "Radioactive decay of iodine-131", y=3.3), _decay_plot(y=-1.7, h=5.6)]
    _show(s, "text_1", "plot_2")


def d_equation_past_right_edge(s):
    _logistic(s)
    s.objects.append(_eq("equation_3", r"\frac{dP}{dt} = rP\left(1-\frac{P}{K}\right) - \frac{hP^2}{1+P^2}",
                         5.6, -0.5, 0.9))
    _show(s, "text_1", "plot_2", "equation_3")


def d_dot_runs_off(s):  # tween drives the anchored dot past the axes and off screen
    s.params["t"] = 0.0
    s.objects += [_t("text_1", "Exponential decay", y=3.3),
                  _plot("plot_2", [{"expr": "np.exp(-0.2*x)", "color": "GREEN"}], [0, 10, 2], [0, 1.2, 0.5],
                        y=-0.6, w=9, h=4.5),
                  _dot("shape_3", "plot_2", "t", "np.exp(-0.2*t)")]
    _show(s, "text_1", "plot_2", "shape_3")
    s.add_step("tween", param="t", value=18, duration=4)


def d_footnote_below_frame(s):
    s.objects += [_t("text_1", "Blackbody spectrum", y=3.3), _planck_plot(),
                  _t("text_3", "Data: CIE standard illuminants", y=-4.0, fs=24, color="GRAY")]
    _show(s, "text_1", "plot_2", "text_3")


# ---------------------------------------------------------------------------
# defect scenes -- collision
# ---------------------------------------------------------------------------

def d_legend_overlap(s):
    s.objects += [_t("text_1", "Blackbody spectrum", y=3.3), _planck_plot(),
                  _t("text_3", "5500 K", 4.6, 1.0, 26, "BLUE"),
                  _t("text_4", "5000 K", 4.6, 0.8, 26, "YELLOW"),
                  _t("text_5", "4500 K", 4.6, 0.6, 26, "RED")]
    _show(s, "text_1", "plot_2", "text_3", "text_4", "text_5")


def d_label_on_inflection(s):
    _logistic(s)
    s.objects.append(_t("text_3", "inflection point", -3.4, -1.08, 28))
    _show(s, "text_1", "plot_2", "text_3")


def d_equation_on_axis_tip(s):
    s.objects += [_plot("plot_1", [{"expr": "x**3/9 - x", "color": "YELLOW"}], [-4, 4, 1], [-3, 3, 1],
                        y=-0.4, w=9, h=5.8),
                  _eq("equation_2", r"f(x) = \frac{x^3}{9} - x", y=2.5, scale=0.9)]
    _show(s, "plot_1", "equation_2")


def d_text_spills_box(s):
    s.objects += [_t("text_1", "Conservation of momentum", 0, 0, 36),
                  _shape("shape_2", "rectangle", 0, 0, "YELLOW", width=3, height=1),
                  _eq("equation_3", r"m_1v_1 + m_2v_2 = m_1v_1' + m_2v_2'", y=-1.8, scale=0.9)]
    _show(s, "shape_2", "text_1", "equation_3")


def d_title_on_equation(s):
    s.objects += [_t("text_1", "Geometric series", y=3.2, fs=40),
                  _eq("equation_2", r"\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r}", y=2.65),
                  _eq("equation_3", r"|r| < 1", y=1.0, scale=0.8)]
    _show(s, "text_1", "equation_2", "equation_3")


def d_arrow_through_label(s):
    _snell(s, glass_at=(0.8, -1.2))


def d_maxwell_cramped(s):
    _maxwell(s, (1.6, 1.0, 0.4, -0.2))


# ---------------------------------------------------------------------------
# defect scenes -- legibility
# ---------------------------------------------------------------------------

def d_tiny_axis_note(s):
    s.objects += [_t("text_1", "Radioactive decay of iodine-131", y=3.3), _decay_plot(),
                  _t("text_3", "time in days, activity normalised to t = 0", 0, -3.55, 13, "GRAY")]
    _show(s, "text_1", "plot_2", "text_3")


def d_scaled_down_caption(s):
    s.objects += [_t("text_1", "Acoustic beats", y=3.3),
                  _plot("plot_2", [{"expr": "np.sin(20*x)+np.sin(22*x)", "color": "BLUE"}],
                        [0, 10, 2], [-2.5, 2.5, 1], y=-0.3, w=11, h=4.5),
                  _t("text_3", "two tones at 440 Hz and 444 Hz give 4 beats per second", y=-3.3, fs=36, scale=0.42)]
    _show(s, "text_1", "plot_2", "text_3")


def d_tiny_equation(s):
    _michaelis(s)
    s.objects.append(_eq("equation_3", r"v = \frac{V_{max}[S]}{K_m + [S]}", 4.6, 1.0, 0.3))
    _show(s, "text_1", "plot_2", "equation_3")


# ---------------------------------------------------------------------------
# defect scenes -- timing
# ---------------------------------------------------------------------------

def d_transform_to_missing(s):
    _logistic(s)
    s.objects.append(_t("text_3", "r = 0.8", 4.6, 0.5, 30))
    _show(s, "text_1", "plot_2", "text_3")
    s.add_step("transform", targets=["text_3"], to="text_9", duration=1)


def d_fade_never_written(s):
    s.objects += [_t("text_1", "Thermodynamics", y=3.3),
                  _eq("equation_2", r"PV = nRT", 0, 0.3, 1.2),
                  _t("text_3", "ideal gas law", 0, -1.5, 30)]
    _show(s, "text_1", "equation_2")
    s.add_step("fade", targets=["text_3"])


def d_highlight_before_dot(s):
    s.objects += [_t("text_1", "Radioactive decay of iodine-131", y=3.3), _decay_plot(),
                  _dot("shape_3", "plot_2", "8.04", "0.5")]
    s.add_step("highlight", targets=["shape_3"])
    _show(s, "text_1", "plot_2", "shape_3")


def d_typo_target(s):
    s.objects += [_t("text_1", "Newton's method for √2", y=3.3),
                  _eq("equation_2", r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}", y=0.5)]
    _show(s, "text_1", "equaton_2")


def d_wrong_param_name(s):
    s.params["w"] = 1.0
    s.objects += [_t("text_1", "Driven oscillator", y=3.3),
                  _plot("plot_2", [{"expr": "np.sin(w*x)", "color": "YELLOW"}], [0, 10, 2], [-1.5, 1.5, 1],
                        y=-0.5, w=10, h=4)]
    _show(s, "text_1", "plot_2")
    s.add_step("tween", param="omega", value=3.0, duration=3)


def d_label_never_shown(s):
    _michaelis(s)
    s.objects += [_eq("equation_3", r"v = \frac{V_{max}[S]}{K_m + [S]}", 4.6, 1.0, 0.75),
                  _t("text_4", "saturation at Vmax", 4.6, -1.0, 26)]
    _show(s, "text_1", "plot_2", "equation_3")


def d_transform_before_intro(s):
    s.objects += [_t("text_1", "Thermodynamics", y=3.3),
                  _t("text_2", "ideal gas law", 0, 0, 34),
                  _t("text_3", "isothermal: PV = const", 0, 0, 34)]
    _show(s, "text_1")
    s.add_step("transform", targets=["text_2"], to="text_3")
    _show(s, "text_2")


# ---------------------------------------------------------------------------
# defect scenes -- compile
# ---------------------------------------------------------------------------

def d_mhchem(s):
    s.objects += [_t("text_1", "Combustion of hydrogen", y=3.3),
                  _eq("equation_2", r"\ce{2H2 + O2 -> 2H2O}", y=0.5)]
    _show(s, "text_1", "equation_2")


def d_siunitx(s):
    s.objects += [_t("text_1", "Free fall", y=3.3),
                  _eq("equation_2", r"g = \SI{9.81}{\metre\per\second\squared}", y=0.5)]
    _show(s, "text_1", "equation_2")


def d_double_superscript(s):
    s.objects += [_t("text_1", "Normal distribution", y=3.3),
                  _eq("equation_2", r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-(x-\mu)^2/2\sigma}^2", y=0.5)]
    _show(s, "text_1", "equation_2")


def d_erf_expr(s):
    s.objects += [_t("text_1", "Diffusion from a step", y=3.3),
                  _plot("plot_2", [{"expr": "0.5*(1 - erf(x/2))", "color": "BLUE"}], [-6, 6, 2], [0, 1.2, 0.5],
                        y=-0.6, w=10, h=4.5)]
    _show(s, "text_1", "plot_2")


def d_ln_expr(s):
    s.objects += [_t("text_1", "Entropy of mixing", y=3.3),
                  _plot("plot_2", [{"expr": "-(x*ln(x) + (1-x)*ln(1-x))", "color": "GREEN"}], [0, 1, 0.25],
                        [0, 0.8, 0.2], y=-0.6, w=8, h=4.5)]
    _show(s, "text_1", "plot_2")


# ---------------------------------------------------------------------------
# defect scenes -- fidelity
# ---------------------------------------------------------------------------

def d_sawtooth_forced_smooth(s):
    s.objects += [_t("text_1", "Sawtooth wave", y=3.3),
                  _plot("plot_2", [{"expr": "x - np.floor(x)", "color": "YELLOW", "smooth": True, "dx": 0.1}],
                        [-3, 3, 1], [-0.5, 1.5, 0.5], y=-0.5, w=10, h=4)]
    _show(s, "text_1", "plot_2")


def d_chirp_coarse(s):
    s.objects += [_t("text_1", "Linear chirp", y=3.3),
                  _plot("plot_2", [{"expr": "np.sin(x**2)", "color": "BLUE", "dx": 0.4}], [0, 8, 1], [-1.5, 1.5, 1],
                        y=-0.5, w=11, h=4)]
    _show(s, "text_1", "plot_2")


def d_staircase_coarse(s):
    s.objects += [_t("text_1", "Quantised energy levels", y=3.3),
                  _plot("plot_2", [{"expr": "np.floor(x)", "color": "YELLOW", "dx": 0.5}], [0, 5, 1], [0, 5, 1],
                        y=-0.5, w=8, h=5)]
    _show(s, "text_1", "plot_2")


def d_beats_aliased(s):
    s.objects += [_t("text_1", "Acoustic beats", y=3.3),
                  _plot("plot_2", [{"expr": "np.sin(20*x)+np.sin(22*x)", "color": "BLUE", "dx": 0.25}],
                        [0, 10, 2], [-2.5, 2.5, 1], y=-0.5, w=11, h=4.5)]
    _show(s, "text_1", "plot_2")


_C, _D = "holdout2/clean/", "holdout2/defect/"

CASES: list[Case] = [
    Case(_C + "radioactive_decay", c_radioactive_decay),
    Case(_C + "rc_charging_tween", c_rc_charging),
    Case(_C + "damped_oscillator", c_damped_oscillator),
    Case(_C + "logistic_growth", c_logistic_growth),
    Case(_C + "lennard_jones", c_lennard_jones),
    Case(_C + "michaelis_menten", c_michaelis_menten),
    Case(_C + "maxwell", c_maxwell),
    Case(_C + "activations", c_activations),
    Case(_C + "heat_equation_tween", c_heat_equation),
    Case(_C + "beats", c_beats),
    Case(_C + "snell", c_snell),
    Case(_C + "planck", c_planck),
    Case(_C + "sawtooth_auto", c_sawtooth_auto),
    Case(_C + "ideal_gas_box", c_ideal_gas_box),
    Case(_C + "newton_method", c_newton_method),
    Case(_C + "spring_work_riemann", c_spring_work_riemann),

    Case(_D + "title_too_wide", d_title_too_wide, {"framing"}),
    Case(_D + "plot_too_low", d_plot_too_low, {"framing"}),
    Case(_D + "equation_past_right_edge", d_equation_past_right_edge, {"framing"}),
    Case(_D + "dot_runs_off", d_dot_runs_off, {"framing"}),
    Case(_D + "footnote_below_frame", d_footnote_below_frame, {"framing"}),

    Case(_D + "legend_overlap", d_legend_overlap, {"collision"}),
    Case(_D + "label_on_inflection", d_label_on_inflection, {"collision"}),
    Case(_D + "equation_on_axis_tip", d_equation_on_axis_tip, {"collision"}),
    Case(_D + "text_spills_box", d_text_spills_box, {"collision"}),
    Case(_D + "title_on_equation", d_title_on_equation, {"collision"}),
    Case(_D + "arrow_through_label", d_arrow_through_label, {"collision"}),
    Case(_D + "maxwell_cramped", d_maxwell_cramped, {"collision"}),

    Case(_D + "tiny_axis_note", d_tiny_axis_note, {"legibility"}),
    Case(_D + "scaled_down_caption", d_scaled_down_caption, {"legibility"}),
    Case(_D + "tiny_equation", d_tiny_equation, {"legibility"}),

    Case(_D + "transform_to_missing", d_transform_to_missing, {"timing"}),
    Case(_D + "fade_never_written", d_fade_never_written, {"timing"}),
    Case(_D + "highlight_before_dot", d_highlight_before_dot, {"timing"}),
    Case(_D + "typo_target", d_typo_target, {"timing"}),
    Case(_D + "wrong_param_name", d_wrong_param_name, {"timing"}),
    Case(_D + "label_never_shown", d_label_never_shown, {"timing"}),
    Case(_D + "transform_before_intro", d_transform_before_intro, {"timing"}),

    Case(_D + "mhchem", d_mhchem, {"compile"}),
    Case(_D + "siunitx", d_siunitx, {"compile"}),
    Case(_D + "double_superscript", d_double_superscript, {"compile"}),
    Case(_D + "erf_expr", d_erf_expr, {"compile"}),
    Case(_D + "ln_expr", d_ln_expr, {"compile"}),

    Case(_D + "sawtooth_forced_smooth", d_sawtooth_forced_smooth, {"fidelity"}),
    Case(_D + "chirp_coarse", d_chirp_coarse, {"fidelity"}),
    # Relabelled clean: written when a coarse dx drew slanted ramps between the
    # steps; curves are now split at jumps even with dx given, and the final frame
    # shows exact horizontal steps (checked visually), so there is no defect left.
    Case(_C + "staircase_coarse", d_staircase_coarse, set()),
    Case(_D + "beats_aliased", d_beats_aliased, {"fidelity"}),
]
