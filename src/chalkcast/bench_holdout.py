"""Held-out verifier benchmark: independently authored, visually checked cases.

These scenes were written without reference to the verifier's logic and each
label was confirmed by rendering the final frame. They are meant to measure the
detection rate and false-positive rate on content the detector was not tuned on.

    from chalkcast.bench_holdout import CASES
"""

from __future__ import annotations

from .bench import Case, _eq, _plot, _shape, _show, _t
from .model import Obj

# ---------------------------------------------------------------------------
# clean scenes
# ---------------------------------------------------------------------------


def c_projectile(s):  # kinematics: title, plot on the left, equations on the right
    s.objects += [
        _t("text_1", "Projectile motion", y=3.2, fs=40),
        _plot("plot_2", [{"expr": "2*x - 0.5*x**2", "color": "GREEN"}], [0, 4, 1], [0, 2.5, 1],
              x=-2.8, y=-0.6, w=6, h=4),
        _eq("equation_3", r"y(t) = v_0 t - \tfrac{1}{2} g t^2", 3.9, 0.6, 0.8),
        _eq("equation_4", r"t_{\max} = \frac{v_0}{g}", 3.9, -1.0, 0.8, color="GREEN"),
    ]
    _show(s, "text_1", "plot_2", "equation_3", "equation_4")
    s.add_step("highlight", targets=["equation_4"])


def c_bayes_callout(s):  # probability: boxed theorem + caption
    s.objects += [
        _t("text_1", "Bayes' theorem", y=2.8, fs=38),
        _eq("equation_2", r"P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}", y=0.3),
        _shape("shape_3", "rectangle", 0, 0.3, "YELLOW", width=7.0, height=2.0),
        _t("text_4", "posterior = likelihood x prior / evidence", y=-2.3, fs=28, color="GRAY"),
    ]
    _show(s, "text_1", "equation_2", "shape_3", "text_4")


def c_matrix_product(s):  # linear algebra: a 2x2 product, centered
    s.objects += [
        _t("text_1", "Matrix-vector product", y=2.9, fs=36),
        _eq("equation_2", r"\begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}"
                          r"\begin{pmatrix} x \\ y \end{pmatrix} = "
                          r"\begin{pmatrix} 2x + y \\ 3y \end{pmatrix}", y=0.2),
        _t("text_3", "each row dotted with the vector", y=-2.4, fs=26, color="BLUE"),
    ]
    _show(s, "text_1", "equation_2", "text_3")


def c_gaussian_area(s):  # probability: shaded one-sigma band, labels outside the axes
    s.objects += [
        _t("text_1", "Standard normal: 68% within one sigma", y=3.3, fs=30),
        _plot("plot_2", [{"expr": "np.exp(-x**2/2)/np.sqrt(2*np.pi)", "color": "YELLOW"}],
              [-3.5, 3.5, 1], [0, 0.5, 0.1], y=-0.5, w=9, h=4.5),
        Obj(id="area_3", kind="area", position=(0, -0.5), color="BLUE",
            payload={"plot": "plot_2", "fn": 1, "x_min": -1, "x_max": 1, "opacity": 0.45}),
    ]
    _show(s, "text_1", "plot_2", "area_3")


def c_taylor_transform(s):  # calculus: Taylor polynomials morphing on shared axes
    xr, yr = [-4, 4, 1], [-2, 2, 1]
    s.objects += [
        _t("text_1", "Taylor series of sin x", y=3.3, fs=34),
        _plot("plot_2", [{"expr": "np.sin(x)", "color": "GRAY"}], xr, yr, x=-1.2, y=-0.6, w=8, h=4.5),
        _plot("plot_3", [{"expr": "x", "color": "ORANGE"}], xr, yr, x=-1.2, y=-0.6, w=8, h=4.5),
        _plot("plot_4", [{"expr": "x - x**3/6", "color": "ORANGE"}], xr, yr, x=-1.2, y=-0.6, w=8, h=4.5),
        _plot("plot_5", [{"expr": "x - x**3/6 + x**5/120", "color": "ORANGE"}], xr, yr,
              x=-1.2, y=-0.6, w=8, h=4.5),
        _t("text_6", "order 1", 5.0, 1.0, 30, "ORANGE"),
        _t("text_7", "order 3", 5.0, 1.0, 30, "ORANGE"),
        _t("text_8", "order 5", 5.0, 1.0, 30, "ORANGE"),
    ]
    _show(s, "text_1", "plot_2", "plot_3", "text_6")
    s.add_step("transform", targets=["plot_3"], to="plot_4", duration=1.2)
    s.add_step("transform", targets=["text_6"], to="text_7", duration=0.5, with_previous=True)
    s.add_step("transform", targets=["plot_3"], to="plot_5", duration=1.2)
    s.add_step("transform", targets=["text_6"], to="text_8", duration=0.5, with_previous=True)


def c_arrhenius_bullets(s):  # chemistry: equation + left-aligned bullet column
    s.objects += [
        _t("text_1", "Arrhenius equation", y=3.0, fs=40),
        _eq("equation_2", r"k = A\, e^{-E_a / (R T)}", y=1.5, scale=1.1, color="TEAL"),
        _t("text_3", "A : pre-exponential factor", y=-0.2, fs=28),
        _t("text_4", "Ea : activation energy", y=-1.1, fs=28),
        _t("text_5", "T : absolute temperature", y=-2.0, fs=28),
    ]
    _show(s, "text_1", "equation_2", "text_3", "text_4", "text_5")


def c_unit_circle(s):  # trigonometry: circle, radius arrow, labels outside the outline
    s.objects += [
        _t("text_1", "The unit circle", x=-3.8, y=3.0, fs=36),
        _shape("shape_2", "circle", -3.0, -0.5, "BLUE", size=4.0),
        Obj(id="shape_3", kind="shape", color="YELLOW",
            payload={"shape": "arrow", "x1": -3.0, "y1": -0.5, "x2": -1.59, "y2": 0.91}),
        _shape("shape_4", "dot", -3.0, -0.5, "WHITE", size=0.6),
        _eq("equation_5", r"x^2 + y^2 = 1", 3.5, 0.8),
        _eq("equation_6", r"(\cos\theta,\ \sin\theta)", 3.5, -0.8, color="YELLOW"),
    ]
    _show(s, "text_1", "shape_2", "shape_4", "shape_3", "equation_5", "equation_6")


def c_riemann_sqrt(s):  # calculus: Riemann rectangles refining, integral on the right
    s.objects += [
        _plot("plot_1", [{"expr": "np.sqrt(x)", "color": "YELLOW"}], [0, 4, 1], [0, 2.2, 1],
              x=-2.3, y=-0.3, w=7, h=4.5),
        Obj(id="riemann_2", kind="riemann", color="PURPLE",
            payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 4, "dx": 0.5}),
        Obj(id="riemann_3", kind="riemann", color="PURPLE",
            payload={"plot": "plot_1", "fn": 1, "x_min": 0, "x_max": 4, "dx": 0.1}),
        _eq("equation_4", r"\int_0^4 \sqrt{x}\,dx = \frac{16}{3}", 4.4, 0.0, 0.8),
        _t("text_5", "Left Riemann sums", y=3.3, fs=32),
    ]
    _show(s, "text_5", "plot_1", "riemann_2", "equation_4")
    s.add_step("transform", targets=["riemann_2"], to="riemann_3", duration=2)


def c_damped_envelope(s):  # physics: damped oscillator with its envelope (3 curves)
    s.objects += [
        _t("text_1", "Damped harmonic oscillator", y=3.3, fs=34),
        _plot("plot_2", [{"expr": "np.exp(-0.3*x)*np.cos(3*x)", "color": "YELLOW"},
                         {"expr": "np.exp(-0.3*x)", "color": "RED"},
                         {"expr": "-np.exp(-0.3*x)", "color": "RED"}],
              [0, 10, 2], [-1.2, 1.2, 0.5], x=-1.6, y=-0.5, w=8.5, h=4.5),
        _t("text_3", "x(t)", 4.9, 0.6, 30, "YELLOW"),
        _t("text_4", "envelope", 4.9, -0.4, 30, "RED"),
    ]
    _show(s, "text_1", "plot_2", "text_3", "text_4")


def c_ideal_gas_isotherm(s):  # chemistry: isotherm with an operating point dot on it
    s.objects += [
        _eq("equation_1", r"PV = nRT", y=3.0, scale=1.1),
        _plot("plot_2", [{"expr": "2/x", "color": "BLUE"}], [0.5, 5, 1], [0, 4.5, 1],
              x=-1.8, y=-0.8, w=7, h=4.5, x_label="V", y_label="P"),
        Obj(id="shape_3", kind="shape", color="RED",
            payload={"shape": "dot", "size": 1.0, "anchor": {"plot": "plot_2", "x": "2", "y": "1"}}),
        _t("text_4", "T = const", 4.6, 0.5, 30, "BLUE"),
    ]
    _show(s, "equation_1", "plot_2", "shape_3", "text_4")


def c_heaviside_auto(s):  # signals: step function with no explicit smoothing (auto-handled)
    s.objects += [
        _t("text_1", "Heaviside step response", y=3.2, fs=34),
        _plot("plot_2", [{"expr": "heaviside(x - 1, 1) * (1 - np.exp(-(x - 1)))", "color": "GREEN"}],
              [-2, 6, 1], [-0.2, 1.2, 0.5], y=-0.6, w=10, h=4.5),
    ]
    _show(s, "text_1", "plot_2")


def c_gradient_two_columns(s):  # multivariable calculus: two columns joined by an arrow
    s.objects += [
        _t("text_1", "Gradient of a scalar field", y=3.0, fs=36),
        _eq("equation_2", r"f(x,y) = x^2 y + \sin y", -3.6, 0.3, 0.9),
        _shape("shape_3", "arrow", 0.0, 0.3, "YELLOW", size=1.2),
        _eq("equation_4", r"\nabla f = \begin{pmatrix} 2xy \\ x^2 + \cos y \end{pmatrix}", 3.6, 0.3, 0.9),
    ]
    _show(s, "text_1", "equation_2", "shape_3", "equation_4")


def c_pendulum_tween(s):  # physics: dot travelling along a potential well, param declared
    s.params["u"] = -2.0
    s.objects += [
        _t("text_1", "Particle in a potential well", y=3.2, fs=32),
        _plot("plot_2", [{"expr": "0.5*x**2", "color": "TEAL"}], [-3, 3, 1], [0, 4.5, 1],
              y=-0.7, w=8, h=4.5),
        Obj(id="shape_3", kind="shape", color="RED",
            payload={"shape": "dot", "size": 1.4, "anchor": {"plot": "plot_2", "x": "u", "y": "0.5*u**2"}}),
    ]
    _show(s, "text_1", "plot_2", "shape_3")
    s.add_step("tween", param="u", value=2.0, duration=3)


def c_fade_then_replace(s):  # electrochemistry: replace a line in place via fade, then write
    s.objects += [
        _t("text_1", "Nernst equation", y=2.9, fs=38),
        _eq("equation_2", r"E = E^\circ - \frac{RT}{nF}\ln Q", y=0.8),
        _t("text_3", "at equilibrium Q = K and E = 0", y=-1.5, fs=28, color="GRAY"),
        _t("text_4", "so  RT ln K = nF E", y=-1.5, fs=28, color="YELLOW"),
    ]
    _show(s, "text_1", "equation_2", "text_3")
    s.add_step("fade", targets=["text_3"], duration=0.6)
    _show(s, "text_4")
    s.add_step("highlight", targets=["equation_2"])


def c_energy_levels(s):  # quantum chemistry: horizontal lines with labels to their right
    s.objects += [_t("text_1", "Hydrogen energy levels", y=3.3, fs=34)]
    ids = ["text_1"]
    for k, (n, y) in enumerate(((1, -2.8), (2, -0.6), (3, 0.6), (4, 1.4))):
        sid, tid = f"shape_{2 + 2 * k}", f"equation_{3 + 2 * k}"
        s.objects += [_shape(sid, "line", -1.0, y, "BLUE", size=5.0),
                      _eq(tid, rf"n={n}", 3.0, y, 0.7)]
        ids += [sid, tid]
    _show(s, *ids)


# ---------------------------------------------------------------------------
# defect scenes
# ---------------------------------------------------------------------------

# --- framing ---------------------------------------------------------------

def d_maxwell_off_right(s):  # long equation centred too far right: its tail leaves the frame
    s.objects += [
        _t("text_1", "Faraday's law", x=-3.5, y=2.8, fs=36),
        _eq("equation_2", r"\oint_{\partial S} \mathbf{E}\cdot d\boldsymbol{\ell} = "
                          r"-\frac{d}{dt}\iint_S \mathbf{B}\cdot d\mathbf{A}", 5.2, 0.0),
    ]
    _show(s, "text_1", "equation_2")


def d_plot_below_frame(s):  # tall plot placed low: its lower part is cut off
    s.objects += [
        _t("text_1", "Exponential decay", y=3.0, fs=36),
        _plot("plot_2", [{"expr": "np.exp(-x)", "color": "ORANGE"}], [0, 5, 1], [0, 1.2, 0.5],
              y=-2.6, w=8, h=5),
    ]
    _show(s, "text_1", "plot_2")


def d_title_above_frame(s):  # title placed at the very top edge
    s.objects += [
        _t("text_1", "Conservation of momentum", y=4.0, fs=40),
        _eq("equation_2", r"m_1 v_1 + m_2 v_2 = m_1 v_1' + m_2 v_2'", y=0.0),
    ]
    _show(s, "text_1", "equation_2")


def d_shape_left_edge(s):  # circle centred near the left edge spills out of frame
    s.objects += [
        _t("text_1", "Gauss's law: spherical surface", x=1.5, y=3.0, fs=32),
        _shape("shape_2", "circle", -6.4, -0.3, "BLUE", size=3.0),
        _eq("equation_3", r"\Phi_E = \frac{Q}{\varepsilon_0}", 2.5, -0.3),
    ]
    _show(s, "text_1", "shape_2", "equation_3")


# --- collision -------------------------------------------------------------

def d_labels_overlap(s):  # two reaction labels placed on top of each other
    s.objects += [
        _t("text_1", "Le Chatelier's principle", y=3.0, fs=36),
        _t("text_2", "Reactants", -0.4, 0.0, 36, "BLUE"),
        _t("text_3", "Products", 0.4, 0.1, 36, "RED"),
    ]
    _show(s, "text_1", "text_2", "text_3")


def d_equation_on_curve(s):  # equation dropped in the middle of the plotted curve
    s.objects += [
        _plot("plot_1", [{"expr": "x**2 - 2", "color": "YELLOW"}], [-3, 3, 1], [-3, 7, 2],
              y=-0.3, w=9, h=5.5),
        _eq("equation_2", r"f(x) = x^2 - 2", -1.7, -0.5, color="WHITE"),
    ]
    _show(s, "plot_1", "equation_2")


def d_text_on_circle_outline(s):  # label straddling a circle's outline
    s.objects += [
        _t("text_1", "Orbit of a satellite", y=3.2, fs=34),
        _shape("shape_2", "circle", 0, -0.5, "TEAL", size=5.0),
        _shape("shape_3", "dot", 0, -0.5, "YELLOW", size=2.0),
        _t("text_4", "geostationary orbit", 2.3, -0.5, 30, "TEAL"),
    ]
    _show(s, "text_1", "shape_2", "shape_3", "text_4")


def d_equations_stacked_too_close(s):  # two tall equations with overlapping fractions
    s.objects += [
        _t("text_1", "Kinetic and potential energy", y=3.0, fs=34),
        _eq("equation_2", r"K = \frac{1}{2} m v^2", y=0.35, scale=1.2),
        _eq("equation_3", r"U = \frac{G M m}{r}", y=-0.35, scale=1.2),
    ]
    _show(s, "text_1", "equation_2", "equation_3")


def d_label_on_x_axis(s):  # axis caption sitting right on the x-axis line
    s.objects += [
        _t("text_1", "Logistic growth", y=3.2, fs=36),
        _plot("plot_2", [{"expr": "4/(1 + np.exp(-(x - 5)))", "color": "GREEN"}], [0, 10, 2], [0, 5, 1],
              y=-0.5, w=9, h=5),
        _t("text_3", "time (days)", 1.0, -3.0, 30, "GRAY"),
    ]
    _show(s, "text_1", "plot_2", "text_3")


# --- legibility ------------------------------------------------------------

def d_tiny_footnote(s):  # footnote font far too small
    s.objects += [
        _t("text_1", "Ohm's law", y=2.5, fs=40),
        _eq("equation_2", r"V = I R", y=0.5, scale=1.4),
        _t("text_3", "valid for ohmic conductors at constant temperature", y=-2.5, fs=12, color="GRAY"),
    ]
    _show(s, "text_1", "equation_2", "text_3")


def d_shrunk_equation(s):  # equation scaled down to a speck
    s.objects += [
        _t("text_1", "Schrodinger equation", y=2.5, fs=40),
        _eq("equation_2", r"i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi", y=0.0, scale=0.12),
    ]
    _show(s, "text_1", "equation_2")


def d_scaled_down_text(s):  # large font size cancelled out by a small scale
    s.objects += [
        _eq("equation_1", r"\det(A - \lambda I) = 0", y=1.5),
        _t("text_2", "characteristic polynomial", y=-0.8, fs=40, scale=0.35, color="YELLOW"),
    ]
    _show(s, "equation_1", "text_2")


# --- timing ----------------------------------------------------------------

def d_fade_before_write(s):  # a label is faded out before it was ever written
    s.objects += [
        _t("text_1", "Half-life", y=2.5, fs=40),
        _eq("equation_2", r"N(t) = N_0\, 2^{-t/T_{1/2}}", y=0.3),
        _t("text_3", "radioactive decay", y=-2.0, fs=30, color="GRAY"),
    ]
    s.add_step("fade", targets=["text_3"])
    _show(s, "text_1", "equation_2", "text_3")


def d_transform_to_missing(s):  # transform target id does not exist
    s.objects += [
        _eq("equation_1", r"\sum_{n=1}^{\infty} \frac{1}{n^2}", y=0.5),
        _t("text_2", "Basel problem", y=2.8, fs=36),
    ]
    _show(s, "text_2", "equation_1")
    s.add_step("transform", targets=["equation_1"], to="equation_9")


def d_tween_undeclared(s):  # tween on a parameter that was never declared
    s.objects += [
        _t("text_1", "Wave on a string", y=3.0, fs=36),
        _plot("plot_2", [{"expr": "np.sin(2*x)", "color": "BLUE"}], [0, 6.3, 1], [-1.5, 1.5, 1],
              y=-0.5, w=9, h=4),
    ]
    _show(s, "text_1", "plot_2")
    s.add_step("tween", param="phase", value=3.14, duration=2)


def d_orphan_object(s):  # an equation is defined but no step ever uses it
    s.objects += [
        _t("text_1", "Entropy", y=2.6, fs=40),
        _eq("equation_2", r"S = k_B \ln \Omega", y=0.5, scale=1.2),
        _eq("equation_3", r"\Delta S \geq \frac{Q}{T}", y=-2.0),
    ]
    _show(s, "text_1", "equation_2")


def d_write_missing_object(s):  # write step references an id that is not in the scene
    s.objects += [
        _t("text_1", "Snell's law", y=2.5, fs=40),
        _eq("equation_2", r"n_1 \sin\theta_1 = n_2 \sin\theta_2", y=0.3),
    ]
    _show(s, "text_1", "equation_2", "text_5")


def d_highlight_before_write(s):  # a highlight fires on a plot not yet drawn
    s.objects += [
        _t("text_1", "Hyperbolic tangent", y=3.1, fs=36),
        _plot("plot_2", [{"expr": "np.tanh(x)", "color": "PINK"}], [-4, 4, 1], [-1.5, 1.5, 1],
              y=-0.5, w=9, h=4.5),
    ]
    s.add_step("highlight", targets=["plot_2"])
    _show(s, "text_1", "plot_2")


# --- compile ---------------------------------------------------------------

def d_latex_unclosed_env(s):  # bmatrix environment opened but never closed
    s.objects += [
        _t("text_1", "Identity matrix", y=2.5, fs=40),
        _eq("equation_2", r"I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1", y=0.0),
    ]
    _show(s, "text_1", "equation_2")


def d_latex_unknown_macro(s):  # misspelt LaTeX command
    s.objects += [
        _t("text_1", "Divergence theorem", y=2.5, fs=40),
        _eq("equation_2", r"\iiint_V \nabbla\cdot\mathbf{F}\,dV = \oint_S \mathbf{F}\cdot d\mathbf{S}", y=0.0),
    ]
    _show(s, "text_1", "equation_2")


def d_unknown_function(s):  # plot expression calls a function that does not exist
    s.objects += [
        _t("text_1", "Error function", y=3.0, fs=36),
        _plot("plot_2", [{"expr": "erf(x)", "color": "YELLOW"}], [-3, 3, 1], [-1.2, 1.2, 0.5],
              y=-0.5, w=9, h=4.5),
    ]
    _show(s, "text_1", "plot_2")


# --- fidelity --------------------------------------------------------------

def d_sawtooth_smoothed(s):  # sawtooth forced smooth with coarse dx: splines overshoot the jumps
    s.objects += [
        _t("text_1", "Sawtooth wave", y=3.2, fs=36),
        _plot("plot_2", [{"expr": "x - np.floor(x)", "color": "YELLOW", "smooth": True, "dx": 0.5}],
              [0, 4, 1], [-0.5, 1.5, 0.5], y=-0.5, w=10, h=4.5),
    ]
    _show(s, "text_1", "plot_2")


def d_ac_aliasing(s):  # 50 Hz-like oscillation sampled far too coarsely
    s.objects += [
        _t("text_1", "Mains voltage", y=3.2, fs=36),
        _plot("plot_2", [{"expr": "np.sin(30*x)", "color": "RED", "dx": 0.25}],
              [0, 6, 1], [-1.5, 1.5, 1], y=-0.5, w=10, h=4.5),
    ]
    _show(s, "text_1", "plot_2")


def d_square_wave_smoothed(s):  # square pulse train with smoothing forced on
    s.objects += [
        _t("text_1", "PWM signal", y=3.2, fs=36),
        _plot("plot_2", [{"expr": "np.where(np.mod(x, 2) < 1, 1.0, 0.0)", "color": "GREEN",
                          "smooth": True, "dx": 0.4}],
              [0, 6, 1], [-0.5, 1.5, 0.5], y=-0.5, w=10, h=4.5),
    ]
    _show(s, "text_1", "plot_2")


# ---------------------------------------------------------------------------

_C = "holdout/clean/"
_D = "holdout/defect/"

CASES: list[Case] = [
    Case(_C + "projectile", c_projectile),
    Case(_C + "bayes_callout", c_bayes_callout),
    Case(_C + "matrix_product", c_matrix_product),
    Case(_C + "gaussian_area", c_gaussian_area),
    Case(_C + "taylor_transform", c_taylor_transform),
    Case(_C + "arrhenius_bullets", c_arrhenius_bullets),
    Case(_C + "unit_circle", c_unit_circle),
    Case(_C + "riemann_sqrt", c_riemann_sqrt),
    Case(_C + "damped_envelope", c_damped_envelope),
    Case(_C + "ideal_gas_isotherm", c_ideal_gas_isotherm),
    Case(_C + "heaviside_auto", c_heaviside_auto),
    Case(_C + "gradient_two_columns", c_gradient_two_columns),
    Case(_C + "pendulum_tween", c_pendulum_tween),
    Case(_C + "fade_then_replace", c_fade_then_replace),
    Case(_C + "energy_levels", c_energy_levels),

    Case(_D + "maxwell_off_right", d_maxwell_off_right, {"framing"}),
    Case(_D + "plot_below_frame", d_plot_below_frame, {"framing"}),
    Case(_D + "title_above_frame", d_title_above_frame, {"framing"}),
    Case(_D + "shape_left_edge", d_shape_left_edge, {"framing"}),
    Case(_D + "labels_overlap", d_labels_overlap, {"collision"}),
    Case(_D + "equation_on_curve", d_equation_on_curve, {"collision"}),
    Case(_D + "text_on_circle_outline", d_text_on_circle_outline, {"collision"}),
    Case(_D + "equations_stacked_too_close", d_equations_stacked_too_close, {"collision"}),
    Case(_D + "label_on_x_axis", d_label_on_x_axis, {"collision"}),
    Case(_D + "tiny_footnote", d_tiny_footnote, {"legibility"}),
    Case(_D + "shrunk_equation", d_shrunk_equation, {"legibility"}),
    Case(_D + "scaled_down_text", d_scaled_down_text, {"legibility"}),
    Case(_D + "fade_before_write", d_fade_before_write, {"timing"}),
    Case(_D + "transform_to_missing", d_transform_to_missing, {"timing"}),
    Case(_D + "tween_undeclared", d_tween_undeclared, {"timing"}),
    Case(_D + "orphan_object", d_orphan_object, {"timing"}),
    Case(_D + "write_missing_object", d_write_missing_object, {"timing"}),
    Case(_D + "highlight_before_write", d_highlight_before_write, {"timing"}),
    Case(_D + "latex_unclosed_env", d_latex_unclosed_env, {"compile"}),
    Case(_D + "latex_unknown_macro", d_latex_unknown_macro, {"compile"}),
    Case(_D + "unknown_function", d_unknown_function, {"compile"}),
    Case(_D + "sawtooth_smoothed", d_sawtooth_smoothed, {"fidelity"}),
    Case(_D + "ac_aliasing", d_ac_aliasing, {"fidelity"}),
    Case(_D + "square_wave_smoothed", d_square_wave_smoothed, {"fidelity"}),
]
