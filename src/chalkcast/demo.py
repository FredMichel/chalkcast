"""Canned scenes for the CLI demo and the tests.

``build_demo_project`` is intentionally LaTeX-free (text + plot + shape) so
``chalkcast demo`` renders out of the box wherever Manim + ffmpeg are present,
without requiring a TeX distribution. ``build_equation_scene`` exercises the
LaTeX/MathTex path and is used by codegen tests.
"""

from __future__ import annotations

from .model import Obj, Project, Scene


def build_demo_project() -> tuple[Project, Scene]:
    """The 'area under y = x²' explainer, no LaTeX required."""
    project = Project(id="proj-demo", quality="low", theme="chalkboard")
    scene = project.new_scene(title="Area under a parabola")

    title = Obj(id="text_1", kind="text", position=(0, 3.0),
                payload={"text": "Area under a parabola", "font_size": 40})
    plot = Obj(id="plot_1", kind="plot", position=(0, -0.8),
               payload={"functions": [{"expr": "x**2", "color": "YELLOW"}],
                        "x_range": [0, 2, 1], "y_range": [0, 4, 1],
                        "width": 6, "height": 4})
    dot = Obj(id="shape_1", kind="shape", position=(1.0, -0.8), color="RED",
              payload={"shape": "dot", "size": 1.0})
    scene.objects += [title, plot, dot]

    scene.add_step("write", targets=["text_1"], duration=1.0, note="show the title")
    scene.add_step("write", targets=["plot_1"], duration=1.5, note="draw axes and the parabola")
    scene.add_step("write", targets=["shape_1"], duration=0.6, note="mark the point x=1")
    scene.add_step("highlight", targets=["shape_1"], duration=0.8, note="emphasize the point")
    scene.add_step("wait", duration=0.5)
    return project, scene


def build_equation_scene() -> Scene:
    """A scene that uses MathTex (requires LaTeX to render)."""
    project = Project(id="proj-eq", theme="chalkboard")
    scene = project.new_scene(title="The integral")
    scene.objects.append(Obj(id="equation_1", kind="equation", position=(0, 1.0),
                             payload={"tex": r"\int_0^1 x^2 \, dx = \tfrac{1}{3}"}))
    scene.add_step("write", targets=["equation_1"], duration=1.0)
    scene.add_step("highlight", targets=["equation_1"], duration=1.0)
    return scene
